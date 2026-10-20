"""
SEED Task 2 stage B: LOSO logistic-regression signal confirmation (3-class
pos/neu/neg), plus the two autopsies (band attribution + bad-channel/artifact
bookkeeping). Reuses splits_seed_loso.json READ-ONLY. No Mamba.

Pipeline per fold (LOSO, 15 folds from splits_seed_loso.json):
  - train = 14 subjects' segments, test = 1 held-out subject's segments.
  - z-score EACH feature dimension with mean/std computed on TRAIN segments
    only, applied unchanged to test (no test statistics leak in). Each
    feature = one channel's band-log-power or temporal stat, so this is the
    per-channel z-score with train-fold-only statistics.
  - multinomial LogisticRegression on segment-level features.
  - predict test segments -> aggregate to TRIAL level (mean class-prob over
    a trial's segments -> argmax) -> subject-level balanced_accuracy over
    that subject's 45 trials (15/class). (segment-level bal_acc also logged.)

Reported: subject-level 3-class balanced_acc mean+-std over the 15 folds,
for the FULL feature set and for each single band (theta/alpha/beta/gamma)
= band attribution. chance = 0.333.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import seed2_config as cfg


def load_all_subjects():
    subs = {}
    for sid in range(1, cfg.scfg.N_SUBJECTS + 1):
        d = np.load(cfg.FEATURES_CACHE / f"{sid}.npz")
        subs[sid] = {k: d[k] for k in d.files}
    return subs


def build_feature_matrix(sub, feature_set):
    """Returns (N, D) feature matrix for the requested feature_set:
    'full' = band-log-power(4x62) + logvar(62) + mobility(62) = 372 dims;
    a band name = that band's log-power (62 dims) only."""
    band = sub["band_logpow"]  # (N, 4, 62)
    if feature_set == "full":
        N = band.shape[0]
        return np.concatenate([band.reshape(N, -1), sub["logvar"], sub["mobility"]], axis=1)
    bi = cfg.BAND_ORDER.index(feature_set)
    return band[:, bi, :]


def labels_to_idx(y):
    return np.array([cfg.LABEL_TO_IDX[int(v)] for v in y], dtype=np.int64)


def run_feature_set(subs, splits, feature_set):
    """Returns per-fold subject-level balanced_acc list + segment-level list."""
    subj_bacc, seg_bacc = [], []
    for fold_name, fold in splits.items():
        test_sid = int(fold["test"][0])
        train_ids = [int(s) for s in fold["train"]]

        Xtr = np.concatenate([build_feature_matrix(subs[s], feature_set) for s in train_ids])
        ytr = labels_to_idx(np.concatenate([subs[s]["seg_label"] for s in train_ids]))

        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-8
        Xtr = (Xtr - mu) / sd

        clf = LogisticRegression(C=cfg.LR_C, max_iter=cfg.LR_MAX_ITER)
        clf.fit(Xtr, ytr)

        Xte = (build_feature_matrix(subs[test_sid], feature_set) - mu) / sd
        yte = labels_to_idx(subs[test_sid]["seg_label"])
        proba = clf.predict_proba(Xte)                       # (n_seg, 3)
        seg_pred = proba.argmax(axis=1)
        seg_bacc.append(balanced_accuracy_score(yte, seg_pred))

        # aggregate segments -> trial (mean prob over a trial's segments)
        trial_ids = subs[test_sid]["seg_trial_id"]
        uniq = np.unique(trial_ids)
        t_true, t_pred = [], []
        for tid in uniq:
            m = trial_ids == tid
            t_pred.append(proba[m].mean(axis=0).argmax())
            t_true.append(yte[m][0])
        subj_bacc.append(balanced_accuracy_score(t_true, t_pred))
    return subj_bacc, seg_bacc


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_seed_task2_lr_loso"
    run_dir.mkdir(parents=True)

    splits = json.load(open(cfg.SPLITS_JSON))
    subs = load_all_subjects()
    manifest = pd.read_csv(cfg.MANIFEST_CSV)

    t0 = time.time()
    results = {}
    for fs in ["full"] + cfg.BAND_ORDER:
        subj_bacc, seg_bacc = run_feature_set(subs, splits, fs)
        results[fs] = {
            "subject_level_balanced_acc": {
                "mean": float(np.mean(subj_bacc)), "std": float(np.std(subj_bacc)),
                "per_fold": [float(x) for x in subj_bacc],
            },
            "segment_level_balanced_acc": {
                "mean": float(np.mean(seg_bacc)), "std": float(np.std(seg_bacc)),
            },
        }
        print(f"[{fs:6s}] subject-level bal_acc = {np.mean(subj_bacc):.3f} +- {np.std(subj_bacc):.3f} "
              f"(segment-level {np.mean(seg_bacc):.3f})")

    # ---- autopsy: bad-channel / artifact bookkeeping + gamma fraction ----
    gamma_fracs = {}
    for sid in range(1, cfg.scfg.N_SUBJECTS + 1):
        d = np.load(cfg.FEATURES_CACHE / f"{sid}.npz")
        gamma_fracs[sid] = float(d["gamma_frac"].mean())
    stageA = json.load(open(cfg.RESULTS_DIR / "seed_task2_stageA_summary.json"))
    bad_by_subject = {r["subject_id"]: r["n_bad_union"] for r in stageA}
    autopsy = {
        "gamma_power_fraction_per_subject": gamma_fracs,
        "gamma_power_fraction_overall_mean": float(np.mean(list(gamma_fracs.values()))),
        "band_attribution": {b: results[b]["subject_level_balanced_acc"]["mean"] for b in cfg.BAND_ORDER},
        "bad_channels_per_subject": bad_by_subject,
        "bad_channels_stats": {"min": min(bad_by_subject.values()),
                               "median": float(np.median(list(bad_by_subject.values()))),
                               "max": max(bad_by_subject.values()),
                               "mean": float(np.mean(list(bad_by_subject.values())))},
        "segment_rejection": "segments dropped only as sub-window tails (design); no artifact-based "
                             "segment rejection at this LR stage -- bad channels flagged (union counts) "
                             "not removed/interpolated.",
    }

    report = {
        "protocol": "LOSO 15-fold, subject-level 3-class balanced_acc (chance=0.333); "
                    "segments aggregated to trial by mean class-prob, bal_acc over 45 trials/subject",
        "n_features_full": 372, "feature_blocks": "band_logpow(4x62) + logvar(62) + mobility(62)",
        "results": results,
        "autopsy": autopsy,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(cfg.RESULTS_DIR / "seed_task2_lr_loso.json", "w") as f:
        json.dump(report, f, indent=2)
    shutil.copy(cfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")  # provenance copy (read-only source)

    print(f"\nfull subject-level bal_acc: {results['full']['subject_level_balanced_acc']['mean']:.3f} "
          f"+- {results['full']['subject_level_balanced_acc']['std']:.3f}")
    print(f"gamma power fraction (overall mean): {autopsy['gamma_power_fraction_overall_mean']:.3f}")
    print(f"elapsed {time.time()-t0:.1f}s; archived {run_dir}")
    return report


if __name__ == "__main__":
    main()
