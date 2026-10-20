"""
SEED Task 2.5: gamma/EMG probe (LR only, no Mamba). Answers how much of the
0.539 subject-level emotion signal could be emotion-related facial/neck EMG
vs genuine neural rhythm. All LOSO 15-fold, subject-level aggregation, splits
reused READ-ONLY. Same LR + z-score + segment->trial aggregation as Task 2.

Probe 1 -- frequency-ceiling scan: gamma split into 30-45 Hz vs 45-75 Hz
  (features from seed_gamma_subbands cache), each fed to LR alone. EMG power
  rises toward higher frequencies -> if bal_acc INCREASES from low to high
  gamma, EMG suspicion; if it concentrates in low gamma and drops in the high
  band, more neural.

Probe 2 -- spatial / EMG-channel control (analogue of MODMA's EOG spatial
  control). EMG contamination is heaviest at temporal / outermost electrodes
  (near masticatory/neck muscle); neural emotion signal is more fronto-central
  / centro-parieto-occipital. Four channel groups, full feature restricted to
  each: (a) all 62; (b) exclude temporal/peripheral; (c) only temporal/
  peripheral; (d) only fronto-central + centro-parieto-occipital midline.
  Signal only in (c) and gone in (b) -> EMG; signal persists in (d) -> neural.

Probe 3 -- gamma-drop robustness: remove the gamma band entirely (keep
  theta/alpha/beta log-power + temporal stats), rerun. If bal_acc stays clearly
  >0.333 (say >0.45) the signal body is in neural bands and gamma is only a
  top-up -> mechanism claim safe.

Channel groups use the real SEED 62-channel order (channel-order.xlsx),
indices 0-based. Group membership is topographic judgment, documented below;
not tuned to the labels (no circularity).
"""
import json
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import seed2_config as cfg

# ---- LR solver (see convergence test in STATUS: lbfgs needs ~457 iters/fit
# = ~7.6 min on the 372-dim full set; saga converges far faster in wall time
# with matching bal_acc). Set after the timing/agreement check. ----
LR_KW = dict(solver="saga", max_iter=200, tol=1e-3, C=cfg.LR_C)

# SEED 62-channel order (0-based indices), from channel-order.xlsx
CH_NAMES = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
            'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
            'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7',
            'P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8',
            'CB1','O1','OZ','O2','CB2']
_IDX = {n: i for i, n in enumerate(CH_NAMES)}

# temporal + outermost/peripheral electrodes (over/near temporalis, masseter,
# and posterior neck muscle) -- the EMG-prone ring
TEMPORAL_PERIPHERAL = ['F7','F8','FT7','FT8','T7','T8','TP7','TP8','P7','P8','PO7','PO8','CB1','CB2']
# fronto-central + centro-parieto-occipital midline/para-midline (away from
# the muscle ring) -- where neural emotion signal is expected
CENTRAL = ['FZ','F1','F2','FCZ','FC1','FC2','CZ','C1','C2','CPZ','CP1','CP2','PZ','P1','P2','POZ','PO3','PO4','OZ']

TEMP_IDX = np.array([_IDX[n] for n in TEMPORAL_PERIPHERAL])
CENTRAL_IDX = np.array([_IDX[n] for n in CENTRAL])
ALL_IDX = np.arange(cfg.N_CH)
EXCL_TEMP_IDX = np.array([i for i in range(cfg.N_CH) if i not in set(TEMP_IDX.tolist())])


def load_main():
    return {sid: dict(np.load(cfg.FEATURES_CACHE / f"{sid}.npz"))
            for sid in range(1, cfg.scfg.N_SUBJECTS + 1)}


def load_gamma():
    return {sid: dict(np.load(cfg.FEATURES_CACHE / f"{sid}_gamma.npz"))
            for sid in range(1, cfg.scfg.N_SUBJECTS + 1)}


def labels_to_idx(y):
    return np.array([cfg.LABEL_TO_IDX[int(v)] for v in y], dtype=np.int64)


# ---- feature builders: each returns (X, seg_label, seg_trial_id) for a subject ----
def build_channel_group(sub, ch_idx):
    """full feature (band_logpow 4xC + logvar C + mobility C) on a channel subset."""
    band = sub["band_logpow"][:, :, ch_idx]           # (N, 4, C)
    N = band.shape[0]
    X = np.concatenate([band.reshape(N, -1), sub["logvar"][:, ch_idx], sub["mobility"][:, ch_idx]], axis=1)
    return X, sub["seg_label"], sub["seg_trial_id"]


def build_drop_gamma(sub):
    """theta/alpha/beta log-power (drop gamma) + logvar + mobility, all 62 ch."""
    bi = [cfg.BAND_ORDER.index(b) for b in ("theta", "alpha", "beta")]
    band = sub["band_logpow"][:, bi, :]
    N = band.shape[0]
    X = np.concatenate([band.reshape(N, -1), sub["logvar"], sub["mobility"]], axis=1)
    return X, sub["seg_label"], sub["seg_trial_id"]


def build_gamma_subband(sub, key):
    return sub[key], sub["seg_label"], sub["seg_trial_id"]


def run_loso(subs, splits, build_fn):
    subj_bacc, seg_bacc = [], []
    for fold in splits.values():
        test_sid = int(fold["test"][0]); train_ids = [int(s) for s in fold["train"]]
        parts = [build_fn(subs[s]) for s in train_ids]
        Xtr = np.concatenate([p[0] for p in parts])
        ytr = labels_to_idx(np.concatenate([p[1] for p in parts]))
        mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-8
        Xtr = (Xtr - mu) / sd
        clf = LogisticRegression(**LR_KW).fit(Xtr, ytr)

        Xte_raw, yte_raw, tids = build_fn(subs[test_sid])
        Xte = (Xte_raw - mu) / sd; yte = labels_to_idx(yte_raw)
        proba = clf.predict_proba(Xte)
        seg_bacc.append(balanced_accuracy_score(yte, proba.argmax(1)))
        t_true, t_pred = [], []
        for u in np.unique(tids):
            m = tids == u; t_pred.append(proba[m].mean(0).argmax()); t_true.append(yte[m][0])
        subj_bacc.append(balanced_accuracy_score(t_true, t_pred))
    return ([float(x) for x in subj_bacc],
            {"mean": float(np.mean(subj_bacc)), "std": float(np.std(subj_bacc)),
             "seg_mean": float(np.mean(seg_bacc))})


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_seed_task2_5_emg_probe"
    run_dir.mkdir(parents=True)
    splits = json.load(open(cfg.SPLITS_JSON))
    main_cache = load_main()
    gamma_cache = load_gamma()

    t0 = time.time()
    results = {"probe1_freq_ceiling": {}, "probe2_spatial": {}, "probe3_drop_gamma": {}}

    # probe 1
    for key in ("gamma_lo_30_45", "gamma_hi_45_75"):
        folds, agg = run_loso(gamma_cache, splits, lambda s, k=key: build_gamma_subband(s, k))
        results["probe1_freq_ceiling"][key] = {**agg, "per_fold": folds}
        print(f"[probe1] {key}: bal_acc={agg['mean']:.3f} +- {agg['std']:.3f} (seg {agg['seg_mean']:.3f})")

    # probe 2
    groups = {"a_all62": ALL_IDX, "b_exclude_temporal_peripheral": EXCL_TEMP_IDX,
              "c_only_temporal_peripheral": TEMP_IDX, "d_only_central": CENTRAL_IDX}
    for name, idx in groups.items():
        folds, agg = run_loso(main_cache, splits, lambda s, i=idx: build_channel_group(s, i))
        results["probe2_spatial"][name] = {**agg, "n_channels": int(len(idx)), "per_fold": folds}
        print(f"[probe2] {name} (n_ch={len(idx)}): bal_acc={agg['mean']:.3f} +- {agg['std']:.3f} (seg {agg['seg_mean']:.3f})")

    # probe 3
    folds, agg = run_loso(main_cache, splits, build_drop_gamma)
    results["probe3_drop_gamma"]["theta_alpha_beta_no_gamma"] = {**agg, "per_fold": folds}
    print(f"[probe3] drop-gamma (theta+alpha+beta): bal_acc={agg['mean']:.3f} +- {agg['std']:.3f} (seg {agg['seg_mean']:.3f})")

    report = {
        "lr_solver": LR_KW,
        "channel_groups": {
            "temporal_peripheral": TEMPORAL_PERIPHERAL, "central": CENTRAL,
            "note": "0-based indices from channel-order.xlsx; topographic judgment, label-blind",
        },
        "protocol": "LOSO 15-fold, subject-level 3-class balanced_acc (chance 0.333); "
                    "segments->trial mean-prob, bal_acc over 45 trials/subject",
        "results": results,
        "task2_full_reference_bal_acc": 0.539,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(cfg.RESULTS_DIR / "seed_task2_5_emg_probe.json", "w") as f:
        json.dump(report, f, indent=2)
    shutil.copy(cfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")
    print(f"\nelapsed {time.time()-t0:.1f}s; archived {run_dir}")
    return report


if __name__ == "__main__":
    main()
