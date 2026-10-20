"""
SEED Task 2.6: CSD spatial-hardening probe (LR only, no Mamba). Closes (or
keeps, honestly) the temporal/peripheral EMG yellow-flag from Task 2.5, on the
CSD (surface-Laplacian) transform. CSD attenuates far-field / volume-conducted
signal (including scalp EMG spread), so if the emotion signal survives CSD in
central sites and the temporal ring stops being uniquely strong, the flag
closes.

All features come from the ONE all-62-channel CSD result (seed_csd_features);
channel groups are taken by mask from that result -- CSD is never recomputed
per subset. Same LOSO 15-fold, saga LR, z-score(train-only), segment->trial
subject aggregation as Task 2 / 2.5.

Outputs:
  1. CSD + all 62 + all bands           (total signal survives CSD?)
  2. CSD + theta/alpha/beta (drop gamma)(neural bands stable after CSD?)
  3. CSD + 4 channel groups             (a all62 / b excl temporal-periph /
                                         c only temporal-periph / d only central)
  4. CSD + 55-75 Hz high-EMG-proxy band alone (its standalone discriminability)

Pre-registered readout printed at the end.
"""
import json
import shutil
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import seed2_config as cfg
from seed_csd_features import FEATURES_CACHE_CSD, CH_NAMES

LR_KW = dict(solver="saga", max_iter=200, tol=1e-3, C=cfg.LR_C)  # matches Task 2.5 (verified == lbfgs)

_IDX = {n: i for i, n in enumerate(CH_NAMES)}
TEMPORAL_PERIPHERAL = ['F7','F8','FT7','FT8','T7','T8','TP7','TP8','P7','P8','PO7','PO8','CB1','CB2']
CENTRAL = ['FZ','F1','F2','FCZ','FC1','FC2','CZ','C1','C2','CPZ','CP1','CP2','PZ','P1','P2','POZ','PO3','PO4','OZ']
TEMP_IDX = np.array([_IDX[n] for n in TEMPORAL_PERIPHERAL])
CENTRAL_IDX = np.array([_IDX[n] for n in CENTRAL])
ALL_IDX = np.arange(cfg.N_CH)
EXCL_TEMP_IDX = np.array([i for i in range(cfg.N_CH) if i not in set(TEMP_IDX.tolist())])


def load_csd():
    return {sid: dict(np.load(FEATURES_CACHE_CSD / f"{sid}.npz"))
            for sid in range(1, cfg.scfg.N_SUBJECTS + 1)}


def labels_to_idx(y):
    return np.array([cfg.LABEL_TO_IDX[int(v)] for v in y], dtype=np.int64)


def build_channel_group(sub, ch_idx):
    band = sub["band_logpow"][:, :, ch_idx]; N = band.shape[0]
    X = np.concatenate([band.reshape(N, -1), sub["logvar"][:, ch_idx], sub["mobility"][:, ch_idx]], axis=1)
    return X, sub["seg_label"], sub["seg_trial_id"]


def build_drop_gamma(sub):
    bi = [cfg.BAND_ORDER.index(b) for b in ("theta", "alpha", "beta")]
    band = sub["band_logpow"][:, bi, :]; N = band.shape[0]
    X = np.concatenate([band.reshape(N, -1), sub["logvar"], sub["mobility"]], axis=1)
    return X, sub["seg_label"], sub["seg_trial_id"]


def build_high_emg(sub):
    return sub["high_emg_logpow"], sub["seg_label"], sub["seg_trial_id"]


def run_loso(subs, splits, build_fn):
    subj_bacc, seg_bacc = [], []
    for fold in splits.values():
        test_sid = int(fold["test"][0]); train_ids = [int(s) for s in fold["train"]]
        parts = [build_fn(subs[s]) for s in train_ids]
        Xtr = np.concatenate([p[0] for p in parts]); ytr = labels_to_idx(np.concatenate([p[1] for p in parts]))
        mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-8
        clf = LogisticRegression(**LR_KW).fit((Xtr - mu) / sd, ytr)
        Xte_raw, yte_raw, tids = build_fn(subs[test_sid])
        Xte = (Xte_raw - mu) / sd; yte = labels_to_idx(yte_raw)
        proba = clf.predict_proba(Xte)
        seg_bacc.append(balanced_accuracy_score(yte, proba.argmax(1)))
        tt, tp = [], []
        for u in np.unique(tids):
            m = tids == u; tp.append(proba[m].mean(0).argmax()); tt.append(yte[m][0])
        subj_bacc.append(balanced_accuracy_score(tt, tp))
    return [float(x) for x in subj_bacc], {"mean": float(np.mean(subj_bacc)),
            "std": float(np.std(subj_bacc)), "seg_mean": float(np.mean(seg_bacc))}


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_seed_task2_6_csd_probe"
    run_dir.mkdir(parents=True)
    splits = json.load(open(cfg.SPLITS_JSON))
    subs = load_csd()
    t0 = time.time()

    res = {}
    folds, agg = run_loso(subs, splits, lambda s: build_channel_group(s, ALL_IDX))
    res["csd_all62_allbands"] = {**agg, "per_fold": folds}
    print(f"[1] CSD all62 all-bands: {agg['mean']:.3f} +- {agg['std']:.3f} (seg {agg['seg_mean']:.3f})")

    folds, agg = run_loso(subs, splits, build_drop_gamma)
    res["csd_drop_gamma"] = {**agg, "per_fold": folds}
    print(f"[2] CSD theta/alpha/beta (drop gamma): {agg['mean']:.3f} +- {agg['std']:.3f} (seg {agg['seg_mean']:.3f})")

    res["csd_channel_groups"] = {}
    for name, idx in {"a_all62": ALL_IDX, "b_exclude_temporal_peripheral": EXCL_TEMP_IDX,
                      "c_only_temporal_peripheral": TEMP_IDX, "d_only_central": CENTRAL_IDX}.items():
        folds, agg = run_loso(subs, splits, lambda s, i=idx: build_channel_group(s, i))
        res["csd_channel_groups"][name] = {**agg, "n_channels": int(len(idx)), "per_fold": folds}
        print(f"[3] CSD group {name} (n_ch={len(idx)}): {agg['mean']:.3f} +- {agg['std']:.3f} (seg {agg['seg_mean']:.3f})")

    folds, agg = run_loso(subs, splits, build_high_emg)
    res["csd_high_emg_proxy_55_75"] = {**agg, "per_fold": folds}
    print(f"[4] CSD high-EMG-proxy 55-75Hz alone: {agg['mean']:.3f} +- {agg['std']:.3f} (seg {agg['seg_mean']:.3f})")

    # ---- pre-registered readout ----
    all62 = res["csd_all62_allbands"]["mean"]
    dropg = res["csd_drop_gamma"]["mean"]
    c = res["csd_channel_groups"]["c_only_temporal_peripheral"]["mean"]
    d = res["csd_channel_groups"]["d_only_central"]["mean"]
    hemg = res["csd_high_emg_proxy_55_75"]["mean"]
    temporal_still_strongest = c > max(all62, d) + 0.0  # c beats both all62 and central
    verdict = []
    if all62 > 0.50 and abs(all62 - dropg) < 0.06 and not (c > d + 0.03):
        verdict.append("GREEN: signal survives CSD, gamma-independent, temporal ring no longer uniquely strong -> yellow flag CLOSED, proceed to Mamba.")
    if all62 > 0.45 and c > d + 0.03:
        verdict.append("YELLOW-KEPT: signal survives CSD but temporal/peripheral still strongest & central weaker -> can proceed to Mamba but paper must caveat temporal contribution as possibly mixed neural/EMG.")
    if all62 < 0.42:
        verdict.append("RED: signal collapses under CSD (~chance) -> stop, reconsider before any mechanism narrative.")
    if hemg < 0.42 and dropg > 0.48:
        verdict.append("STRONG-GREEN support: standalone 55-75Hz high-EMG-proxy is low while theta/alpha/beta stays high -> further supports non-EMG-dominated signal.")
    if not verdict:
        verdict.append("AMBIGUOUS: does not cleanly match any pre-registered branch; inspect numbers manually.")

    report = {"lr_solver": LR_KW,
              "csd": "mne.preprocessing.compute_current_source_density on all 62 ch (standard_1005 montage; "
                     "CB1/CB2 -> OI1/OI2 positions); channel groups masked from the single 62-ch CSD result",
              "channel_groups": {"temporal_peripheral": TEMPORAL_PERIPHERAL, "central": CENTRAL},
              "task2_5_precsd_reference": {"all62": 0.536, "drop_gamma": 0.510,
                                           "c_temporal": 0.560, "d_central": 0.461},
              "results": res, "preregistered_readout": verdict}
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(cfg.RESULTS_DIR / "seed_task2_6_csd_probe.json", "w") as f:
        json.dump(report, f, indent=2)
    shutil.copy(cfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")
    print("\n--- pre-registered readout ---")
    for v in verdict:
        print(v)
    print(f"\nelapsed {time.time()-t0:.1f}s; archived {run_dir}")
    return report


if __name__ == "__main__":
    main()
