"""
Task 4 ERP: a SECOND, stricter round of artifact flagging.

Purpose: Stage 1 (erp_stage1_pipeline.py) used a k=5 pooled-MAD threshold
that turned out extremely lenient (0.055% of trials flagged, 47/53 subjects
with zero flags) -- too few flags for the MDD-vs-HC group test to have any
power. This script adds three INDEPENDENT, stricter flag sets on top of the
existing epoch tensors, purely as new flag arrays -- it does NOT touch
epochs_cache/*.npz (read-only input), does NOT drop any trial, and does NOT
do any modeling or band attribution (that stays Stage 2).

Input: epochs_cache/<sid>.npz's `bc_epochs` (baseline-corrected epoch
tensor, (n_trials, 128, 200), float32, Volts -- produced by Stage 1,
read-only here).

Units and definitions (all computed on the 128 EEG channels only, per
trial, over the whole 200-sample epoch = -200..+596ms):
  - ptp (peak-to-peak) of a channel within a trial = max(x) - min(x) over
    the 200 time samples of that channel, in microvolts (uV = V * 1e6).
  - abs amplitude of a channel within a trial = max(|x|) over the 200 time
    samples of that channel, in microvolts.
  - a trial's scalar summary for each metric = the MAX over all 128
    channels (i.e. "any channel exceeds X" -> flagged), matching the
    "任一 EEG 通道" wording of the spec.

Three independent flag sets (a trial can be flagged by any subset of
these; they are NOT combined into one "final" flag here):
  1. ptp_100uv:  trial-level max-over-channels ptp_uv > 100 uV.
  2. abs_100uv:  trial-level max-over-channels abs_uv > 100 uV.
  3. mad_k3:     same underlying metric as (1) (trial-level max-ptp, in
                 uV), but thresholded with a k=3 pooled-log-MAD threshold
                 (same style/formula as Stage 1's k=5 threshold, just a
                 stricter k). The threshold is derived from the FULL
                 POOLED distribution across all 53 subjects x 480 trials
                 BEFORE any subject-level flag is assigned, so it cannot be
                 contaminated by group membership.

Run with the base conda env (MNE 1.12.1) purely for consistency with the
rest of Task 4 -- this script itself only needs numpy/scipy/pandas, never
mamba3.
"""
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

import erp_config as ecfg

FLAGS_DIR = ecfg.TASK4_DIR / "artifact_flags"
FLAGS_DIR.mkdir(parents=True, exist_ok=True)

FIXED_THRESH_UV = 100.0
STRICT_MAD_K = 3.0


def mad_log_threshold(values_uv, k):
    """Same formula as Stage 1's mad_log_threshold, applied to values in uV."""
    log_v = np.log(values_uv + 1e-12)
    med = np.median(log_v)
    mad = np.median(np.abs(log_v - med)) * 1.4826 + 1e-12
    return med + k * mad


def per_subject_metrics(sid):
    d = np.load(ecfg.EPOCHS_CACHE / f"{sid}.npz")
    bc_epochs = d["bc_epochs"]  # (n_trials, 128, 200), Volts
    conditions = d["conditions"]

    ptp_per_ch_uv = np.ptp(bc_epochs, axis=2) * 1e6           # (n_trials, 128)
    abs_per_ch_uv = np.max(np.abs(bc_epochs), axis=2) * 1e6   # (n_trials, 128)

    trial_max_ptp_uv = ptp_per_ch_uv.max(axis=1)  # (n_trials,)
    trial_max_abs_uv = abs_per_ch_uv.max(axis=1)  # (n_trials,)

    return {
        "subject_id": sid, "conditions": conditions,
        "trial_max_ptp_uv": trial_max_ptp_uv, "trial_max_abs_uv": trial_max_abs_uv,
        "n_trials": bc_epochs.shape[0],
    }


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ecfg.RUNS_DIR / f"{ts}_artifact_flags_strict"
    run_dir.mkdir(parents=True)

    manifest = pd.read_csv(ecfg.MANIFEST_128REST, dtype={"subject_id": str})
    label_of = dict(zip(manifest["subject_id"], manifest["label"]))
    type_of = dict(zip(manifest["subject_id"], manifest["type"]))

    sids = sorted(p.stem for p in ecfg.EPOCHS_CACHE.glob("*.npz"))
    print(f"loading cached epoch tensors for {len(sids)} subjects (read-only)...")
    t0 = time.time()
    subjects = {sid: per_subject_metrics(sid) for sid in sids}
    print(f"loaded in {time.time() - t0:.1f}s")

    # ---- pool trial_max_ptp_uv across ALL subjects/trials, derive k=3 MAD threshold ----
    all_ptp_uv = np.concatenate([d["trial_max_ptp_uv"] for d in subjects.values()])
    mad_k3_thresh_log = mad_log_threshold(all_ptp_uv, STRICT_MAD_K)
    mad_k3_thresh_uv = float(np.exp(mad_k3_thresh_log))
    print(f"pooled k={STRICT_MAD_K} MAD threshold (log-ptp): {mad_k3_thresh_log:.4f} "
          f"(ptp={mad_k3_thresh_uv:.2f} uV)")

    log_lines = [f"pooled k={STRICT_MAD_K} MAD threshold: {mad_k3_thresh_uv:.2f} uV"]
    per_subject_report = {"ptp_100uv": [], "abs_100uv": [], "mad_k3": []}

    for sid, d in subjects.items():
        trial_max_ptp_uv = d["trial_max_ptp_uv"]
        trial_max_abs_uv = d["trial_max_abs_uv"]
        conditions = d["conditions"]

        flags = {
            "ptp_100uv": trial_max_ptp_uv > FIXED_THRESH_UV,
            "abs_100uv": trial_max_abs_uv > FIXED_THRESH_UV,
            "mad_k3": np.log(trial_max_ptp_uv + 1e-12) > mad_k3_thresh_log,
        }

        out_path = FLAGS_DIR / f"{sid}.npz"
        np.savez_compressed(
            out_path,
            trial_max_ptp_uv=trial_max_ptp_uv, trial_max_abs_uv=trial_max_abs_uv,
            conditions=conditions,
            ptp_100uv_flag=flags["ptp_100uv"], abs_100uv_flag=flags["abs_100uv"],
            mad_k3_flag=flags["mad_k3"],
        )

        for name, flag in flags.items():
            rate = float(flag.mean())
            cond_rates = {}
            for letter in ("f", "h", "s"):
                m = conditions == letter
                cond_rates[letter] = float(flag[m].mean()) if m.sum() else None
            per_subject_report[name].append({
                "subject_id": sid, "label": int(label_of[sid]), "type": type_of[sid],
                "n_trials": d["n_trials"], "n_flagged": int(flag.sum()),
                "rejection_rate": rate, "condition_rejection_rate": cond_rates,
            })
        line = (f"{sid} ({type_of[sid]}): ptp100={int(flags['ptp_100uv'].sum())} "
                f"abs100={int(flags['abs_100uv'].sum())} madk3={int(flags['mad_k3'].sum())} "
                f"/ {d['n_trials']} trials")
        print(line); log_lines.append(line)

    # ---- aggregate + MDD-vs-HC group test, per flag set ----
    aggregate = {}
    for name, rows in per_subject_report.items():
        rates = np.array([r["rejection_rate"] for r in rows])
        mdd_rates = np.array([r["rejection_rate"] for r in rows if r["type"] == "MDD"])
        hc_rates = np.array([r["rejection_rate"] for r in rows if r["type"] == "HC"])
        u_stat, p_value = stats.mannwhitneyu(mdd_rates, hc_rates, alternative="two-sided")
        total_flagged = sum(r["n_flagged"] for r in rows)
        total_trials = sum(r["n_trials"] for r in rows)
        aggregate[name] = {
            "total_flagged": total_flagged, "total_trials": total_trials,
            "overall_rejection_rate": total_flagged / total_trials,
            "per_subject_rate": {
                "min": float(rates.min()), "median": float(np.median(rates)), "max": float(rates.max()),
                "mean": float(rates.mean()),
            },
            "n_subjects_zero_flags": int((rates == 0).sum()),
            "mdd_vs_hc": {
                "mdd_mean": float(mdd_rates.mean()), "mdd_median": float(np.median(mdd_rates)), "n_mdd": len(mdd_rates),
                "hc_mean": float(hc_rates.mean()), "hc_median": float(np.median(hc_rates)), "n_hc": len(hc_rates),
                "mannwhitney_u": float(u_stat), "p_value": float(p_value),
            },
        }

    splits_check = json.load(open(ecfg.SPLITS_JSON))
    split_ids = set()
    for seed_splits in splits_check.values():
        for fold_def in seed_splits.values():
            split_ids.update(fold_def["train"]); split_ids.update(fold_def["test"])

    report = {
        "definitions": {
            "ptp_100uv": "trial flagged if max-over-128-channels peak-to-peak (max-min over 200 samples) > 100 uV",
            "abs_100uv": "trial flagged if max-over-128-channels max(|x|) over 200 samples > 100 uV",
            "mad_k3": f"trial flagged if log(trial max-ptp uV) > pooled median(log ptp)+{STRICT_MAD_K}*MAD "
                      f"(threshold = {mad_k3_thresh_uv:.2f} uV, derived across all {len(sids)} subjects "
                      "before any per-subject flag was assigned)",
        },
        "mad_k3_threshold_uv": mad_k3_thresh_uv,
        "n_subjects": len(sids),
        "splits_128rest_subject_set_matches_erp": split_ids == set(sids),
        "aggregate": aggregate,
        "per_subject": per_subject_report,
    }

    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(run_dir / "artifact_flags_strict.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(ecfg.RESULTS_DIR / "erp_artifact_flags_strict_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n=== strict artifact flagging summary (n={len(sids)} subjects) ===")
    for name, agg in aggregate.items():
        print(f"[{name}] overall_rejection_rate={agg['overall_rejection_rate']:.4f} "
              f"({agg['total_flagged']}/{agg['total_trials']}) "
              f"per_subject_rate={agg['per_subject_rate']} "
              f"n_subjects_zero_flags={agg['n_subjects_zero_flags']} "
              f"MDD_mean={agg['mdd_vs_hc']['mdd_mean']:.4f} HC_mean={agg['mdd_vs_hc']['hc_mean']:.4f} "
              f"p={agg['mdd_vs_hc']['p_value']:.4f}")
    print(f"splits_128rest.json subject set matches ERP: {report['splits_128rest_subject_set_matches_erp']}")
    print(f"per-subject flag arrays written to {FLAGS_DIR}/<sid>.npz (new dir, epochs_cache untouched)")
    print(f"archived to {run_dir}")
    return report


if __name__ == "__main__":
    main()
