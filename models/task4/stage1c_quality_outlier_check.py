"""
Task 4 ERP Stage 1c: objective signal-quality QC check.

Trigger: Stage 1b (artifact_flags_strict.py) flagged subject 02030009 as an
extreme outlier under all three stricter artifact-flag criteria (100% of
trials under ptp_100uv and abs_100uv, 78.3% under mad_k3). This step
determines, with a PRE-REGISTERED rule fixed before any individual result
below is examined, whether 02030009 (and, applying the identical rule to
all 53 subjects, anyone else) should be treated as an objective
signal-quality outlier.

No modeling, no trial deletion, no changes to splits_128rest.json or to any
existing cached array (epochs_cache/*.npz, artifact_flags/*.npz are
READ-ONLY inputs here). Output is a QC judgment + comparison figures + an
exclusion list (possibly empty).

--- Pre-registered protocol (fixed before results were examined) ---

1. Statistical outlier criterion, applied identically to all 53 subjects,
   no subject singled out a priori: for each of the 3 Stage-1b flag sets
   (ptp_100uv, abs_100uv, mad_k3), a subject's per-trial REJECTION RATE is
   a statistical outlier if it exceeds
       pooled_median(rate across 53 subjects) + 3 * MAD(rate) * 1.4826
   (same scaled-MAD convention as bad-channel detection and the Stage 1
   artifact threshold elsewhere in this project). Applied on the raw rate,
   not log -- these rates already span [0,1] and include exact zeros, so a
   log transform isn't meaningful here (unlike the amplitude thresholds).

2. Visual check, for every statistical outlier found in step 1, against
   ONE representative non-outlier HC subject (closest-to-median rates
   among non-outlier HCs) for comparison: butterfly plot (all 128 channels
   overlaid) of a "typical" trial (subject's median max-ptp trial) and the
   "worst" trial (subject's max max-ptp trial), a per-trial max-ptp trace
   across all 480 trials with the fixed/data-driven thresholds overlaid,
   and a per-channel mean-ptp profile (128 channels) to see whether any
   elevation is diffuse (whole-scalp) or channel-localized.

3. Exclusion rule (fixed before any visual check result is known):
   a subject is marked "primary analysis excluded by objective
   signal-quality criterion; sensitivity analysis included" IF AND ONLY IF
   BOTH (a) statistical outlier on >=1 flag set in step 1, AND (b) the
   visual check in step 2 confirms a genuine, diffuse quality problem
   (not a plausible real neural response). If only (a) or only (b) holds,
   the subject is KEPT, no exclusion, unqualified.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import erp_config as ecfg

FLAGS_DIR = ecfg.TASK4_DIR / "artifact_flags"
FLAG_NAMES = ["ptp_100uv", "abs_100uv", "mad_k3"]
MAD_K = 3.0
TIME_MS = (np.arange(ecfg.EPOCH_LEN) - ecfg.PRE_SAMPLES) * (1000.0 / ecfg.FS)  # -200..+596ms
FIXED_100UV_THRESH = 100.0
MAD_K3_AMP_THRESH_UV = 945.63  # from Stage 1b's pooled k=3 log-MAD threshold, for reference lines only


def load_subject_flags(sid):
    d = np.load(FLAGS_DIR / f"{sid}.npz")
    return {
        "trial_max_ptp_uv": d["trial_max_ptp_uv"],
        "trial_max_abs_uv": d["trial_max_abs_uv"],
        "flags": {name: d[f"{name}_flag"] for name in FLAG_NAMES},
    }


def scaled_mad_threshold(rates, k):
    med = np.median(rates)
    mad = np.median(np.abs(rates - med)) * 1.4826
    return med, mad, med + k * mad


def butterfly_panel(ax, epoch_uv, title):
    # epoch_uv: (128, 200) microvolts
    for ch in range(epoch_uv.shape[0]):
        ax.plot(TIME_MS, epoch_uv[ch], lw=0.4, alpha=0.5, color="steelblue")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ms"); ax.set_ylabel("uV")


def trial_trace_panel(ax, trial_max_ptp_uv, title):
    ax.plot(trial_max_ptp_uv, lw=0.7, color="darkorange")
    ax.axhline(FIXED_100UV_THRESH, color="red", ls="--", lw=0.8, label="100uV")
    ax.axhline(MAD_K3_AMP_THRESH_UV, color="purple", ls="--", lw=0.8, label="mad_k3 (945.6uV)")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("trial #"); ax.set_ylabel("max-ch ptp (uV)")
    ax.legend(fontsize=6)


def per_channel_panel(ax, bc_epochs_uv, title):
    # mean ptp per channel across all trials
    ptp_per_ch = np.ptp(bc_epochs_uv, axis=2)  # (n_trials, 128)
    mean_ptp = ptp_per_ch.mean(axis=0)  # (128,)
    ax.bar(np.arange(len(mean_ptp)), mean_ptp, color="teal", width=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("channel idx (E1..E128)"); ax.set_ylabel("mean ptp (uV)")


def make_comparison_figure(sid_outlier, sid_compare, out_path):
    d_out = np.load(ecfg.EPOCHS_CACHE / f"{sid_outlier}.npz")
    d_cmp = np.load(ecfg.EPOCHS_CACHE / f"{sid_compare}.npz")
    f_out = load_subject_flags(sid_outlier)
    f_cmp = load_subject_flags(sid_compare)

    bc_out_uv = d_out["bc_epochs"] * 1e6
    bc_cmp_uv = d_cmp["bc_epochs"] * 1e6

    def typical_and_worst_idx(trial_max_ptp_uv):
        med = np.median(trial_max_ptp_uv)
        typical = int(np.argmin(np.abs(trial_max_ptp_uv - med)))
        worst = int(np.argmax(trial_max_ptp_uv))
        return typical, worst

    t_out, w_out = typical_and_worst_idx(f_out["trial_max_ptp_uv"])
    t_cmp, w_cmp = typical_and_worst_idx(f_cmp["trial_max_ptp_uv"])

    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    butterfly_panel(axes[0, 0], bc_out_uv[t_out], f"{sid_outlier} typical trial #{t_out}")
    butterfly_panel(axes[0, 1], bc_out_uv[w_out], f"{sid_outlier} WORST trial #{w_out}")
    trial_trace_panel(axes[0, 2], f_out["trial_max_ptp_uv"], f"{sid_outlier} per-trial max-ptp (480 trials)")
    per_channel_panel(axes[0, 3], bc_out_uv, f"{sid_outlier} per-channel mean ptp")

    butterfly_panel(axes[1, 0], bc_cmp_uv[t_cmp], f"{sid_compare} typical trial #{t_cmp}")
    butterfly_panel(axes[1, 1], bc_cmp_uv[w_cmp], f"{sid_compare} WORST trial #{w_cmp}")
    trial_trace_panel(axes[1, 2], f_cmp["trial_max_ptp_uv"], f"{sid_compare} per-trial max-ptp (480 trials)")
    per_channel_panel(axes[1, 3], bc_cmp_uv, f"{sid_compare} per-channel mean ptp")

    fig.suptitle(f"Signal-quality visual QC: {sid_outlier} (statistical outlier) vs {sid_compare} (representative non-outlier HC)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ecfg.RUNS_DIR / f"{ts}_stage1c_quality_outlier_check"
    run_dir.mkdir(parents=True)

    manifest = pd.read_csv(ecfg.MANIFEST_128REST, dtype={"subject_id": str})
    type_of = dict(zip(manifest["subject_id"], manifest["type"]))

    sids = sorted(p.stem for p in FLAGS_DIR.glob("*.npz"))
    rates = {name: {} for name in FLAG_NAMES}
    for sid in sids:
        f = load_subject_flags(sid)
        for name in FLAG_NAMES:
            rates[name][sid] = float(f["flags"][name].mean())

    # ---- step 1: statistical outlier criterion, all 53 subjects, all 3 flag sets ----
    thresholds = {}
    outlier_flags = {name: {} for name in FLAG_NAMES}
    for name in FLAG_NAMES:
        rate_arr = np.array([rates[name][sid] for sid in sids])
        med, mad, thresh = scaled_mad_threshold(rate_arr, MAD_K)
        thresholds[name] = {"median": float(med), "mad_scaled": float(mad), "threshold": float(thresh)}
        for sid in sids:
            outlier_flags[name][sid] = rates[name][sid] > thresh

    any_outlier = {sid: any(outlier_flags[name][sid] for name in FLAG_NAMES) for sid in sids}
    outlier_sids = sorted([sid for sid in sids if any_outlier[sid]])

    print("=== Step 1: statistical outlier thresholds (median + 3*MAD) ===")
    for name in FLAG_NAMES:
        print(f"[{name}] median={thresholds[name]['median']:.4f} mad_scaled={thresholds[name]['mad_scaled']:.4f} "
              f"threshold={thresholds[name]['threshold']:.4f}")
    print(f"\nsubjects exceeding >=1 threshold ({len(outlier_sids)}): {outlier_sids}")
    for sid in outlier_sids:
        rr = {name: round(rates[name][sid], 4) for name in FLAG_NAMES}
        exceeds = {name: bool(outlier_flags[name][sid]) for name in FLAG_NAMES}
        print(f"  {sid} ({type_of[sid]}): rates={rr} exceeds={exceeds}")

    target_sid = "02030009"
    print(f"\n--- {target_sid} specifically ---")
    print(f"  rates={ {name: round(rates[name][target_sid], 4) for name in FLAG_NAMES} }")
    print(f"  thresholds={ {name: round(thresholds[name]['threshold'], 4) for name in FLAG_NAMES} }")
    print(f"  is_statistical_outlier={any_outlier[target_sid]}")

    # ---- step 2: visual check, one figure per outlier vs a representative non-outlier HC ----
    non_outlier_hc = [sid for sid in sids if type_of[sid] == "HC" and not any_outlier[sid]]
    hc_ptp_med = np.median([rates["ptp_100uv"][sid] for sid in non_outlier_hc])
    hc_abs_med = np.median([rates["abs_100uv"][sid] for sid in non_outlier_hc])
    hc_madk3_med = np.median([rates["mad_k3"][sid] for sid in non_outlier_hc])

    def dist_to_hc_median(sid):
        return (abs(rates["ptp_100uv"][sid] - hc_ptp_med)
                + abs(rates["abs_100uv"][sid] - hc_abs_med)
                + abs(rates["mad_k3"][sid] - hc_madk3_med))

    representative_hc = min(non_outlier_hc, key=dist_to_hc_median)
    print(f"\nrepresentative comparison HC subject: {representative_hc} "
          f"(rates={ {name: round(rates[name][representative_hc], 4) for name in FLAG_NAMES} })")

    figure_paths = {}
    for sid in outlier_sids:
        out_path = run_dir / f"visual_qc_{sid}_vs_{representative_hc}.png"
        make_comparison_figure(sid, representative_hc, out_path)
        figure_paths[sid] = str(out_path)
        print(f"  wrote {out_path}")

    # ---- assemble report; exclusion decision (step 3) left as null pending the
    #      visual-check judgment, which is recorded separately after this script
    #      is run and the figures are actually inspected ----
    report = {
        "protocol": {
            "statistical_rule": "rate > pooled_median(rate) + 3*MAD(rate)*1.4826, per flag set, all 53 subjects",
            "exclusion_rule": "excluded from primary (kept in sensitivity) iff statistical outlier on >=1 flag AND visual check confirms genuine diffuse quality problem",
        },
        "thresholds": thresholds,
        "representative_comparison_hc": representative_hc,
        "per_subject_rates": {sid: {name: rates[name][sid] for name in FLAG_NAMES} for sid in sids},
        "statistical_outliers": {
            sid: {
                "type": type_of[sid],
                "rates": {name: rates[name][sid] for name in FLAG_NAMES},
                "exceeds": {name: bool(outlier_flags[name][sid]) for name in FLAG_NAMES},
                "figure": figure_paths[sid],
            } for sid in outlier_sids
        },
        "target_subject_02030009": {
            "is_statistical_outlier": bool(any_outlier[target_sid]),
            "rates": {name: rates[name][target_sid] for name in FLAG_NAMES},
        },
        "visual_check_result": "PENDING_MANUAL_INSPECTION",
        "final_exclusion_list": "PENDING_VISUAL_CHECK",
    }

    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(ecfg.RESULTS_DIR / "erp_stage1c_quality_outlier_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\narchived to {run_dir}")
    return report


if __name__ == "__main__":
    main()
