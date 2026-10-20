"""
Task 4 ERP, pre-Stage-2 checks (no modeling, no trial deletion, no file
changes to epochs_cache/artifact_flags/splits). Three numbers requested
before starting Stage 2:

1. Primary-analysis (n=49, Stage 1c exclusions applied) MDD/HC split, plus
   the label of each of the 4 excluded subjects.
2. For the 5 "more blinking, kept" subjects flagged in Stage 1c
   (02010004, 02010005, 02010011, 02030004, 02030019): per-subject trial
   RETENTION rate (1 - rejection rate) under each of the 3 Stage 1b flag
   sets (ptp_100uv, abs_100uv, mad_k3).
3. For all 49 primary subjects: per-subject-per-condition (f/h/s) trial
   count AFTER dropping ptp_100uv-flagged trials, then the distribution
   (min/median/max) of that count across all 49*3=147 (subject,condition)
   cells -- this is the number that would feed a per-subject-per-condition
   averaged-ERP representation in Stage 2.
"""
import json

import numpy as np
import pandas as pd

import erp_config as ecfg

FLAGS_DIR = ecfg.TASK4_DIR / "artifact_flags"
EXCLUDED = ["02010030", "02010034", "02030009", "02030020"]
BLINK_KEPT = ["02010004", "02010005", "02010011", "02030004", "02030019"]


def main():
    manifest = pd.read_csv(ecfg.MANIFEST_128REST, dtype={"subject_id": str})
    label_of = dict(zip(manifest["subject_id"], manifest["label"]))
    type_of = dict(zip(manifest["subject_id"], manifest["type"]))

    all_sids = sorted(p.stem for p in FLAGS_DIR.glob("*.npz"))
    assert len(all_sids) == 53
    primary_sids = sorted(set(all_sids) - set(EXCLUDED))
    assert len(primary_sids) == 49

    # ---- 1. primary MDD/HC split + excluded subjects' labels ----
    primary_types = [type_of[s] for s in primary_sids]
    n_mdd_primary = primary_types.count("MDD")
    n_hc_primary = primary_types.count("HC")
    excluded_labels = {s: type_of[s] for s in EXCLUDED}

    print("=== 1. Primary analysis (n=49) MDD/HC split ===")
    print(f"MDD={n_mdd_primary} HC={n_hc_primary} (full 53: MDD=24 HC=29)")
    print(f"excluded subjects' labels: {excluded_labels}")

    # ---- 2. blink-kept subjects: retention rate under 3 flag sets ----
    print("\n=== 2. Blink-heavy (kept) subjects: trial retention rate per flag set ===")
    blink_report = {}
    for sid in BLINK_KEPT:
        d = np.load(FLAGS_DIR / f"{sid}.npz")
        rates = {}
        for name in ("ptp_100uv", "abs_100uv", "mad_k3"):
            flag = d[f"{name}_flag"]
            rates[name] = {
                "retention_rate": float(1.0 - flag.mean()),
                "n_kept": int((~flag).sum()), "n_total": int(flag.shape[0]),
            }
        blink_report[sid] = rates
        print(f"  {sid} ({type_of[sid]}): "
              f"ptp_100uv retain={rates['ptp_100uv']['retention_rate']:.3f} ({rates['ptp_100uv']['n_kept']}/480)  "
              f"abs_100uv retain={rates['abs_100uv']['retention_rate']:.3f} ({rates['abs_100uv']['n_kept']}/480)  "
              f"mad_k3 retain={rates['mad_k3']['retention_rate']:.3f} ({rates['mad_k3']['n_kept']}/480)")

        # per-condition breakdown under ptp_100uv specifically (the criterion used in #3)
        conditions = d["conditions"]
        ptp_flag = d["ptp_100uv_flag"]
        cond_counts = {}
        for letter in ("f", "h", "s"):
            m = conditions == letter
            cond_counts[letter] = int((~ptp_flag[m]).sum())
        blink_report[sid]["ptp_100uv_per_condition_n_kept"] = cond_counts
        print(f"    -> per-condition n_kept after ptp_100uv drop: {cond_counts}")

    # ---- 3. all 49 primary subjects: per-subject-per-condition trial count after
    #      dropping ptp_100uv-flagged trials ----
    print("\n=== 3. Primary 49 subjects: per-subject-per-condition trial count "
          "after dropping ptp_100uv-flagged trials ===")
    per_cell_counts = []
    per_subject_cond = {}
    for sid in primary_sids:
        d = np.load(FLAGS_DIR / f"{sid}.npz")
        conditions = d["conditions"]
        ptp_flag = d["ptp_100uv_flag"]
        cond_counts = {}
        for letter in ("f", "h", "s"):
            m = conditions == letter
            n_kept = int((~ptp_flag[m]).sum())
            cond_counts[letter] = n_kept
            per_cell_counts.append(n_kept)
        per_subject_cond[sid] = cond_counts

    arr = np.array(per_cell_counts)
    assert len(arr) == 49 * 3
    dist = {"min": int(arr.min()), "median": float(np.median(arr)), "max": int(arr.max()),
            "mean": float(arr.mean()), "n_cells": len(arr)}
    print(f"distribution over {dist['n_cells']} (subject,condition) cells: "
          f"min={dist['min']} median={dist['median']} max={dist['max']} mean={dist['mean']:.1f}")

    # which cell(s) hit the min, for context
    min_cells = [(sid, letter, n) for sid, cc in per_subject_cond.items()
                 for letter, n in cc.items() if n == dist["min"]]
    print(f"cell(s) at min: {min_cells}")

    report = {
        "primary_analysis_split": {"n_mdd": n_mdd_primary, "n_hc": n_hc_primary, "n_total": len(primary_sids)},
        "excluded_subjects_labels": excluded_labels,
        "blink_kept_subjects_retention": blink_report,
        "primary49_per_condition_trial_count_after_ptp100uv_drop": {
            "distribution": dist, "min_cells": min_cells,
            "per_subject": per_subject_cond,
        },
    }
    with open(ecfg.RESULTS_DIR / "erp_stage1d_pre_stage2_checks.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwritten to {ecfg.RESULTS_DIR / 'erp_stage1d_pre_stage2_checks.json'}")
    return report


if __name__ == "__main__":
    main()
