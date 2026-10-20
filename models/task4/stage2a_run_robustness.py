"""
Task 4 Stage 2A robustness sweep: for the two least-frontal-confounded
signals found in Protections 1+2 (all/s and posterior_only/s -- NOT
frontal_only/h or frontal_only/s, which Protection 2's own logic already
flags as EOG-risk), re-run across:
  - mad_k3, n=49 (primary, already computed in protections12 report)
  - all_trials, n=49 (no trial cleaning at all)
  - abs_100uv, n=49
  - ptp_100uv, n=49
  - mad_k3, n=53 (Stage 1c's 4 quality-fault exclusions added back)
Conclusion is only trustworthy if it holds up (doesn't flip sign / collapse
to chance) across these variants -- LR only, 5seed x 5fold, same
subject-grouped CV protocol throughout.
"""
import json
import time

import pandas as pd

import erp_config as ecfg
import stage2a_config as cfg
from erp_roi import channel_set_variants
from stage2a_data import determine_primary_pool
from stage2a_eval import run_layer_cv, summarize_points

FLAGS_DIR = ecfg.TASK4_DIR / "artifact_flags"
TARGETS = [("all", "s"), ("posterior_only", "s")]


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_stage2a_robustness"
    run_dir.mkdir(parents=True)

    manifest = pd.read_csv(ecfg.MANIFEST_128REST, dtype={"subject_id": str})
    label_of = dict(zip(manifest["subject_id"], manifest["label"]))
    splits = cfg.load_splits()

    all_sids = sorted(p.stem for p in FLAGS_DIR.glob("*.npz"))
    stage1c_pool = sorted(set(all_sids) - set(cfg.EXCLUDED_STAGE1C))
    primary_ids, _ = determine_primary_pool(stage1c_pool)
    pool_53 = all_sids  # Stage 1c's 4 exclusions added back

    variants = channel_set_variants()

    pool_variants = [
        ("mad_k3_n49", "mad_k3", primary_ids),
        ("all_trials_n49", "all_trials", primary_ids),
        ("abs_100uv_n49", "abs_100uv", primary_ids),
        ("ptp_100uv_n49", "ptp_100uv", primary_ids),
        ("mad_k3_n53", "mad_k3", pool_53),
    ]

    results = {}
    log_lines = []
    t0 = time.time()
    for vname, layer in TARGETS:
        roi_dict = variants[vname]
        key = f"{vname}__{layer}"
        results[key] = {}
        for pv_name, flag_set, pool in pool_variants:
            points = run_layer_cv(pool, flag_set, roi_dict, layer, label_of, splits)
            summ = summarize_points(points)
            results[key][pv_name] = summ
            line = (f"[{key}][{pv_name}] n_pool={len(pool)} "
                    f"balanced_acc={summ['balanced_acc_mean']:.3f}+/-{summ['balanced_acc_std']:.3f} "
                    f"auc={summ['auc_mean']:.3f}+/-{summ['auc_std']:.3f} (n_points={summ['n_points']})")
            print(line); log_lines.append(line)
    print(f"robustness sweep done in {time.time()-t0:.1f}s")

    report = {"targets": TARGETS, "pool_variants": [p[0] for p in pool_variants], "results": results}
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(run_dir / "robustness.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(cfg.RESULTS_DIR / "erp_stage2a_robustness_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"archived to {run_dir}")
    return report


if __name__ == "__main__":
    main()
