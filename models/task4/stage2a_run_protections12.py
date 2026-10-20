"""
Task 4 Stage 2A, Protections 1+2: low-dim ROI x window ERP feature
confirmation (Protection 1) x 4 channel-set spatial control variants
(Protection 2), primary flag_set = mad_k3, primary pool = 49 subjects.
LR only. 5seed x 5fold, subject-grouped CV (pseudo-replication guard
enforced by construction, see stage2a_eval.py docstring).

No trial deletion, no file changes to epochs_cache/artifact_flags/splits.
Writes cached per-subject ERPs to stage2a_erp_cache/ (new dir).
"""
import json
import time

import pandas as pd

import erp_config as ecfg
import stage2a_config as cfg
from erp_roi import channel_set_variants
from stage2a_data import determine_primary_pool
from stage2a_eval import run_layer_cv, summarize_points
from stage2a_features import LAYER_NAMES

FLAGS_DIR = ecfg.TASK4_DIR / "artifact_flags"


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_stage2a_protections12"
    run_dir.mkdir(parents=True)

    manifest = pd.read_csv(ecfg.MANIFEST_128REST, dtype={"subject_id": str})
    label_of = dict(zip(manifest["subject_id"], manifest["label"]))
    splits = cfg.load_splits()

    all_sids = sorted(p.stem for p in FLAGS_DIR.glob("*.npz"))
    stage1c_pool = sorted(set(all_sids) - set(cfg.EXCLUDED_STAGE1C))
    primary_ids, min_trial_excluded = determine_primary_pool(stage1c_pool)
    print(f"Stage 1c pool n={len(stage1c_pool)}, additional min-trial exclusions: {len(min_trial_excluded)} "
          f"{list(min_trial_excluded.keys())}")
    print(f"Stage 2A primary pool: n={len(primary_ids)}")

    variants = channel_set_variants()
    results = {}
    log_lines = [f"primary pool n={len(primary_ids)}: {primary_ids}",
                 f"min_trial_excluded: {min_trial_excluded}"]

    t0 = time.time()
    for vname, roi_dict in variants.items():
        results[vname] = {}
        for layer in LAYER_NAMES:
            points = run_layer_cv(primary_ids, "mad_k3", roi_dict, layer, label_of, splits)
            summ = summarize_points(points)
            results[vname][layer] = {"summary": summ, "points": points}
            line = (f"[{vname}][{layer}] "
                    f"balanced_acc={summ['balanced_acc_mean']:.3f}+/-{summ['balanced_acc_std']:.3f} "
                    f"auc={summ['auc_mean']:.3f}+/-{summ['auc_std']:.3f} (n_points={summ['n_points']})")
            print(line); log_lines.append(line)
    print(f"protections 1+2 done in {time.time()-t0:.1f}s")

    # ---- flag any cell exceeding the pre-registered signal threshold ----
    hits = []
    for vname, layers in results.items():
        for layer, r in layers.items():
            if r["summary"]["balanced_acc_mean"] > cfg.SIGNAL_THRESHOLD:
                hits.append({"channel_set": vname, "layer": layer,
                             "balanced_acc_mean": r["summary"]["balanced_acc_mean"],
                             "auc_mean": r["summary"]["auc_mean"]})
    hits.sort(key=lambda h: -h["balanced_acc_mean"])
    print(f"\ncells exceeding balanced_acc > {cfg.SIGNAL_THRESHOLD}: {len(hits)}")
    for h in hits:
        print(f"  {h}")

    report = {
        "primary_pool": primary_ids, "primary_pool_n": len(primary_ids),
        "min_trial_excluded": min_trial_excluded,
        "flag_set": "mad_k3",
        "channel_set_variants": {v: {r: [int(i) for i in idx] for r, idx in rois.items()} for v, rois in variants.items()},
        "time_windows_ms": cfg.TIME_WINDOWS_MS,
        "results": results,
        "signal_threshold": cfg.SIGNAL_THRESHOLD,
        "hits_above_threshold": hits,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(run_dir / "protections12.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(cfg.RESULTS_DIR / "erp_stage2a_protections12_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\narchived to {run_dir}")
    return report


if __name__ == "__main__":
    main()
