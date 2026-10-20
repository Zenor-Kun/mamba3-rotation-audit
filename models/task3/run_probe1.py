"""
Task 3 / Probe 1 orchestrator: 4 variants x N folds, one seed at a time.
Run with `python3 run_probe1.py` for the seed=42 directional pass (default);
pass --seeds to extend to the full sweep once seed42 clears the pre-registered
decision line (see STATUS.md) -- never launch the full 4x5x5=100 sweep blind.
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

import probe1_config as pcfg
import probe1_runs as runs
import windows
from probe1_train import run_fold
from variant_model import build_classifier, trainable_param_count


def run_variant_seed(variant, seed, manifest_df, splits, device, log_lines):
    run_dir = runs.new_run_dir(f"probe1_{variant}_seed{seed}")
    runs.write_snapshot(run_dir)

    fold_results = []
    t0 = time.time()
    for fold in range(pcfg.N_FOLDS):
        line = f"  [{variant} seed={seed}] fold {fold} start"
        print(line); log_lines.append(line)
        bal_acc, auc, n_subj, inner_val_bal_acc, n_params = run_fold(variant, seed, fold, manifest_df, splits, device)
        line = (f"  [{variant} seed={seed}] fold {fold}: TEST balanced_acc={bal_acc:.3f} auc={auc:.3f} "
                f"(n_test_subj={n_subj}, inner_val_bal_acc={inner_val_bal_acc:.3f}, params={n_params:,})")
        print(line); log_lines.append(line)
        fold_results.append({"fold": fold, "balanced_acc": bal_acc, "auc": auc,
                              "n_test_subj": n_subj, "inner_val_bal_acc": inner_val_bal_acc,
                              "n_params": n_params})

    bal_accs = [r["balanced_acc"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    summary = {
        "variant": variant, "seed": seed,
        "fold_results": fold_results,
        "mean_balanced_acc": float(np.mean(bal_accs)), "std_balanced_acc": float(np.std(bal_accs)),
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
        "n_params": fold_results[0]["n_params"],
        "elapsed_sec": time.time() - t0,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "train_run.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"  archived to {run_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[pcfg.SEED_FIRST])
    ap.add_argument("--variants", type=str, nargs="+", default=pcfg.VARIANTS)
    args = ap.parse_args()

    device = pcfg.DEVICE
    manifest_df = pd.read_csv(pcfg.task2_cfg.MANIFEST_IN, dtype={"subject_id": str})
    splits = windows.load_splits()

    all_summaries = []
    for seed in args.seeds:
        for variant in args.variants:
            log_lines = []
            print(f"\n=== variant={variant} seed={seed} ===")
            summary = run_variant_seed(variant, seed, manifest_df, splits, device, log_lines)
            all_summaries.append(summary)

    # ---- table: variant x {balanced_acc, auc} mean+/-std across whatever seeds were run, + params ----
    rows = []
    for variant in args.variants:
        variant_summaries = [s for s in all_summaries if s["variant"] == variant]
        bal_means = [s["mean_balanced_acc"] for s in variant_summaries]
        auc_means = [s["mean_auc"] for s in variant_summaries]
        # fold-level pooled across all seeds run so far, for the table's std column
        all_bal = [r["balanced_acc"] for s in variant_summaries for r in s["fold_results"]]
        all_auc = [r["auc"] for s in variant_summaries for r in s["fold_results"]]
        rows.append({
            "variant": variant,
            "n_params": variant_summaries[0]["n_params"] if variant_summaries else None,
            "seeds_run": [s["seed"] for s in variant_summaries],
            "balanced_acc_mean": float(np.mean(all_bal)) if all_bal else None,
            "balanced_acc_std": float(np.std(all_bal)) if all_bal else None,
            "auc_mean": float(np.mean(all_auc)) if all_auc else None,
            "auc_std": float(np.std(all_auc)) if all_auc else None,
            "n_folds_pooled": len(all_bal),
        })

    tag = "seed42" if args.seeds == [42] else "multiseed"
    out_json = pcfg.RESULTS_DIR / f"probe1_table_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n=== Probe 1 table ({tag}) ===")
    print(f"{'variant':<28}{'n_params':>12}{'bal_acc':>18}{'auc':>18}{'n_folds':>10}")
    for r in rows:
        ba = f"{r['balanced_acc_mean']:.3f}+/-{r['balanced_acc_std']:.3f}" if r['balanced_acc_mean'] is not None else "n/a"
        au = f"{r['auc_mean']:.3f}+/-{r['auc_std']:.3f}" if r['auc_mean'] is not None else "n/a"
        print(f"{r['variant']:<28}{r['n_params']:>12,}{ba:>18}{au:>18}{r['n_folds_pooled']:>10}")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
