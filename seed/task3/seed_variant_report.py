"""
Stage 2 report: four theta-variants x {subject bal_acc mean+-std, n_params},
plus V1-V2 / V1-V3 / V1-V4 differences with PAIRED statistics (folds are paired
-- same held-out test subject per fold across variants). Applies the pre-
registered decision lines. Archives to seed/runs/.
"""
import json
import shutil
import time

import numpy as np
from scipy import stats

import seed_mamba_config as cfg

RES = cfg.RESULTS_DIR
FILES = {"V1_full": "var_V1.json", "V2_theta_zero": "var_V2.json",
         "V3_theta_frozen_random": "var_V3.json", "V4_real_only": "var_V4.json"}


def load(v):
    rows = sorted(json.load(open(RES / FILES[v])), key=lambda r: int(r["fold"].split("_")[1]))
    accs = np.array([r["subject_bal_acc"] for r in rows])
    return rows, accs, rows[0]["n_params"]


def paired(a_v1, a_other):
    d = a_v1 - a_other
    mean = float(d.mean()); sd = float(d.std(ddof=1))
    n = len(d); se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    t_p = float(stats.ttest_rel(a_v1, a_other).pvalue)
    try:
        w_p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        w_p = None
    return {"mean_diff": mean, "ci95": [float(ci[0]), float(ci[1])],
            "paired_t_p": t_p, "wilcoxon_p": w_p,
            "diff_over_v1_foldstd": mean / float(a_v1.std(ddof=1))}


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_seed_mamba_stage2_theta_ablation"
    run_dir.mkdir(parents=True)

    data = {v: load(v) for v in FILES}
    summary = {}
    for v, (rows, accs, npar) in data.items():
        summary[v] = {"mean": float(accs.mean()), "std": float(accs.std(ddof=1)),
                      "n_params": npar, "per_fold": rows}

    a1 = data["V1_full"][1]
    diffs = {f"V1_minus_{v}": paired(a1, data[v][1]) for v in ["V2_theta_zero", "V3_theta_frozen_random", "V4_real_only"]}

    # ---- pre-registered decision ----
    d12, d13, d14 = diffs["V1_minus_V2_theta_zero"], diffs["V1_minus_V3_theta_frozen_random"], diffs["V1_minus_V4_real_only"]
    fold_std = summary["V1_full"]["std"]
    v1, v2, v4 = summary["V1_full"]["mean"], summary["V2_theta_zero"]["mean"], summary["V4_real_only"]["mean"]
    verdict = []
    strong12 = d12["mean_diff"] > fold_std
    strong13 = d13["mean_diff"] > fold_std
    if strong12 and strong13 and v1 >= v4:
        verdict.append("MECHANISM HOLDS: learnable selective complex rotation is the signal source (V1-V2 and V1-V3 both exceed fold-std, V1>=V4). Core paper result -> proceed to probe 2 (synthetic injection).")
    if abs(d12["mean_diff"]) < 0.5 * fold_std:
        verdict.append("V1~V2: rotation buys little -> complex state may not be the key on this task; rethink thesis (but now under a signal-present premise, so the conclusion is trustworthy).")
    if d12["mean_diff"] > 0 and abs(d13["mean_diff"]) < 0.5 * fold_std:
        verdict.append("V1>V2 but V1~V3: oscillatory prior helps, selectivity adds nothing -> narrative retreats to 'complex/oscillatory prior'.")
    if v1 < v4:
        verdict.append("V1<V4: capacity illusion -> ALARM.")
    if not verdict:
        verdict.append("AMBIGUOUS / MILD: differences do not cleanly clear fold-std. Per the honest expectation, effect may be real-but-mild; recommend more seeds / statistical testing before claiming mechanism.")

    report = {
        "protocol": "LOSO 15-fold, 3-class subject bal_acc (chance 0.333); four theta-variants share V1's exact config "
                    "(d_model=128,d_state=128,headdim=32,mimo_rank=4,is_mimo=True,bf16); only theta computation differs. "
                    "V4=real Mamba2 param-matched. Folds paired by held-out test subject.",
        "consistency": "V1_full reproduces stock Mamba3 exactly (max_abs_diff=0); see seed_variant_consistency.py",
        "variants": {v: {"mean": summary[v]["mean"], "std": summary[v]["std"], "n_params": summary[v]["n_params"]} for v in FILES},
        "differences_paired": diffs,
        "fold_std_V1": fold_std,
        "preregistered_verdict": verdict,
        "lr_reference": 0.539,
        "per_fold": {v: summary[v]["per_fold"] for v in FILES},
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(RES / "seed_mamba_stage2_report.json", "w") as f:
        json.dump(report, f, indent=2)
    for v in FILES:
        shutil.copy(RES / FILES[v], run_dir / FILES[v])
    shutil.copy(cfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")

    print("=== SEED Mamba Stage 2: theta ablation (LOSO 15-fold, chance 0.333) ===")
    for v in FILES:
        print(f"  {v:24s} bal_acc={summary[v]['mean']:.3f} +- {summary[v]['std']:.3f}   n_params={summary[v]['n_params']:,}")
    print(f"  V1 fold-std = {fold_std:.3f}")
    for k, d in diffs.items():
        print(f"  {k}: mean_diff={d['mean_diff']:+.3f}  CI95=[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]  "
              f"paired_t_p={d['paired_t_p']:.3f}  wilcoxon_p={d['wilcoxon_p']}  diff/foldstd={d['diff_over_v1_foldstd']:+.2f}")
    print("  --- pre-registered verdict ---")
    for x in verdict:
        print("   *", x)
    print(f"  archived {run_dir}")
    return report


if __name__ == "__main__":
    main()
