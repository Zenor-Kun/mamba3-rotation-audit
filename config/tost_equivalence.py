"""
TOST (two one-sided tests) equivalence analysis for the mechanism claim.
NO new training -- reads existing per-fold (SEED) and per-cell (synthetic) results.

Primary: SEED RQ1, V1 vs {V2,V3,V4} on the 15 LOSO folds (paired by test subject,
the pseudo-replication-safe unit), balanced_acc. Equivalence margins ±0.03 (pre-
specified) and ±0.05 (sensitivity). Equivalence at alpha=0.05 <=> the 90% CI of the
paired mean difference lies entirely within (-margin, +margin).

Supplementary: synthetic RQ3, V1 vs {V2,V4,GRU}, per-(cell,seed) paired diffs
pooled across cells (ceiling-dominated -> flagged) and across the noise-limited
cells only (where headroom exists).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent


def tost_paired(a, b, margin, alpha=0.05):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b; n = len(d)
    mean = float(d.mean()); sd = float(d.std(ddof=1)); se = sd / np.sqrt(n); df = n - 1
    t1 = (mean + margin) / se            # H0: mu <= -margin  (upper one-sided)
    p1 = float(stats.t.sf(t1, df))       # P(T > t1)
    t2 = (mean - margin) / se            # H0: mu >=  margin  (lower one-sided)
    p2 = float(stats.t.cdf(t2, df))      # P(T < t2)
    p_tost = max(p1, p2)
    tcrit = stats.t.ppf(1 - alpha, df)   # 90% CI (1-2alpha) for the mean diff
    ci90 = (mean - tcrit * se, mean + tcrit * se)
    return {"n": n, "mean_diff": mean, "se": se,
            "p1_gt_negmargin": p1, "p2_lt_posmargin": p2, "p_tost": p_tost,
            "equivalent": bool(p_tost < alpha),
            "ci90": [float(ci90[0]), float(ci90[1])],
            "ci90_within_margin": bool(ci90[0] > -margin and ci90[1] < margin)}


def load_seed():
    def accs(v):
        rows = sorted(json.load(open(HERE / f"task3/results/var_{v}.json")),
                      key=lambda r: int(r["fold"].split("_")[1]))
        return np.array([r["subject_bal_acc"] for r in rows])
    return {v: accs(v) for v in ["V1", "V2", "V3", "V4"]}


def load_synth():
    rows = []
    for t in ["A", "B"]:
        for suf in ["", "_hard"]:
            fp = HERE / f"synthetic_phase/results/train_{t}{suf}.json"
            if fp.exists():
                rows += json.load(open(fp))
    idx = {(r["task"], r["snr"], r["variant"], r["seed"]): r["test_acc"] for r in rows}
    ORA_LT = {(t, s) for t in ["A", "B"] for s in [0.1, 0.15, 0.2, 0.3]
              if not ((t == "A" and s == 0.3))}  # oracle<0.9 cells (A@0.3 oracle .908>=.9)
    # build paired vectors
    cells = sorted({(r["task"], r["snr"]) for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    def pair(var_other, subset):
        v1, vo = [], []
        for (t, s) in cells:
            if subset == "noise" and (t, s) not in ORA_LT:
                continue
            for sd in seeds:
                k1 = (t, s, "V1_full", sd); ko = (t, s, var_other, sd)
                if k1 in idx and ko in idx:
                    v1.append(idx[k1]); vo.append(idx[ko])
        return np.array(v1), np.array(vo)
    return pair


def main():
    out = {"margins": [0.03, 0.05], "seed_rq1": {}, "synth_rq3": {}}
    seed = load_seed()
    print("=== TOST -- SEED RQ1 (15 LOSO folds, balanced_acc, chance 0.333) ===")
    print(f"  means: V1={seed['V1'].mean():.3f} V2={seed['V2'].mean():.3f} "
          f"V3={seed['V3'].mean():.3f} V4={seed['V4'].mean():.3f}")
    for other in ["V2", "V3", "V4"]:
        out["seed_rq1"][f"V1_vs_{other}"] = {}
        for m in [0.03, 0.05]:
            r = tost_paired(seed["V1"], seed[other], m)
            out["seed_rq1"][f"V1_vs_{other}"][f"margin_{m}"] = r
            print(f"  V1 vs {other}  margin±{m}: diff={r['mean_diff']:+.3f} "
                  f"90%CI=[{r['ci90'][0]:+.3f},{r['ci90'][1]:+.3f}] p_TOST={r['p_tost']:.3f} "
                  f"-> {'EQUIVALENT' if r['equivalent'] else 'not equivalent'}")

    pair = load_synth()
    print("\n=== TOST -- synthetic RQ3 (test_acc, chance 0.5) ===")
    for subset, label in [("all", "all 14 cells [ceiling-dominated]"), ("noise", "noise-limited cells only")]:
        out["synth_rq3"][subset] = {}
        print(f"  -- {label} --")
        for other in ["V2_theta_zero", "V4_real_only", "GRU"]:
            v1, vo = pair(other, subset)
            out["synth_rq3"][subset][f"V1_vs_{other}"] = {}
            for m in [0.03, 0.05]:
                r = tost_paired(v1, vo, m)
                out["synth_rq3"][subset][f"V1_vs_{other}"][f"margin_{m}"] = r
                if m == 0.03:
                    print(f"    V1 vs {other:16s} n={r['n']:2d} diff={r['mean_diff']:+.4f} "
                          f"90%CI=[{r['ci90'][0]:+.4f},{r['ci90'][1]:+.4f}] p_TOST(±.03)={r['p_tost']:.3f} "
                          f"-> {'EQUIVALENT' if r['equivalent'] else 'not equiv'}")

    with open(HERE / "synthetic_phase/results/tost_equivalence.json", "w") as f:
        json.dump(out, f, indent=2)
    # also drop a copy next to SEED results
    with open(HERE / "task3/results/tost_equivalence.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  wrote tost_equivalence.json (task3/results + synthetic_phase/results)")


if __name__ == "__main__":
    main()
