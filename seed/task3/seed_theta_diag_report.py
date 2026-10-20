"""
Stage 3a report + archive. Reads seed_theta_diag_analysis.json, attaches the
pre-registered verdict (A: theta collapses to ~0 -> model declines to rotate;
B: theta active/data-dependent but zero-theta ~= identity -> rotation used but
redundant with the real path), and archives the whole evidence bundle (analysis,
gen jsons, checkpoints, splits) immutably to seed/runs/.
"""
import json
import shutil
import time
from pathlib import Path

import seed_mamba_config as cfg

HERE = Path(__file__).parent
ANALYSIS = cfg.RESULTS_DIR / "seed_theta_diag_analysis.json"
CKPT_DIR = HERE / "theta_diag" / "ckpt"


def main():
    rep = json.load(open(ANALYSIS))
    agg = rep["aggregate"]
    L = agg["layers"]
    nL = agg["n_layers"]

    # --- collapse test (is theta OFF?) ---
    tanh_tr = [L[str(i)]["abs_tanh_angle_mean_trained"][0] if str(i) in L else L[i]["abs_tanh_angle_mean_trained"][0] for i in range(nL)]
    # json keys may be ints; normalize
    def lay(i):
        return L[str(i)] if str(i) in L else L[i]
    tanh_tr = [lay(i)["abs_tanh_angle_mean_trained"][0] for i in range(nL)]
    tanh_in = [lay(i)["abs_tanh_angle_mean_init"][0] for i in range(nL)]
    cumrot = [lay(i)["cum_rot_over_2pi_mean_trained"][0] for i in range(nL)]
    cross = [lay(i)["crossSample_std_tanh_trained"][0] for i in range(nL)]
    wratio = [agg["w_angle_final_over_init"][str(i)][0] if str(i) in agg["w_angle_final_over_init"] else agg["w_angle_final_over_init"][i][0] for i in range(nL)]
    gratio = [agg["theta_grad_over_inproj_grad_lateEpochs"][str(i)][0] if str(i) in agg["theta_grad_over_inproj_grad_lateEpochs"] else agg["theta_grad_over_inproj_grad_lateEpochs"][i][0] for i in range(nL)]
    iv = agg["intervention"]

    collapsed = max(tanh_tr) < 0.05 and max(cumrot) < 0.02 and max(wratio) < 0.5
    zero_equals_identity = abs(iv["drop_zero_mean"]) < 0.02 and abs(iv["drop_shuffle_mean"]) < 0.02

    verdict = []
    if collapsed and zero_equals_identity:
        verdict.append("(A) theta COLLAPSED to ~0: the trained model actively drives rotation to zero "
                       "and zero-theta == identity -> strong evidence the network chooses NOT to rotate.")
    elif (not collapsed) and zero_equals_identity:
        verdict.append("(B) theta ACTIVE but REDUNDANT: rotation is switched on (|tanh(theta)| ~ %.2f, "
                       "cumulative rotation ~ %.2f turns/segment, data-dependent cross-sample std ~ %.2f, "
                       "angle weights even GREW x%.2f), yet ablating it post-hoc costs nothing "
                       "(zero-theta drop=%+.3f, shuffle-theta drop=%+.3f). The emotion-relevant information is "
                       "carried by the real/decay path; the complex rotation is decision-IRRELEVANT on SEED. "
                       "This mechanistically explains Stage 2 (V1~V2~V4)."
                       % (sum(tanh_tr)/nL, sum(cumrot)/nL, sum(cross)/nL, sum(wratio)/nL,
                          iv["drop_zero_mean"], iv["drop_shuffle_mean"]))
        verdict.append("Low optimization pressure corroborates: late-epoch theta-row gradient is only "
                       "~%.1f%% of the full in_proj gradient -> the loss is nearly flat in theta." % (100*sum(gratio)/nL))
    elif not zero_equals_identity:
        verdict.append("(C) theta MATTERS: removing/shuffling rotation changes accuracy "
                       "(zero drop=%+.3f, shuffle drop=%+.3f) -> rotation carries decision-relevant signal. "
                       "Revisit the Stage-2 negative under this lens." % (iv["drop_zero_mean"], iv["drop_shuffle_mean"]))
    else:
        verdict.append("AMBIGUOUS: collapse and intervention signals disagree; inspect per-fold.")

    rep["preregistered_verdict"] = verdict
    rep["verdict_inputs"] = {
        "abs_tanh_angle_trained_by_layer": tanh_tr,
        "abs_tanh_angle_init_by_layer": tanh_in,
        "cum_rot_over_2pi_by_layer": cumrot,
        "crossSample_std_by_layer": cross,
        "w_angle_final_over_init_by_layer": wratio,
        "theta_grad_over_inproj_grad_by_layer": gratio,
        "intervention": {"identity": iv["identity"][0], "zero_theta": iv["zero_theta"][0],
                         "shuffle_theta": iv["shuffle_theta"][0],
                         "drop_zero": iv["drop_zero_mean"], "drop_shuffle": iv["drop_shuffle_mean"]},
        "collapsed": collapsed, "zero_equals_identity": zero_equals_identity,
    }

    with open(ANALYSIS, "w") as f:
        json.dump(rep, f, indent=2)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_seed_mamba_stage3a_theta_diagnostic"
    (run_dir / "ckpt").mkdir(parents=True)
    shutil.copy(ANALYSIS, run_dir / "seed_theta_diag_analysis.json")
    for p in ["theta_diag/gen_gpu0.json", "theta_diag/gen_gpu1.json",
              "theta_diag/gen_gpu0.log", "theta_diag/gen_gpu1.log"]:
        fp = HERE / p
        if fp.exists():
            shutil.copy(fp, run_dir / Path(p).name)
    for cp in sorted(CKPT_DIR.glob("fold_*.pt")):
        shutil.copy(cp, run_dir / "ckpt" / cp.name)
    shutil.copy(cfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")
    for src in ["seed_theta_diag_gen.py", "seed_theta_diag_analyze.py", "seed_theta_diag_report.py"]:
        shutil.copy(HERE / src, run_dir / src)

    print("=== SEED Mamba Stage 3a: theta usage diagnostic (regenerated V1_full, no new experiment) ===")
    print(f"  regen V1 mean bal_acc (sanity vs Stage2 0.548): "
          f"{sum(f['test_bal_acc_regen'] for f in rep['per_fold'])/len(rep['per_fold']):.3f}")
    for i in range(nL):
        print(f"  layer{i}: |tanhθ| init={tanh_in[i]:.3f}->trained={tanh_tr[i]:.3f}  "
              f"cumrot/2π={cumrot[i]:.3f}  crossStd={cross[i]:.3f}  "
              f"Wangle f/i={wratio[i]:.3f}  θgrad/inproj={gratio[i]*100:.1f}%")
    print(f"  intervention (trained model): identity={iv['identity'][0]:.3f}  "
          f"zero-θ={iv['zero_theta'][0]:.3f} (Δ{iv['drop_zero_mean']:+.3f})  "
          f"shuffle-θ={iv['shuffle_theta'][0]:.3f} (Δ{iv['drop_shuffle_mean']:+.3f})")
    print("  --- pre-registered verdict ---")
    for v in verdict:
        print("   *", v)
    print(f"  archived {run_dir}")


if __name__ == "__main__":
    main()
