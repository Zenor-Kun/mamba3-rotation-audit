"""
SEED Mamba Stage 3a (θ diagnostic) -- STEP 2: analyze regenerated V1_full.

Answers "does the trained model actually USE the complex rotation?" using the
checkpoints from seed_theta_diag_gen.py. NO new model is trained here.

θ semantics (kernel mamba3_mimo_rotary_step.py L76-77):
    per-step rotation  w_t[h,s] = tanh(angle_proj_t[s]) * dt_t[h] * pi
    cumulative angle    = sum_t w_t
angle_proj = last num_rope_angles(=32) dims of in_proj(u), shared across heads;
dt[h] = softplus(dd_dt[h] + dt_bias[h]).  "θ off" <=> tanh(angle_proj) -> 0.

Reports, per layer, init-model vs trained-model on the SAME real test data:
  1. |tanh(angle)| and |per-step rotation w| and |cumulative rotation| distributions.
  2. θ-weight shrink: ||W_angle|| init vs final (from gen json).
  3. θ gradient-norm trajectory + late-epoch value and ratio to full in_proj grad.
  4. data-dependence: cross-sample std of tanh(angle) at matched (frame,slot).
  5. post-hoc intervention on the TRAINED model: identity vs zero-θ vs shuffle-θ
     -> subject bal_acc change.
Verdict:
  (A) θ collapses to ~0 (weights shrink, |tanh|~0, cumulative<<2pi, zero-θ ~= identity)
      -> model actively declines to rotate: strong evidence rotation is unused.
  (B) θ large / data-dependent but zero-θ ~= identity -> rotation used but redundant
      with the real path (complex path expressible by the real decay).
"""
import argparse
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import torch

import seed_mamba_config as cfg
from seed_mamba_train import load_subject, subject_bal_acc, predict_proba, DTYPE
from seed_variant_model import SeedVariantClassifier

CKPT_DIR = Path(__file__).parent / "theta_diag" / "ckpt"
DEV = "cuda:0"
MAX_SEG = 128  # segments used for activation stats (x62 channels -> thousands of seqs)


def build_v1_loaded(state):
    m = SeedVariantClassifier("V1_full").to(DEV).eval()
    m.load_state_dict(state)
    return m


def split_sizes(mamba):
    return [mamba.d_inner, mamba.d_inner,
            mamba.d_state * mamba.num_bc_heads * mamba.mimo_rank,
            mamba.d_state * mamba.num_bc_heads * mamba.mimo_rank,
            mamba.nheads, mamba.nheads, mamba.nheads, mamba.num_rope_angles]


@torch.no_grad()
def extract_angles(model, raw, mean, std, max_seg=MAX_SEG, batch=16):
    """Forward real segments; capture per-layer angle_proj (pre-tanh) + dt via a
    hook on each in_proj. Returns per-layer dicts of concatenated tensors on CPU:
      tanh_angle (Nseq, L, S), dt (Nseq, L, H)."""
    n = min(max_seg, raw.shape[0])
    idx = np.linspace(0, raw.shape[0] - 1, n).astype(int)
    caps = {i: [] for i in range(len(model.blocks))}

    hooks = []
    sizes = [split_sizes(blk.mamba) for blk in model.blocks]

    def mk(li):
        def hook(mod, inp, out):
            parts = torch.split(out, sizes[li], dim=-1)
            dd_dt = parts[4]; angle = parts[7]
            m = model.blocks[li].mamba
            dt = torch.nn.functional.softplus(dd_dt + m.dt_bias)  # (bc, L, H)
            tanh_a = torch.tanh(angle)                            # (bc, L, S)
            caps[li].append((tanh_a.float().cpu(), dt.float().cpu()))
        return hook

    for li, blk in enumerate(model.blocks):
        hooks.append(blk.mamba.in_proj.register_forward_hook(mk(li)))

    for i in range(0, n, batch):
        bidx = idx[i:i + batch]
        xb = (raw[bidx] - mean[None, :, None]) / std[None, :, None]
        xb = torch.from_numpy(xb).to(DEV)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            model(xb)
    for h in hooks:
        h.remove()

    res = {}
    for li in caps:
        ta = torch.cat([c[0] for c in caps[li]], dim=0)   # (Nseq, L, S)
        dt = torch.cat([c[1] for c in caps[li]], dim=0)   # (Nseq, L, H)
        res[li] = {"tanh_angle": ta, "dt": dt}
    return res


def q(t, p, cap=8_000_000):
    """Quantile via numpy (no torch 16M-elem limit); random-subsample if huge."""
    x = t.flatten().numpy()
    if x.size > cap:
        rng = np.random.default_rng(0)
        x = x[rng.integers(0, x.size, cap)]
    return float(np.percentile(x, p * 100))


def angle_stats(ext):
    """From tanh_angle (Nseq,L,S) and dt (Nseq,L,H) compute rotation stats.
    Cumulative rotation via einsum (avoids materializing the (N,L,H,S) tensor):
        cum[n,h,s] = pi * sum_l tanh_a[n,l,s]*dt[n,l,h]."""
    out = {}
    for li, d in ext.items():
        ta = d["tanh_angle"]; dt = d["dt"]           # (N,L,S), (N,L,H)
        N, L, S = ta.shape; H = dt.shape[-1]
        abs_ta = ta.abs()
        cum = math.pi * torch.einsum("nls,nlh->nhs", ta, dt)  # (N,H,S) cumulative over frames
        abs_cum = cum.abs()
        # per-step |w| mean (exact, separable): mean_{h,s}|ta[s]|*dt[h] = mean_s|ta|*mean_h dt
        step_mean = float((abs_ta.mean(dim=2) * dt.mean(dim=2)).mean().item() * math.pi)
        # per-step p95 via a modest sub-block (avoid the full (N,L,H,S) tensor)
        ns = min(N, 48)
        wb = (abs_ta[:ns, :, None, :] * dt[:ns, :, :, None]) * math.pi   # (ns,L,H,S)
        step_p95 = q(wb, 0.95)
        # data-dependence: std across samples (N) at matched (L,S), averaged
        cross_std = float(ta.std(dim=0).mean().item())        # over (L,S)
        within_std = float(ta.std(dim=1).mean().item())       # variation over time within a seq
        out[li] = {
            "abs_tanh_angle_mean": float(abs_ta.mean().item()),
            "abs_tanh_angle_p50": q(abs_ta, 0.50),
            "abs_tanh_angle_p95": q(abs_ta, 0.95),
            "abs_step_rot_mean_rad": step_mean,
            "abs_step_rot_p95_rad": step_p95,
            "abs_cum_rot_mean_rad": float(abs_cum.mean().item()),
            "abs_cum_rot_p95_rad": q(abs_cum, 0.95),
            "cum_rot_over_2pi_mean": float(abs_cum.mean().item() / (2 * math.pi)),
            "cum_rot_over_2pi_p95": q(abs_cum, 0.95) / (2 * math.pi),
            "crossSample_std_tanh": cross_std,
            "withinSeq_std_tanh": within_std,
        }
    return out


def make_shuffle(model):
    """Monkeypatch each mamba to shuffle its angles along time (destroys learned
    schedule, preserves per-seq magnitude distribution)."""
    def shuffle_switch(self, angles_raw, batch, seqlen):
        a = angles_raw.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)  # (b,l,h,s)
        b, l, h, s = a.shape
        perm = torch.argsort(torch.rand(b, l, h, s, device=a.device), dim=1)
        return torch.gather(a, 1, perm)
    for blk in model.blocks:
        blk.mamba._theta_switch = types.MethodType(shuffle_switch, blk.mamba)


def intervention_baccs(best_state, raw, mean, std, label, trial):
    """Subject bal_acc of the TRAINED model under: identity θ, zero θ, shuffled θ."""
    # identity
    m = build_v1_loaded(best_state)
    p = predict_proba(m, raw, mean, std, DEV)
    ident = subject_bal_acc(p, label, trial)
    del m; torch.cuda.empty_cache()
    # zero-theta: V2 classifier shares V1 param names exactly (theta-switch adds no params)
    mz = SeedVariantClassifier("V2_theta_zero").to(DEV).eval()
    mz.load_state_dict(best_state)
    pz = predict_proba(mz, raw, mean, std, DEV)
    zero = subject_bal_acc(pz, label, trial)
    del mz; torch.cuda.empty_cache()
    # shuffle-theta
    ms = build_v1_loaded(best_state)
    make_shuffle(ms)
    torch.manual_seed(0)
    ps = predict_proba(ms, raw, mean, std, DEV)
    shuf = subject_bal_acc(ps, label, trial)
    del ms; torch.cuda.empty_cache()
    return float(ident), float(zero), float(shuf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(cfg.RESULTS_DIR / "seed_theta_diag_analysis.json"))
    args = ap.parse_args()

    # merge gen jsons for weight-shrink + grad trajectory
    gen = []
    for p in ["theta_diag/gen_gpu0.json", "theta_diag/gen_gpu1.json"]:
        fp = Path(__file__).parent / p
        if fp.exists():
            gen += json.load(open(fp))
    gen_by_fold = {g["fold"]: g for g in gen}

    ckpts = sorted(CKPT_DIR.glob("fold_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    per_fold = []
    for cp in ckpts:
        d = torch.load(cp, map_location="cpu", weights_only=False)
        fold = d["fold"]; sid = d["test_subject"]
        mean = d["channel_mean"]; std = d["channel_std"]
        subj = load_subject(sid)
        raw, label, trial = subj["raw"], subj["label"], subj["trial"]

        m_init = build_v1_loaded(d["init_state"])
        m_best = build_v1_loaded(d["best_state"])
        ext_init = extract_angles(m_init, raw, mean, std)
        ext_best = extract_angles(m_best, raw, mean, std)
        st_init = angle_stats(ext_init)
        st_best = angle_stats(ext_best)
        del m_init, m_best; torch.cuda.empty_cache()

        ident, zero, shuf = intervention_baccs(d["best_state"], raw, mean, std, label, trial)

        g = gen_by_fold.get(fold, {})
        per_fold.append({
            "fold": fold, "test_subject": sid, "best_epoch": d.get("best_epoch"),
            "test_bal_acc_regen": g.get("subject_bal_acc"),
            "init_w_angle": g.get("init_w_angle"),
            "final_w_angle": g.get("final_w_angle"),
            "grad_trajectory": g.get("grad_trajectory"),
            "angle_stats_init": st_init,
            "angle_stats_trained": st_best,
            "intervention": {"identity": ident, "zero_theta": zero, "shuffle_theta": shuf,
                             "drop_zero": ident - zero, "drop_shuffle": ident - shuf},
        })
        print(f"  {fold} sid={sid} bacc_regen={g.get('subject_bal_acc')} "
              f"|tanhθ|trained L0={st_best[0]['abs_tanh_angle_mean']:.3f} "
              f"cumrot/2pi L0={st_best[0]['cum_rot_over_2pi_mean']:.3f} "
              f"interv ident={ident:.3f} zero={zero:.3f} shuf={shuf:.3f}", flush=True)

    # ---------- aggregate ----------
    def agg_layer(key, sub, layer):
        return [f[key][layer][sub] for f in per_fold]

    n_layers = len(per_fold[0]["angle_stats_trained"])
    aggregate = {"n_folds": len(per_fold), "n_layers": n_layers, "layers": {}}
    for L in range(n_layers):
        def mstd(key, sub):
            v = np.array(agg_layer(key, sub, L)); return [float(v.mean()), float(v.std(ddof=1))]
        aggregate["layers"][L] = {
            "abs_tanh_angle_mean_init": mstd("angle_stats_init", "abs_tanh_angle_mean"),
            "abs_tanh_angle_mean_trained": mstd("angle_stats_trained", "abs_tanh_angle_mean"),
            "abs_step_rot_mean_rad_trained": mstd("angle_stats_trained", "abs_step_rot_mean_rad"),
            "abs_cum_rot_mean_rad_trained": mstd("angle_stats_trained", "abs_cum_rot_mean_rad"),
            "cum_rot_over_2pi_mean_trained": mstd("angle_stats_trained", "cum_rot_over_2pi_mean"),
            "crossSample_std_tanh_trained": mstd("angle_stats_trained", "crossSample_std_tanh"),
            "withinSeq_std_tanh_trained": mstd("angle_stats_trained", "withinSeq_std_tanh"),
        }
    # weight shrink (final/init ratio of ||W_angle||) per layer
    wshrink = {}
    for L in range(n_layers):
        r = []
        for f in per_fold:
            iw = f["init_w_angle"][L]["w_angle_fro"]; fw = f["final_w_angle"][L]["w_angle_fro"]
            r.append(fw / iw)
        wshrink[L] = [float(np.mean(r)), float(np.std(r, ddof=1))]
    aggregate["w_angle_final_over_init"] = wshrink
    # late-epoch theta grad ratio (mean over last 3 recorded epochs) per layer
    grad_ratio = {}
    for L in range(n_layers):
        rr = []
        for f in per_fold:
            traj = f["grad_trajectory"] or []
            last = traj[-3:] if len(traj) >= 3 else traj
            for e in last:
                tg = e["theta_gradnorm_by_layer"][L]; ig = e["inproj_gradnorm_by_layer"][L]
                if tg is not None and ig:
                    rr.append(tg / ig)
        grad_ratio[L] = [float(np.mean(rr)), float(np.std(rr, ddof=1))] if rr else [None, None]
    aggregate["theta_grad_over_inproj_grad_lateEpochs"] = grad_ratio

    interv = {
        "identity": [float(np.mean([f["intervention"]["identity"] for f in per_fold])),
                     float(np.std([f["intervention"]["identity"] for f in per_fold], ddof=1))],
        "zero_theta": [float(np.mean([f["intervention"]["zero_theta"] for f in per_fold])),
                       float(np.std([f["intervention"]["zero_theta"] for f in per_fold], ddof=1))],
        "shuffle_theta": [float(np.mean([f["intervention"]["shuffle_theta"] for f in per_fold])),
                          float(np.std([f["intervention"]["shuffle_theta"] for f in per_fold], ddof=1))],
        "drop_zero_mean": float(np.mean([f["intervention"]["drop_zero"] for f in per_fold])),
        "drop_shuffle_mean": float(np.mean([f["intervention"]["drop_shuffle"] for f in per_fold])),
    }
    aggregate["intervention"] = interv

    report = {"protocol": "Stage 3a theta diagnostic on regenerated V1_full (no new training). "
                          "LOSO folds; angle_proj=last 32 dims of in_proj (data-dependent), "
                          "w=tanh(angle)*dt*pi, cumulative over 112 frames.",
              "aggregate": aggregate, "per_fold": per_fold}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== aggregate ===")
    for L in range(n_layers):
        a = aggregate["layers"][L]
        print(f" layer{L}: |tanhθ| init={a['abs_tanh_angle_mean_init'][0]:.3f} "
              f"trained={a['abs_tanh_angle_mean_trained'][0]:.3f} | "
              f"cumrot/2pi={a['cum_rot_over_2pi_mean_trained'][0]:.3f} | "
              f"crossSampleStd={a['crossSample_std_tanh_trained'][0]:.3f} | "
              f"Wangle final/init={wshrink[L][0]:.3f} | θgrad/inprojgrad={grad_ratio[L][0]}")
    print(f" intervention: identity={interv['identity'][0]:.3f} zero={interv['zero_theta'][0]:.3f} "
          f"shuffle={interv['shuffle_theta'][0]:.3f} | drop_zero={interv['drop_zero_mean']:+.3f} "
          f"drop_shuffle={interv['drop_shuffle_mean']:+.3f}")
    print(f" wrote {args.out}")


if __name__ == "__main__":
    main()
