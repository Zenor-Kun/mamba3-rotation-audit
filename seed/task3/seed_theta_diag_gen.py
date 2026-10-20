"""
SEED Mamba Stage 3a (θ diagnostic) -- STEP 1: regenerate V1_full checkpoints.

Stage 2 never persisted weights (seed_variant_train.run_fold discarded best_state
after evaluating), so there is nothing on disk to introspect. This script re-runs
V1_full with the EXACT Stage-2 recipe (same builder, same helpers, same
hyper-params) but (a) seeds numpy per fold so batch order is reproducible from now
on, (b) saves each fold's init + best state_dict, and (c) records the gradient norm
of the θ-producing in_proj weight rows across epochs.

Honesty note: because Stage-2 did NOT seed the per-epoch batch permutation, these
weights are a same-recipe / same-distribution V1_full, NOT bit-identical to the
Stage-2 run. The θ-usage question ("does the trained model actually rotate?") is a
property of the learned solution and is expected to be seed-robust; we still print
the per-fold bal_acc so it can be compared against Stage-2's V1 (mean 0.548).

θ semantics (from the kernel, mamba3_mimo_rotary_step.py L76-77):
    per-step rotation  w_t = tanh(angle_proj_t) * dt_t * pi
    cumulative angle   = sum_t w_t
where angle_proj = last `num_rope_angles` dims of in_proj(u) (data-dependent),
shared across heads. "θ collapses to 0" <=> tanh(angle_proj) -> 0.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import seed_mamba_config as cfg
from seed_mamba_train import (load_subject, channel_stats, train_subsample_idx,
                              labels_to_idx, subject_bal_acc, predict_proba, DTYPE)
from seed_variant_model import build_variant, trainable_param_count

CKPT_DIR = Path(__file__).parent / "theta_diag" / "ckpt"


def angle_rows_slice(mamba):
    """The last num_rope_angles rows of in_proj.weight produce θ (split order
    [z,x,B,C,dd_dt,dd_A,trap,angle] -> angle is last)."""
    return mamba.num_rope_angles


def collect_angle_weight(model):
    """||W_angle|| and ||W_all|| per layer for the in_proj angle rows."""
    out = []
    for i, blk in enumerate(model.blocks):
        m = blk.mamba
        nra = angle_rows_slice(m)
        W = m.in_proj.weight.detach()
        w_ang = W[-nra:, :]
        out.append({"layer": i, "num_rope_angles": nra,
                    "w_angle_fro": float(w_ang.norm().item()),
                    "w_angle_absmean": float(w_ang.abs().mean().item()),
                    "w_all_fro": float(W.norm().item()),
                    "w_angle_rowmean_absmean": float(w_ang.abs().mean(dim=1).mean().item())})
    return out


def run_fold(fold_name, fold, device, log):
    test_sid = int(fold["test"][0]); train_ids = [int(s) for s in fold["train"]]
    rng = np.random.default_rng(1234 + test_sid)
    inner_val_ids = list(rng.choice(train_ids, size=cfg.INNER_VAL_SUBJECTS, replace=False))
    inner_train_ids = [s for s in train_ids if s not in inner_val_ids]

    subj = {s: load_subject(s) for s in train_ids + [test_sid]}
    mean, std = channel_stats([subj[s]["raw"] for s in inner_train_ids])

    tr_raw, tr_lab = [], []
    for s in inner_train_ids:
        d = subj[s]; sel = train_subsample_idx(d["label"], d["trial"])
        tr_raw.append(d["raw"][sel]); tr_lab.append(d["label"][sel])
    tr_raw = np.concatenate(tr_raw); tr_y = labels_to_idx(np.concatenate(tr_lab))
    n_train = tr_raw.shape[0]

    torch.manual_seed(1234 + test_sid)
    np.random.seed(1234 + test_sid)  # <-- reproducible batch order (Stage-2 lacked this)
    model = build_variant("V1_full").to(device)
    n_params = trainable_param_count(model)

    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    init_wangle = collect_angle_weight(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    # θ grad-norm trajectory: per epoch, mean over minibatches of the Frobenius norm
    # of the θ rows' gradient (and, for scale, of the full in_proj gradient).
    grad_traj = []
    best_val, best_state, patience, best_epoch = -1.0, None, 0, -1
    for epoch in range(cfg.EPOCHS):
        model.train()
        perm = np.random.permutation(n_train)
        ep_gn = [[] for _ in model.blocks]      # θ-row grad norm per layer
        ep_gn_all = [[] for _ in model.blocks]  # full in_proj grad norm per layer
        for i in range(0, n_train, cfg.BATCH_SEGMENTS):
            bidx = perm[i:i + cfg.BATCH_SEGMENTS]
            xb = (tr_raw[bidx] - mean[None, :, None]) / std[None, :, None]
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(tr_y[bidx]).to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                logits = model(xb)
            loss_fn(logits.float(), yb).backward()
            for li, blk in enumerate(model.blocks):
                m = blk.mamba; nra = angle_rows_slice(m)
                g = m.in_proj.weight.grad
                if g is not None:
                    ep_gn[li].append(float(g[-nra:, :].norm().item()))
                    ep_gn_all[li].append(float(g.norm().item()))
            opt.step()
        grad_traj.append({"epoch": epoch,
                          "theta_gradnorm_by_layer": [float(np.mean(x)) if x else None for x in ep_gn],
                          "inproj_gradnorm_by_layer": [float(np.mean(x)) if x else None for x in ep_gn_all]})
        vaccs = [subject_bal_acc(predict_proba(model, subj[s]["raw"], mean, std, device),
                                 subj[s]["label"], subj[s]["trial"]) for s in inner_val_ids]
        val = float(np.mean(vaccs))
        improved = val > best_val + 1e-4
        if improved:
            best_val = val; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        log.append(f"    [theta_gen] {fold_name} ep{epoch} val={val:.3f}{' *' if improved else ''} "
                   f"theta_gn={grad_traj[-1]['theta_gradnorm_by_layer']}")
        if patience >= cfg.EARLY_STOP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    d = subj[test_sid]
    p = predict_proba(model, d["raw"], mean, std, device)
    test_bacc = subject_bal_acc(p, d["label"], d["trial"])
    final_wangle = collect_angle_weight(model)

    torch.save({"fold": fold_name, "test_subject": test_sid,
                "init_state": init_state, "best_state": best_state,
                "channel_mean": mean, "channel_std": std,
                "inner_val_ids": inner_val_ids, "best_epoch": best_epoch},
               CKPT_DIR / f"{fold_name}.pt")

    log.append(f"  [theta_gen] {fold_name}: test={test_sid} bal_acc={test_bacc:.3f} "
               f"best_ep={best_epoch} n_params={n_params}")
    del model; torch.cuda.empty_cache()
    return {"fold": fold_name, "test_subject": test_sid, "subject_bal_acc": float(test_bacc),
            "best_inner_val": best_val, "best_epoch": best_epoch, "n_params": int(n_params),
            "init_w_angle": init_wangle, "final_w_angle": final_wangle,
            "grad_trajectory": grad_traj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--folds", default="", help="comma list of fold indices (e.g. 0,1,2); empty=all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    splits = json.load(open(cfg.SPLITS_JSON))
    fold_names = list(splits)
    if args.folds:
        want = set(int(x) for x in args.folds.split(","))
        fold_names = [fn for fn in fold_names if int(fn.split("_")[1]) in want]
    log, results = [], []
    t0 = time.time()
    for fn in fold_names:
        r = run_fold(fn, splits[fn], args.device, log)
        results.append(r); print(log[-1], flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        with open(args.log, "w") as f:
            f.write("\n".join(log) + "\n")
    print(f"[theta_gen] done {len(results)} folds in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
