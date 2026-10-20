"""
Stage 2 training: one theta-variant across LOSO 15-fold, identical protocol /
recipe / front-end to Stage 1 (only the model builder differs -> fair ablation).
Reuses Stage 1's data + training helpers unchanged.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

import seed_mamba_config as cfg
from seed_mamba_train import (load_subject, channel_stats, train_subsample_idx,
                              labels_to_idx, subject_bal_acc, predict_proba, DTYPE)
from seed_variant_model import build_variant, trainable_param_count


def run_fold(fold_name, fold, variant, device, log):
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
    model = build_variant(variant).to(device)
    n_params = trainable_param_count(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, patience = -1.0, None, 0
    for epoch in range(cfg.EPOCHS):
        model.train()
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, cfg.BATCH_SEGMENTS):
            bidx = perm[i:i + cfg.BATCH_SEGMENTS]
            xb = (tr_raw[bidx] - mean[None, :, None]) / std[None, :, None]
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(tr_y[bidx]).to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                logits = model(xb)
            loss_fn(logits.float(), yb).backward(); opt.step()
        vaccs = [subject_bal_acc(predict_proba(model, subj[s]["raw"], mean, std, device),
                                 subj[s]["label"], subj[s]["trial"]) for s in inner_val_ids]
        val = float(np.mean(vaccs))
        improved = val > best_val + 1e-4
        if improved:
            best_val = val; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        log.append(f"    [{variant}] {fold_name} ep{epoch} inner_val={val:.3f}{' *' if improved else ''}")
        if patience >= cfg.EARLY_STOP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    d = subj[test_sid]
    p = predict_proba(model, d["raw"], mean, std, device)
    test_bacc = subject_bal_acc(p, d["label"], d["trial"])
    log.append(f"  [{variant}] {fold_name}: test={test_sid} bal_acc={test_bacc:.3f} n_params={n_params}")
    del model; torch.cuda.empty_cache()
    return {"fold": fold_name, "test_subject": test_sid, "subject_bal_acc": float(test_bacc),
            "best_inner_val": best_val, "n_params": int(n_params)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1_full", "V2_theta_zero", "V3_theta_frozen_random", "V4_real_only"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    splits = json.load(open(cfg.SPLITS_JSON))
    log, results = [], []
    t0 = time.time()
    for fold_name in splits:
        r = run_fold(fold_name, splits[fold_name], args.variant, args.device, log)
        results.append(r); print(log[-1], flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        with open(args.log, "w") as f:
            f.write("\n".join(log) + "\n")
    print(f"[{args.variant}] done {len(results)} folds in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
