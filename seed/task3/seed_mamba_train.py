"""
SEED Stage 1 training: LOSO 15-fold, one V1_full model trained per fold, for a
given condition (full / drop_gamma). Segment-level 3-class training, segment
predictions aggregated to subject level (mean class-prob over a subject's
segments -> per-trial -> subject balanced_accuracy over 45 trials).

Leak controls:
  - LOSO test subject never in train.
  - inner-val for early stopping = a few TRAIN subjects held out (subject-level,
    never the test subject).
  - per-channel z-score stats computed on TRAIN segments only, applied to all.

Compute control: training uses <= TRAIN_SEG_PER_TRIAL_CAP evenly-spaced
segments per trial (channel-folding multiplies step cost x62). Eval / inner-val
use ALL segments.
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

import seed_mamba_config as cfg
from seed_mamba_model import build_model, trainable_param_count

DTYPE = torch.bfloat16


def load_subject(sid):
    d = np.load(cfg.SEGMENTS_CACHE / f"{sid}.npz")
    return {"raw": d["raw_segments"], "label": d["seg_label"], "trial": d["seg_trial_id"]}


def train_subsample_idx(labels, trials):
    """<= cap evenly-spaced segment indices per trial."""
    idx = []
    for tid in np.unique(trials):
        pos = np.where(trials == tid)[0]
        if len(pos) > cfg.TRAIN_SEG_PER_TRIAL_CAP:
            sel = np.linspace(0, len(pos) - 1, cfg.TRAIN_SEG_PER_TRIAL_CAP).round().astype(int)
            pos = pos[np.unique(sel)]
        idx.append(pos)
    return np.concatenate(idx)


def channel_stats(train_arrays):
    """per-channel mean/std over all train segments' samples. arrays: list of (n_seg,62,800)."""
    s = np.zeros(cfg.N_CH, np.float64); ss = np.zeros(cfg.N_CH, np.float64); cnt = 0
    for a in train_arrays:
        s += a.sum(axis=(0, 2)); ss += (a.astype(np.float64) ** 2).sum(axis=(0, 2))
        cnt += a.shape[0] * a.shape[2]
    mean = s / cnt
    std = np.sqrt(np.maximum(ss / cnt - mean ** 2, 1e-24))
    return mean.astype(np.float32), std.astype(np.float32)


def labels_to_idx(y):
    return np.array([cfg.LABEL_TO_IDX[int(v)] for v in y], dtype=np.int64)


def subject_bal_acc(proba, labels, trials):
    """aggregate segment probs -> per-trial argmax -> balanced_acc over trials."""
    y = labels_to_idx(labels); tt, tp = [], []
    for u in np.unique(trials):
        m = trials == u; tp.append(proba[m].mean(0).argmax()); tt.append(y[m][0])
    return balanced_accuracy_score(tt, tp)


@torch.no_grad()
def predict_proba(model, raw, mean, std, device, batch=64):
    model.eval()
    out = []
    for i in range(0, raw.shape[0], batch):
        xb = (raw[i:i + batch] - mean[None, :, None]) / std[None, :, None]
        xb = torch.from_numpy(xb).to(device)  # fp32; bf16 handled by autocast inside
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            logits = model(xb)
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def run_fold(fold_name, fold, condition, device, log=None):
    log = log if log is not None else []
    drop_gamma = (condition == "drop_gamma")
    test_sid = int(fold["test"][0]); train_ids = [int(s) for s in fold["train"]]
    rng = np.random.default_rng(1234 + test_sid)
    inner_val_ids = list(rng.choice(train_ids, size=cfg.INNER_VAL_SUBJECTS, replace=False))
    inner_train_ids = [s for s in train_ids if s not in inner_val_ids]

    subj = {s: load_subject(s) for s in train_ids + [test_sid]}
    mean, std = channel_stats([subj[s]["raw"] for s in inner_train_ids])

    # build train pool (subsampled per trial), pooled across inner-train subjects
    tr_raw, tr_lab = [], []
    for s in inner_train_ids:
        d = subj[s]; sel = train_subsample_idx(d["label"], d["trial"])
        tr_raw.append(d["raw"][sel]); tr_lab.append(d["label"][sel])
    tr_raw = np.concatenate(tr_raw); tr_y = labels_to_idx(np.concatenate(tr_lab))
    n_train = tr_raw.shape[0]

    torch.manual_seed(1234 + test_sid)
    model = build_model(drop_gamma=drop_gamma).to(device)
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
            xb = torch.from_numpy(xb).to(device)  # fp32; bf16 via autocast
            yb = torch.from_numpy(tr_y[bidx]).to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                logits = model(xb)
            loss = loss_fn(logits.float(), yb)
            loss.backward(); opt.step()
        # inner-val: subject-level bal_acc on held-out train subjects (all segments)
        vaccs = []
        for s in inner_val_ids:
            d = subj[s]; p = predict_proba(model, d["raw"], mean, std, device)
            vaccs.append(subject_bal_acc(p, d["label"], d["trial"]))
        val = float(np.mean(vaccs))
        improved = val > best_val + 1e-4
        if improved:
            best_val = val; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        log.append(f"    [{condition}] {fold_name} ep{epoch} loss={loss.item():.3f} inner_val_bal_acc={val:.3f}{' *' if improved else ''}")
        if patience >= cfg.EARLY_STOP_PATIENCE:
            log.append(f"    [{condition}] {fold_name} early stop ep{epoch} (best inner_val={best_val:.3f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    d = subj[test_sid]
    p = predict_proba(model, d["raw"], mean, std, device)
    test_bacc = subject_bal_acc(p, d["label"], d["trial"])
    log.append(f"  [{condition}] {fold_name}: test_subj={test_sid} subject_bal_acc={test_bacc:.3f} n_params={n_params} n_train_seg={n_train}")
    del model
    torch.cuda.empty_cache()
    return {"fold": fold_name, "test_subject": test_sid, "subject_bal_acc": float(test_bacc),
            "best_inner_val": best_val, "n_params": int(n_params), "n_train_seg": int(n_train)}


def run_folds(fold_items, condition, device, out_path, log_path):
    splits = json.load(open(cfg.SPLITS_JSON))
    log, results = [], []
    t0 = time.time()
    for fold_name in fold_items:
        r = run_fold(fold_name, splits[fold_name], condition, device, log)
        results.append(r)
        print(log[-1]);
        with open(log_path, "w") as f:
            f.write("\n".join(log) + "\n")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    print(f"[{condition}] {device} done {len(results)} folds in {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["full", "drop_gamma"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--folds", default="all", help="comma list of fold names or 'all'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    splits = json.load(open(cfg.SPLITS_JSON))
    folds = list(splits.keys()) if args.folds == "all" else args.folds.split(",")
    run_folds(folds, args.condition, args.device, args.out, args.log)
