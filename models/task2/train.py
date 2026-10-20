"""
Task 2 V1 link-check training: seed=42, 5-fold, subject-independent.
Window-level training, subject-level aggregation (mean logit) for evaluation.

Early stopping uses an INNER validation split carved out of the fold's train
subjects only (never the held-out test fold) -- this is picking when to stop
overfitting using train-only data, not tuning against the test signal.
"""
import copy
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config as cfg
import runs
from windows import build_fold_tensors, load_splits
from model import MambaV1Classifier


def subject_level_eval(logits, labels, subj_ids):
    df = pd.DataFrame({"logit": logits, "label": labels, "subj": subj_ids})
    agg = df.groupby("subj").agg(mean_logit=("logit", "mean"), label=("label", "first"))
    pred = (agg["mean_logit"] > 0).astype(int)
    bal_acc = balanced_accuracy_score(agg["label"], pred)
    try:
        auc = roc_auc_score(agg["label"], agg["mean_logit"])
    except ValueError:
        auc = float("nan")
    return bal_acc, auc, len(agg)


@torch.no_grad()
def predict_logits(model, X, device):
    model.eval()
    out = []
    for i in range(0, X.shape[0], cfg.BATCH_SIZE):
        xb = X[i : i + cfg.BATCH_SIZE].to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def run_fold(seed, fold, manifest_df, splits, device):
    ft = build_fold_tensors(seed, fold, manifest_df, splits)

    train_ids = ft["train_ids"]
    labels_by_subj = dict(zip(manifest_df["subject_id"], manifest_df["label"]))
    inner_train_ids, inner_val_ids = train_test_split(
        train_ids,
        test_size=cfg.INNER_VAL_FRAC,
        random_state=seed,
        stratify=[labels_by_subj[s] for s in train_ids],
    )
    inner_train_ids, inner_val_ids = set(inner_train_ids), set(inner_val_ids)

    subj_train = ft["subj_train"]
    mask_itr = np.isin(subj_train, list(inner_train_ids))
    mask_ival = np.isin(subj_train, list(inner_val_ids))

    X_itr = torch.from_numpy(ft["X_train"][mask_itr])
    y_itr = torch.from_numpy(ft["y_train"][mask_itr]).float()
    X_ival = torch.from_numpy(ft["X_train"][mask_ival])
    y_ival = ft["y_train"][mask_ival]
    subj_ival = subj_train[mask_ival]

    X_test = torch.from_numpy(ft["X_test"])
    y_test = ft["y_test"]
    subj_test = ft["subj_test"]

    model = MambaV1Classifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    n_train = X_itr.shape[0]
    best_val_bal_acc = -1.0
    best_state = None
    epochs_since_best = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, cfg.BATCH_SIZE):
            idx = perm[i : i + cfg.BATCH_SIZE]
            xb = X_itr[idx].to(device, non_blocking=True)
            yb = y_itr[idx].to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= n_train

        val_logits = predict_logits(model, X_ival, device)
        val_bal_acc, val_auc, _ = subject_level_eval(val_logits, y_ival, subj_ival)

        improved = val_bal_acc > best_val_bal_acc
        if improved:
            best_val_bal_acc = val_bal_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch % 5 == 0 or improved or epoch == cfg.EPOCHS - 1:
            print(f"    fold {fold} epoch {epoch:2d}: train_loss={epoch_loss:.4f} "
                  f"inner_val_bal_acc={val_bal_acc:.3f}{' *' if improved else ''}")

        if epochs_since_best >= cfg.EARLY_STOP_PATIENCE:
            print(f"    fold {fold}: early stop at epoch {epoch} (best inner_val_bal_acc={best_val_bal_acc:.3f})")
            break

    model.load_state_dict(best_state)
    test_logits = predict_logits(model, X_test, device)
    bal_acc, auc, n_subj = subject_level_eval(test_logits, y_test, subj_test)
    return bal_acc, auc, n_subj, best_val_bal_acc


class Tee:
    """Mirror stdout to a file so every run self-archives its own log, regardless
    of whether the caller also redirects/tees the shell output."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    device = cfg.DEVICE
    manifest_df = pd.read_csv(cfg.MANIFEST_IN, dtype={"subject_id": str})
    splits = load_splits()

    # every run gets its own immutable timestamped dir under runs/ -- never
    # overwritten or deleted; results/*.json below is just a "latest" pointer.
    run_dir = runs.new_run_dir(f"v1_seed{cfg.SEED}")
    runs.write_snapshot(run_dir)
    log_file = open(run_dir / "train_run.log", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    results = []
    t0 = time.time()
    for fold in range(cfg.N_FOLDS):
        print(f"  fold {fold} start")
        bal_acc, auc, n_subj, inner_val_bal_acc = run_fold(cfg.SEED, fold, manifest_df, splits, device)
        print(f"  fold {fold}: TEST subject-level balanced_acc={bal_acc:.3f} auc={auc:.3f} "
              f"(n_test_subj={n_subj}, inner_val_bal_acc={inner_val_bal_acc:.3f})")
        results.append({"fold": fold, "balanced_acc": bal_acc, "auc": auc,
                         "n_test_subj": n_subj, "inner_val_bal_acc": inner_val_bal_acc})

    bal_accs = [r["balanced_acc"] for r in results]
    aucs = [r["auc"] for r in results]
    summary = {
        "seed": cfg.SEED,
        "results": results,
        "mean_balanced_acc": float(np.mean(bal_accs)),
        "std_balanced_acc": float(np.std(bal_accs)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "elapsed_sec": time.time() - t0,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(cfg.RESULTS_DIR / "v1_seed42_report.json", "w") as f:  # latest pointer, safe to overwrite
        json.dump(summary, f, indent=2)

    print(f"\nrun archived to {run_dir}")
    print(f"V1 Full baseline, seed={cfg.SEED}, 5-fold subject-level:")
    print(f"  balanced_acc: mean={summary['mean_balanced_acc']:.3f} std={summary['std_balanced_acc']:.3f}")
    print(f"  auc:          mean={summary['mean_auc']:.3f} std={summary['std_auc']:.3f}")
    print(f"  elapsed: {summary['elapsed_sec']:.1f}s")

    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
