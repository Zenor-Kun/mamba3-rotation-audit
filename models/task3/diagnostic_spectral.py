"""
Probe 1 diagnostic (cheap, decisive, run BEFORE deciding anything about the
theta mechanism): does this data have any subject-independent-extractable
signal at all, using a classic band-power spectral representation instead of
raw waveform?

Deliberately NOT touching Mamba-3, NOT touching phase (magnitude/power only --
phase is Probe 2's territory, out of scope here same as it was for Probe 1).
Same splits_128rest.json, same main_B+notch windows, same window-level train /
subject-level aggregation as Task 2/3, so the answer is directly comparable.

Two classifiers on the same 640-dim feature (128 channels x 5 bands, log
power): logistic regression (no overfitting risk, sanity floor) and a small
MLP (in case the boundary is mildly nonlinear). Both subject-independent
5-fold, seed=42 only -- this is a yes/no diagnostic, not a results run.

Decision this feeds (see STATUS.md):
  - at chance  -> the difficulty is data/subject-independent-split, not the
    model; Probe 1 being uninterpretable is a real finding, consider route (3).
  - clearly >0.5 -> signal exists; Mamba-3's raw-waveform input just isn't
    reaching it; rebuild V1 on a phase-preserving spectral representation
    (route 1) and redo Probe 1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

TASK2_DIR = Path("/root/newcode/task2")
TASK3_DIR = Path("/root/newcode/task3")
sys.path.insert(0, str(TASK2_DIR))
import config as task2_cfg  # noqa: E402
import windows  # noqa: E402

FEATURES_CACHE = TASK3_DIR / "diagnostic_features_cache"
FEATURES_CACHE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = TASK3_DIR / "results"

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
SEED = 42
N_FOLDS = task2_cfg.N_FOLDS
INNER_VAL_FRAC = task2_cfg.INNER_VAL_FRAC
DEVICE = task2_cfg.DEVICE


def band_power_features(window, fs=task2_cfg.FS):
    """window: (128, 1000) raw (main_B+notch, pre-zscore) -> (640,) log band power."""
    freqs, psd = welch(window, fs=fs, nperseg=window.shape[-1], axis=-1)  # psd: (128, 501)
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        band_power = psd[:, mask].sum(axis=1)  # (128,)
        feats.append(np.log(band_power + 1e-12))
    return np.concatenate(feats).astype(np.float32)  # (640,)


def build_subject_features(sid):
    cache_path = FEATURES_CACHE / f"{sid}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    win = windows.load_subject_windows(sid)  # (n_win, 128, 1000)
    feats = np.stack([band_power_features(w) for w in win])  # (n_win, 640)
    np.save(cache_path, feats)
    return feats


def build_all_features():
    df = pd.read_csv(task2_cfg.MANIFEST_IN, dtype={"subject_id": str})
    for sid in df["subject_id"]:
        build_subject_features(sid)
    print(f"feature cache ready: {len(list(FEATURES_CACHE.glob('*.npy')))} subjects")


def build_fold_feature_tensors(seed, fold, manifest_df, splits):
    fold_def = splits[str(seed)][str(fold)]
    train_ids, test_ids = fold_def["train"], fold_def["test"]
    label_of = dict(zip(manifest_df["subject_id"], manifest_df["label"]))

    def gather(ids):
        xs, ys, subj = [], [], []
        for sid in ids:
            f = build_subject_features(sid)
            xs.append(f)
            ys.append(np.full(f.shape[0], label_of[sid], dtype=np.int64))
            subj.append(np.array([sid] * f.shape[0]))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(subj)

    X_train, y_train, subj_train = gather(train_ids)
    X_test, y_test, subj_test = gather(test_ids)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    return {
        "X_train": X_train_n.astype(np.float32), "y_train": y_train, "subj_train": subj_train,
        "X_test": X_test_n.astype(np.float32), "y_test": y_test, "subj_test": subj_test,
        "train_ids": train_ids, "test_ids": test_ids,
    }


def subject_level_eval(scores, labels, subj_ids, is_proba):
    d = pd.DataFrame({"score": scores, "label": labels, "subj": subj_ids})
    agg = d.groupby("subj").agg(mean_score=("score", "mean"), label=("label", "first"))
    threshold = 0.5 if is_proba else 0.0
    pred = (agg["mean_score"] > threshold).astype(int)
    bal_acc = balanced_accuracy_score(agg["label"], pred)
    try:
        auc = roc_auc_score(agg["label"], agg["mean_score"])
    except ValueError:
        auc = float("nan")
    return bal_acc, auc, len(agg)


def run_logreg_fold(ft):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(ft["X_train"], ft["y_train"])
    test_proba = clf.predict_proba(ft["X_test"])[:, 1]
    return subject_level_eval(test_proba, ft["y_test"], ft["subj_test"], is_proba=True)


class TinyMLP(nn.Module):
    def __init__(self, in_dim=640, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_mlp_fold(ft, manifest_df, seed, epochs=60, patience=8, lr=1e-3, wd=1e-2, batch_size=64):
    train_ids = ft["train_ids"]
    labels_by_subj = dict(zip(manifest_df["subject_id"], manifest_df["label"]))
    inner_train_ids, inner_val_ids = train_test_split(
        train_ids, test_size=INNER_VAL_FRAC, random_state=seed,
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

    model = TinyMLP(in_dim=ft["X_train"].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_bal_acc, best_state, epochs_since_best = -1.0, None, 0
    n = X_itr.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_itr[idx].to(DEVICE), y_itr[idx].to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_ival.to(DEVICE)).cpu().numpy()
        val_bal_acc, _, _ = subject_level_eval(val_logits, y_ival, subj_ival, is_proba=False)
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if epochs_since_best >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(DEVICE)).cpu().numpy()
    return subject_level_eval(test_logits, ft["y_test"], ft["subj_test"], is_proba=False)


def main():
    build_all_features()
    manifest_df = pd.read_csv(task2_cfg.MANIFEST_IN, dtype={"subject_id": str})
    splits = windows.load_splits()

    logreg_results, mlp_results = [], []
    for fold in range(N_FOLDS):
        ft = build_fold_feature_tensors(SEED, fold, manifest_df, splits)

        lr_bal, lr_auc, n_subj = run_logreg_fold(ft)
        print(f"fold {fold} [logreg]: balanced_acc={lr_bal:.3f} auc={lr_auc:.3f} (n_test_subj={n_subj})")
        logreg_results.append({"fold": fold, "balanced_acc": lr_bal, "auc": lr_auc, "n_test_subj": n_subj})

        mlp_bal, mlp_auc, n_subj = run_mlp_fold(ft, manifest_df, seed=SEED)
        print(f"fold {fold} [mlp]:    balanced_acc={mlp_bal:.3f} auc={mlp_auc:.3f} (n_test_subj={n_subj})")
        mlp_results.append({"fold": fold, "balanced_acc": mlp_bal, "auc": mlp_auc, "n_test_subj": n_subj})

    def summarize(results):
        bal = [r["balanced_acc"] for r in results]
        auc = [r["auc"] for r in results]
        return {"fold_results": results,
                "mean_balanced_acc": float(np.mean(bal)), "std_balanced_acc": float(np.std(bal)),
                "mean_auc": float(np.mean(auc)), "std_auc": float(np.std(auc))}

    summary = {
        "seed": SEED,
        "features": "128ch x 5 bands (delta/theta/alpha/beta/gamma) log power, train-fold z-score",
        "logreg": summarize(logreg_results),
        "mlp": summarize(mlp_results),
    }

    import json
    out_path = RESULTS_DIR / "diagnostic_spectral_seed42.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Probe 1 diagnostic: does spectral-magnitude signal exist? (seed=42, 5-fold) ===")
    print(f"logreg: balanced_acc={summary['logreg']['mean_balanced_acc']:.3f}+/-{summary['logreg']['std_balanced_acc']:.3f}  "
          f"auc={summary['logreg']['mean_auc']:.3f}+/-{summary['logreg']['std_auc']:.3f}")
    print(f"mlp:    balanced_acc={summary['mlp']['mean_balanced_acc']:.3f}+/-{summary['mlp']['std_balanced_acc']:.3f}  "
          f"auc={summary['mlp']['mean_auc']:.3f}+/-{summary['mlp']['std_auc']:.3f}")
    print(f"\nwrote {out_path}")
    return summary


if __name__ == "__main__":
    main()
