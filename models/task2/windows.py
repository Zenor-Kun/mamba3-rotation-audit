"""
Task 2 — windowing (fold-independent) + per-fold train-only z-score.

Windowing runs once on the cached continuous main_B signal (see preprocess.py):
  win_len=1000 (4s), step=500 (50% overlap), incomplete tail window dropped.

Normalization is NOT baked into the cached windows: mean/std depend on which
subjects are in the training split of a given (seed, fold), so z-score is
computed fresh per fold from the train subjects only and applied to both
train and test. This file exposes that as `build_fold_tensors`.
"""
import json
import numpy as np
import pandas as pd

import config as cfg


def window_subject(data):
    """data: (128, T) -> (n_windows, 128, WIN_LEN) float32."""
    T = data.shape[1]
    n_win = (T - cfg.WIN_LEN) // cfg.STEP + 1
    assert n_win >= 1, f"recording too short: T={T}"
    out = np.empty((n_win, cfg.N_DATA_CH, cfg.WIN_LEN), dtype=np.float32)
    for i in range(n_win):
        s = i * cfg.STEP
        out[i] = data[:, s : s + cfg.WIN_LEN]
    return out


def build_all_windows():
    """Load manifest, window every subject's cached signal, cache per-subject windows.
    Returns dict subject_id -> n_windows, and writes windows_cache/<sid>.npy."""
    df = pd.read_csv(cfg.MANIFEST_IN, dtype={"subject_id": str})
    counts = {}
    for sid in df["subject_id"]:
        cache_path = cfg.WINDOWS_CACHE / f"{sid}.npy"
        if cache_path.exists():
            counts[sid] = np.load(cache_path, mmap_mode="r").shape[0]
            continue
        d = np.load(cfg.PREPROC_CACHE / f"{sid}.npz")
        windows = window_subject(d["data"])
        np.save(cache_path, windows)
        counts[sid] = windows.shape[0]
    return counts


def load_splits():
    with open(cfg.SPLITS_JSON) as f:
        return json.load(f)


def load_subject_windows(sid):
    return np.load(cfg.WINDOWS_CACHE / f"{sid}.npy")


def build_fold_tensors(seed, fold, manifest_df, splits):
    """
    Returns dict with:
      X_train (N_tr, 128, 1000) float32, y_train (N_tr,), subj_train (N_tr,)
      X_test  (N_te, 128, 1000) float32, y_test  (N_te,), subj_test  (N_te,)
      mean (128,), std (128,)   -- computed from TRAIN windows only
    """
    fold_def = splits[str(seed)][str(fold)]
    train_ids, test_ids = fold_def["train"], fold_def["test"]

    label_of = dict(zip(manifest_df["subject_id"], manifest_df["label"]))

    def gather(ids):
        xs, ys, subj = [], [], []
        for sid in ids:
            w = load_subject_windows(sid)
            xs.append(w)
            ys.append(np.full(w.shape[0], label_of[sid], dtype=np.int64))
            subj.append(np.array([sid] * w.shape[0]))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(subj)

    X_train, y_train, subj_train = gather(train_ids)
    X_test, y_test, subj_test = gather(test_ids)

    # train-only per-channel z-score
    mean = X_train.mean(axis=(0, 2))
    std = X_train.std(axis=(0, 2)) + 1e-8

    X_train_n = (X_train - mean[None, :, None]) / std[None, :, None]
    X_test_n = (X_test - mean[None, :, None]) / std[None, :, None]

    return {
        "X_train": X_train_n.astype(np.float32), "y_train": y_train, "subj_train": subj_train,
        "X_test": X_test_n.astype(np.float32), "y_test": y_test, "subj_test": subj_test,
        "mean": mean, "std": std,
        "train_ids": train_ids, "test_ids": test_ids,
    }


if __name__ == "__main__":
    counts = build_all_windows()
    vals = np.array(list(counts.values()))
    print(f"windows per subject: min={vals.min()} median={int(np.median(vals))} max={vals.max()}")
    print(f"total windows across {len(counts)} subjects: {vals.sum()}")
    for sid, n in counts.items():
        if n < 149 or n > 167:
            print(f"  OUT OF EXPECTED RANGE: {sid} -> {n} windows")
