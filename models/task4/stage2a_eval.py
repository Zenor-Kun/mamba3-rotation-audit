"""
Task 4 Stage 2A: subject-grouped CV + LR evaluation harness. Reuses the
zscore+LR pattern from task3_5/diagnostic_temporal_stats.py (unchanged
style), but the fold definitions are subject-id-filtered subsets of
splits_128rest.json (never modified, never re-derived) restricted to
whichever subject pool the caller passes in.

Pseudo-replication guard (see STATUS.md "Stage 2 hard prerequisite"): every
layer defined in stage2a_features.py produces exactly ONE feature row per
subject. This module's build_fold_arrays therefore ALWAYS gathers exactly
one X row (and one y) per subject id -- there is no (subject,condition) row
axis anywhere in this file. CV grouping, evaluation, and the resulting
balanced_acc/AUC are consequently already at subject level (n = pool size)
by construction, not via a post-hoc aggregation step.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import stage2a_config as cfg
from stage2a_features import subject_layer_features


def build_fold_ids(seed, fold, splits, valid_ids):
    fold_def = splits[str(seed)][str(fold)]
    valid = set(valid_ids)
    train_ids = [s for s in fold_def["train"] if s in valid]
    test_ids = [s for s in fold_def["test"] if s in valid]
    return train_ids, test_ids


def zscore_fit_apply(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def eval_logreg(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba > 0.5).astype(int)
    bal_acc = balanced_accuracy_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    return bal_acc, auc


def gather_layer_xy(ids, flag_set, roi_dict, layer, label_of):
    X, y = [], []
    for sid in ids:
        feats = subject_layer_features(sid, flag_set, roi_dict)
        X.append(feats[layer])
        y.append(label_of[sid])
    return np.stack(X), np.array(y, dtype=np.int64)


def run_layer_cv(pool_ids, flag_set, roi_dict, layer, label_of, splits):
    """Returns list of 25 (seed,fold) result dicts, subject-level metrics."""
    points = []
    for seed in cfg.SEEDS_FULL:
        for fold in range(cfg.N_FOLDS):
            train_ids, test_ids = build_fold_ids(seed, fold, splits, pool_ids)
            if len(train_ids) < 4 or len(test_ids) < 2:
                continue
            X_train, y_train = gather_layer_xy(train_ids, flag_set, roi_dict, layer, label_of)
            X_test, y_test = gather_layer_xy(test_ids, flag_set, roi_dict, layer, label_of)
            if len(np.unique(y_train)) < 2:
                continue
            Xn_train, Xn_test = zscore_fit_apply(X_train, X_test)
            bal_acc, auc = eval_logreg(Xn_train, y_train, Xn_test, y_test)
            points.append({
                "seed": seed, "fold": fold, "n_train": len(train_ids), "n_test": len(test_ids),
                "balanced_acc": bal_acc, "auc": auc,
            })
    return points


def gather_band_xy(ids, condition, flag_set, roi_dict, band, repr_kind, label_of):
    from stage2a_band_features import subject_band_features
    X, y = [], []
    for sid in ids:
        feats = subject_band_features(sid, condition, roi_dict, flag_set)
        X.append(feats[band][repr_kind])
        y.append(label_of[sid])
    return np.stack(X), np.array(y, dtype=np.int64)


def run_band_cv(pool_ids, condition, flag_set, roi_dict, band, repr_kind, label_of, splits):
    points = []
    for seed in cfg.SEEDS_FULL:
        for fold in range(cfg.N_FOLDS):
            train_ids, test_ids = build_fold_ids(seed, fold, splits, pool_ids)
            if len(train_ids) < 4 or len(test_ids) < 2:
                continue
            X_train, y_train = gather_band_xy(train_ids, condition, flag_set, roi_dict, band, repr_kind, label_of)
            X_test, y_test = gather_band_xy(test_ids, condition, flag_set, roi_dict, band, repr_kind, label_of)
            if len(np.unique(y_train)) < 2:
                continue
            Xn_train, Xn_test = zscore_fit_apply(X_train, X_test)
            bal_acc, auc = eval_logreg(Xn_train, y_train, Xn_test, y_test)
            points.append({"seed": seed, "fold": fold, "balanced_acc": bal_acc, "auc": auc})
    return points


def summarize_points(points):
    bal_accs = np.array([p["balanced_acc"] for p in points], dtype=np.float64)
    aucs = np.array([p["auc"] for p in points if not np.isnan(p["auc"])], dtype=np.float64)
    return {
        "n_points": len(points),
        "balanced_acc_mean": float(bal_accs.mean()), "balanced_acc_std": float(bal_accs.std()),
        "auc_mean": float(aucs.mean()) if len(aucs) else float("nan"),
        "auc_std": float(aucs.std()) if len(aucs) else float("nan"),
    }
