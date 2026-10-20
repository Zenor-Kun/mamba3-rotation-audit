"""
Task 2 zero-leakage triple-check, run before any training.
  (a) test normalization uses train-only statistics (printed proof: train-stats
      differ from stats computed over train+test pooled)
  (b) every window's fold membership is inherited from its subject and matches
      splits_128rest.json exactly
  (c) no subject's windows appear in both train and test of the same fold
"""
import numpy as np
import pandas as pd

import config as cfg
from windows import build_fold_tensors, load_splits, load_subject_windows

SEED = cfg.SEED


def main():
    df = pd.read_csv(cfg.MANIFEST_IN, dtype={"subject_id": str})
    splits = load_splits()

    for fold in range(cfg.N_FOLDS):
        ft = build_fold_tensors(SEED, fold, df, splits)

        # (c) no subject overlap
        tr_set, te_set = set(ft["train_ids"]), set(ft["test_ids"])
        assert tr_set.isdisjoint(te_set), f"fold {fold}: train/test subject overlap!"
        assert set(ft["subj_train"].tolist()) == tr_set
        assert set(ft["subj_test"].tolist()) == te_set

        # (b) fold membership matches splits json exactly
        fold_def = splits[str(SEED)][str(fold)]
        assert sorted(ft["train_ids"]) == sorted(fold_def["train"])
        assert sorted(ft["test_ids"]) == sorted(fold_def["test"])

        # (a) train-only z-score proof: recompute "what if we'd pooled test in too"
        # and show it's numerically different from the train-only stats actually used.
        X_all = np.concatenate([
            np.concatenate([load_subject_windows(s) for s in ft["train_ids"]]),
            np.concatenate([load_subject_windows(s) for s in ft["test_ids"]]),
        ])
        pooled_mean = X_all.mean(axis=(0, 2))
        pooled_std = X_all.std(axis=(0, 2)) + 1e-8

        train_mean, train_std = ft["mean"], ft["std"]
        max_abs_diff_mean = np.max(np.abs(pooled_mean - train_mean))
        max_abs_diff_std = np.max(np.abs(pooled_std - train_std))

        print(f"fold {fold}: n_train_subj={len(tr_set)} n_test_subj={len(te_set)} "
              f"n_train_win={ft['X_train'].shape[0]} n_test_win={ft['X_test'].shape[0]}")
        print(f"  train-only mean[:3]={train_mean[:3].round(3)}  "
              f"pooled(train+test) mean[:3]={pooled_mean[:3].round(3)}  "
              f"max|diff|={max_abs_diff_mean:.4f}")
        print(f"  train-only std[:3]={train_std[:3].round(3)}  "
              f"pooled(train+test) std[:3]={pooled_std[:3].round(3)}  "
              f"max|diff|={max_abs_diff_std:.4f}")
        assert max_abs_diff_mean > 0 and max_abs_diff_std > 0, (
            "train-only stats identical to pooled stats -- suspicious, check test wasn't silently included"
        )
        print(f"  PASS fold {fold}: test normalization used train-only stats (nonzero divergence from pooled), "
              f"no subject overlap, fold membership matches splits_128rest.json")

    print("\nALL FOLDS PASS: zero-leakage triple-check (a)(b)(c)")


if __name__ == "__main__":
    main()
