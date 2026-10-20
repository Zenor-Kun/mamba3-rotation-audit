"""
Task 3 / Probe 1 training loop -- identical recipe to Task 2's V1 training
(same batch size, lr, weight decay, dropout, epochs, early-stopping patience,
inner-val fraction carved from TRAIN subjects only). The only thing that
varies across calls is `variant`, which selects the model via
variant_model.build_classifier. This is what makes the ablation fair.
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import probe1_config as pcfg
import windows  # task2's windows.py (build_fold_tensors, load_splits) -- unmodified
from variant_model import build_classifier, trainable_param_count


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
    for i in range(0, X.shape[0], pcfg.BATCH_SIZE):
        xb = X[i: i + pcfg.BATCH_SIZE].to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def run_fold(variant, seed, fold, manifest_df, splits, device):
    ft = windows.build_fold_tensors(seed, fold, manifest_df, splits)

    train_ids = ft["train_ids"]
    labels_by_subj = dict(zip(manifest_df["subject_id"], manifest_df["label"]))
    inner_train_ids, inner_val_ids = train_test_split(
        train_ids,
        test_size=pcfg.INNER_VAL_FRAC,
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

    model = build_classifier(variant).to(device)
    n_params = trainable_param_count(model)
    opt = torch.optim.AdamW(model.parameters(), lr=pcfg.LR, weight_decay=pcfg.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    n_train = X_itr.shape[0]
    best_val_bal_acc = -1.0
    best_state = None
    epochs_since_best = 0

    for epoch in range(pcfg.EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, pcfg.BATCH_SIZE):
            idx = perm[i: i + pcfg.BATCH_SIZE]
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

        if epoch % 5 == 0 or improved or epoch == pcfg.EPOCHS - 1:
            print(f"    [{variant}] fold {fold} epoch {epoch:2d}: train_loss={epoch_loss:.4f} "
                  f"inner_val_bal_acc={val_bal_acc:.3f}{' *' if improved else ''}")

        if epochs_since_best >= pcfg.EARLY_STOP_PATIENCE:
            print(f"    [{variant}] fold {fold}: early stop at epoch {epoch} (best inner_val_bal_acc={best_val_bal_acc:.3f})")
            break

    model.load_state_dict(best_state)
    test_logits = predict_logits(model, X_test, device)
    bal_acc, auc, n_subj = subject_level_eval(test_logits, y_test, subj_test)
    return bal_acc, auc, n_subj, best_val_bal_acc, n_params
