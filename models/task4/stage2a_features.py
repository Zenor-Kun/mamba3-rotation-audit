"""
Task 4 Stage 2A: low-dimensional ROI x time-window feature extraction from
per-subject per-condition trial-averaged ERPs (never touches the raw
76800-dim [128ch x 200samp x 3cond] space directly for classification).

Three layers, each producing exactly ONE feature row per subject (never a
per-condition row -- this is what keeps every downstream CV/metric
naturally at subject-level n, see the pseudo-replication guard in
STATUS.md):
  1. "overall": condition-AVERAGED ERP (mean of f,h,s ERPs) -> ROI x window
     mean(|amplitude|) features. Tests MDD vs HC collapsed across emotion.
  2. per-condition (f / h / s): ERP for that single condition only -> same
     feature scheme. Tests which single emotional condition is most
     separable.
  3. per-contrast (f-h / s-h / f-s): POINTWISE difference of two condition
     ERPs, then ROI x window mean(|diff|) features. Tests whether
     within-subject emotional MODULATION carries more signal than any
     single condition's absolute level.

Feature vector layout for a given channel-set variant with ROI dict
{roi_name: idx}: for roi_name in sorted(roi_dict), for window in
TIME_WINDOWS (fixed order) -> one scalar = mean(|erp[roi_idx, lo:hi]|).
"""
import numpy as np

import stage2a_config as cfg
from stage2a_data import subject_condition_erps


def erp_roi_window_features(erp_uv, roi_dict):
    """erp_uv: (128, 200) array (any unit -- caller decides). roi_dict:
    {roi_name: channel_idx_array}. Returns 1D array, len = n_roi * n_windows,
    each = mean(|erp|) over that ROI's channels and that window's samples."""
    feats = []
    for roi_name in sorted(roi_dict):
        idx = roi_dict[roi_name]
        sub = erp_uv[idx]  # (n_ch_roi, 200)
        for _wname, lo, hi in cfg.TIME_WINDOWS:
            feats.append(float(np.mean(np.abs(sub[:, lo:hi]))))
    return np.array(feats, dtype=np.float64)


def feature_names(roi_dict):
    names = []
    for roi_name in sorted(roi_dict):
        for wname, _lo, _hi in cfg.TIME_WINDOWS:
            names.append(f"{roi_name}__{wname}")
    return names


def subject_layer_features(sid, flag_set, roi_dict):
    """Returns {layer_name: feature_vector} for one subject, one flag_set,
    one channel-set variant. layer_name in {"overall","f","h","s",
    "f_minus_h","s_minus_h","f_minus_s"}."""
    erps = subject_condition_erps(sid)
    erp_of = {c: erps[(c, flag_set)][0] * 1e6 for c in cfg.CONDITIONS}  # -> microvolts

    out = {}
    overall = np.mean([erp_of[c] for c in cfg.CONDITIONS], axis=0)
    out["overall"] = erp_roi_window_features(overall, roi_dict)
    for c in cfg.CONDITIONS:
        out[c] = erp_roi_window_features(erp_of[c], roi_dict)
    for name, a, b in cfg.CONTRASTS:
        diff = erp_of[a] - erp_of[b]
        out[name] = erp_roi_window_features(diff, roi_dict)
    return out


LAYER_NAMES = ["overall"] + list(cfg.CONDITIONS) + [c[0] for c in cfg.CONTRASTS]
