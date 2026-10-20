"""
Task 4 Stage 2A: per-subject, per-condition, trial-cleaned averaged ERP,
cached per subject. Reads epochs_cache/*.npz and artifact_flags/*.npz
READ-ONLY (no trial dropped from those files, no modification). Builds,
for every (subject, condition, flag_set) combination, the mean epoch
(128, 200) in VOLTS over whichever trials survive that flag_set's
cleaning for that condition, plus the kept-trial count.

flag_set == "all_trials" means no exclusion at all (every trial in that
condition kept) -- included as one of the 4 robustness variants, not as
"the" primary choice (mad_k3 is primary per the Stage 2A spec: "只作
极端伪迹剔除").
"""
import numpy as np

import erp_config as ecfg
import stage2a_config as cfg

FLAGS_DIR = ecfg.TASK4_DIR / "artifact_flags"


def load_subject_raw(sid):
    d_epoch = np.load(ecfg.EPOCHS_CACHE / f"{sid}.npz")
    d_flag = np.load(FLAGS_DIR / f"{sid}.npz")
    return {
        "bc_epochs": d_epoch["bc_epochs"],  # (480, 128, 200) Volts
        "conditions": d_epoch["conditions"],  # (480,) f/h/s
        "ptp_100uv_flag": d_flag["ptp_100uv_flag"],
        "abs_100uv_flag": d_flag["abs_100uv_flag"],
        "mad_k3_flag": d_flag["mad_k3_flag"],
    }


def flag_array(raw, flag_set):
    if flag_set == cfg.FLAG_SET_ALL_TRIALS:
        return np.zeros(raw["bc_epochs"].shape[0], dtype=bool)
    return raw[f"{flag_set}_flag"]


def subject_condition_erps(sid):
    """Returns {(condition, flag_set): (erp_(128,200)_float32_volts, n_kept)}
    for all CONDITIONS x FLAG_SETS, cached to disk per subject."""
    cache_path = cfg.ERP_CACHE / f"{sid}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        out = {}
        for c in cfg.CONDITIONS:
            for fs in cfg.FLAG_SETS:
                out[(c, fs)] = (d[f"erp_{c}_{fs}"], int(d[f"n_{c}_{fs}"]))
        return out

    raw = load_subject_raw(sid)
    bc = raw["bc_epochs"]
    conditions = raw["conditions"]

    save_kwargs = {}
    out = {}
    for c in cfg.CONDITIONS:
        cond_mask = conditions == c
        for fs in cfg.FLAG_SETS:
            flags = flag_array(raw, fs)
            keep_mask = cond_mask & (~flags)
            n_kept = int(keep_mask.sum())
            if n_kept > 0:
                erp = bc[keep_mask].mean(axis=0).astype(np.float32)
            else:
                erp = np.full((ecfg.N_DATA_CH, ecfg.EPOCH_LEN), np.nan, dtype=np.float32)
            out[(c, fs)] = (erp, n_kept)
            save_kwargs[f"erp_{c}_{fs}"] = erp
            save_kwargs[f"n_{c}_{fs}"] = n_kept

    np.savez_compressed(cache_path, **save_kwargs)
    return out


def determine_primary_pool(stage1c_pool):
    """Stage 1c gave n=49. Stage 2A spec adds: any subject with <20
    mad_k3-cleaned trials in ANY condition is further excluded from the
    primary analysis (kept in sensitivity). Returns (primary_ids,
    excluded_by_min_trial: {sid: {condition: n_kept}})."""
    excluded = {}
    for sid in stage1c_pool:
        erps = subject_condition_erps(sid)
        low = {c: n for c in cfg.CONDITIONS for (cc, fs), (_, n) in erps.items()
               if cc == c and fs == "mad_k3" and n < cfg.MIN_TRIAL_THRESHOLD}
        if low:
            excluded[sid] = low
    primary_ids = sorted(set(stage1c_pool) - set(excluded))
    return primary_ids, excluded


if __name__ == "__main__":
    import pandas as pd
    all_sids = sorted(p.stem for p in FLAGS_DIR.glob("*.npz"))
    stage1c_pool = sorted(set(all_sids) - set(cfg.EXCLUDED_STAGE1C))
    print(f"Stage 1c pool: n={len(stage1c_pool)}")
    primary_ids, excluded = determine_primary_pool(stage1c_pool)
    print(f"additional min-trial(<{cfg.MIN_TRIAL_THRESHOLD}) exclusions: {len(excluded)}")
    for sid, low in excluded.items():
        print(f"  {sid}: {low}")
    print(f"final Stage 2A primary pool: n={len(primary_ids)}")
