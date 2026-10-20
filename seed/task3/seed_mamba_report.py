"""
Combine the two condition results (full / drop_gamma) for SEED Stage 1 into a
single archived report: subject-level balanced_acc mean+-std over the 15 LOSO
folds for each condition, and the drop-gamma vs full comparison. Archives to
seed/runs/.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import seed_mamba_config as cfg


def summarize(path):
    rows = json.load(open(path))
    rows = sorted(rows, key=lambda r: int(r["fold"].split("_")[1]))
    baccs = [r["subject_bal_acc"] for r in rows]
    return {"n_folds": len(rows), "mean": float(np.mean(baccs)), "std": float(np.std(baccs)),
            "per_fold": rows}


def main(full_json, dropg_json):
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_seed_mamba_stage1_v1_full"
    run_dir.mkdir(parents=True)

    full = summarize(full_json)
    dropg = summarize(dropg_json)
    delta = full["mean"] - dropg["mean"]

    report = {
        "protocol": "LOSO 15-fold, V1_full (stock Mamba3, MIMO rank4, bf16), complex-STFT front-end; "
                    "segment->trial->subject balanced_acc (chance 0.333)",
        "frontend": {"stft_n_fft": cfg.STFT_N_FFT, "stft_hop": cfg.STFT_HOP, "n_frames_seq_len": cfg.SEQ_LEN,
                     "effective_frame_rate_hz": cfg.EFFECTIVE_FRAME_RATE_HZ,
                     "theta_to_hz_const": cfg.THETA_TO_HZ_CONST,
                     "kept_freq_bins": cfg.N_FREQ_BINS, "bins_per_band": None},
        "mamba": {"is_mimo": cfg.IS_MIMO, "mimo_rank": cfg.MIMO_RANK, "d_state": cfg.D_STATE,
                  "headdim": cfg.HEADDIM, "d_model": cfg.D_MODEL, "n_layers": cfg.N_LAYERS,
                  "dtype": cfg.DTYPE},
        "train_seg_per_trial_cap": cfg.TRAIN_SEG_PER_TRIAL_CAP,
        "results": {
            "full_allbands": {"mean": full["mean"], "std": full["std"], "per_fold": full["per_fold"]},
            "drop_gamma_theta_alpha_beta": {"mean": dropg["mean"], "std": dropg["std"], "per_fold": dropg["per_fold"]},
            "drop_gamma_minus_full": delta,
        },
        "lr_reference": {"full_allbands": 0.539, "drop_gamma": 0.510},
    }
    from collections import Counter
    report["frontend"]["bins_per_band"] = dict(Counter(cfg.KEPT_BIN_BAND))

    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(cfg.RESULTS_DIR / "seed_mamba_stage1_report.json", "w") as f:
        json.dump(report, f, indent=2)
    for p in (full_json, dropg_json):
        shutil.copy(p, run_dir / Path(p).name)
    shutil.copy(cfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")

    print("=== SEED Mamba Stage 1 (V1_full) ===")
    print(f"(a) full all-bands  subject bal_acc = {full['mean']:.3f} +- {full['std']:.3f}   (LR ref 0.539)")
    print(f"(b) drop-gamma      subject bal_acc = {dropg['mean']:.3f} +- {dropg['std']:.3f}   (LR ref 0.510)")
    print(f"    full - drop_gamma = {delta:+.3f}")
    print(f"archived {run_dir}")
    return report


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
