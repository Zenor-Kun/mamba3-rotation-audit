"""
Cache raw preprocessed 4 s segments per subject for the Mamba front-end.
Reuses Task 2 preprocessing UNCHANGED (50 Hz notch -> average reference
excluding bad channels; NO CSD -- Stage 1 uses the same signal Task 2's LR
scored 0.539 on, so the Mamba/LR comparison is apples-to-apples). Segments are
the same 800/400 (50% overlap) windows as Task 2.

Stores per subject: raw_segments (n_seg, 62, 800) float32 (Volts),
seg_label (n_seg,), seg_trial_id (n_seg,). STFT is computed later on GPU in the
model, not cached (would be ~60 GB).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import seed_mamba_config as cfg

sys.path.insert(0, str(Path("/root/newcode/seed/task2")))
from seed_preprocess_features import (  # noqa: E402
    load_session_trials, avg_reference_excluding_bad, segment_trial,
)
sys.path.insert(0, str(Path("/root/newcode/task2")))
from preprocess import notch_filter, detect_bad_channels  # noqa: E402


def process_subject(sid, session_files):
    segs_all, lab, tid = [], [], []
    for sess_idx, fname in enumerate(session_files):
        trials = load_session_trials(cfg.s2cfg.PREPROC_DIR / fname)
        concat = np.concatenate(trials, axis=1)
        bad = detect_bad_channels(concat, cfg.s2cfg.BAD_CH_LOGVAR_MAD_THRESH, cfg.s2cfg.BAD_CH_MIN_MEAN_ABS_CORR)
        for t_idx, trial in enumerate(trials):
            notched = notch_filter(trial, cfg.FS, cfg.s2cfg.NOTCH_FREQ, cfg.s2cfg.NOTCH_Q)
            reref = avg_reference_excluding_bad(notched, bad)
            segs = segment_trial(reref)          # (n_seg, 62, 800)
            if segs.shape[0] == 0:
                continue
            segs_all.append(segs.astype(np.float32))
            n = segs.shape[0]
            lab.append(np.full(n, cfg.TRIAL_LABELS[t_idx], dtype=np.int32))
            tid.append(np.full(n, sess_idx * cfg.scfg.N_TRIALS_PER_SESSION + t_idx, dtype=np.int32))
    np.savez(
        cfg.SEGMENTS_CACHE / f"{sid}.npz",
        raw_segments=np.concatenate(segs_all), seg_label=np.concatenate(lab),
        seg_trial_id=np.concatenate(tid),
    )
    return np.concatenate(lab).shape[0]


def main():
    manifest = pd.read_csv(cfg.MANIFEST_CSV)
    t0 = time.time()
    for _, row in manifest.iterrows():
        sid = int(row["subject_id"])
        n = process_subject(sid, [row["session_1_file"], row["session_2_file"], row["session_3_file"]])
        print(f"subj {sid:2d}: {n} segments cached ({time.time()-t0:.0f}s)")
    print(f"segment cache done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
