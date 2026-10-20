"""
SEED Task 2.5 helper: recompute per-segment, per-channel log-power for two
gamma sub-bands (30-45 Hz and 45-75 Hz) -- needed by EMG probe 1 (frequency-
ceiling scan) and NOT present in the Task 2 feature cache (which stopped at
gamma=30-45).

Reuses the EXACT Task 2 preprocessing + segmentation (imported unchanged from
seed_preprocess_features) so segment order matches the main cache. Caches to
features_cache/<sid>_gamma.npz with its own seg_label/seg_trial_id copy so
the probe is self-contained.

Note on the 50 Hz notch: the mains notch sits INSIDE the 45-75 Hz sub-band and
removes a ~1.7 Hz slice around 50 Hz. That is line noise, not EMG, so removing
it makes the high-gamma EMG test if anything conservative (less non-neural
power in the high band), not inflated.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import seed2_config as cfg
from seed_preprocess_features import (
    load_session_trials, avg_reference_excluding_bad, segment_trial,
)

sys.path.insert(0, str(Path("/root/newcode/task2")))
from preprocess import notch_filter, detect_bad_channels  # noqa: E402

_FREQS = np.fft.rfftfreq(cfg.WIN_LEN, d=1.0 / cfg.FS)
_SUBBANDS = {"gamma_lo_30_45": (30.0, 45.0), "gamma_hi_45_75": (45.0, 75.0)}
_MASKS = {k: (_FREQS >= lo) & (_FREQS < hi) for k, (lo, hi) in _SUBBANDS.items()}


def subband_logpow(segs):
    X = np.fft.rfft(segs, axis=-1)
    psd = np.abs(X) ** 2
    out = {}
    for k, mask in _MASKS.items():
        out[k] = np.log(psd[:, :, mask].sum(axis=-1) + 1e-12)  # (n_seg, 62)
    return out


def process_subject(sid, session_files):
    lo, hi, labels, tids = [], [], [], []
    for sess_idx, fname in enumerate(session_files):
        trials = load_session_trials(cfg.PREPROC_DIR / fname)
        concat = np.concatenate(trials, axis=1)
        bad = detect_bad_channels(concat, cfg.BAD_CH_LOGVAR_MAD_THRESH, cfg.BAD_CH_MIN_MEAN_ABS_CORR)
        for t_idx, trial in enumerate(trials):
            notched = notch_filter(trial, cfg.FS, cfg.NOTCH_FREQ, cfg.NOTCH_Q)
            reref = avg_reference_excluding_bad(notched, bad)
            segs = segment_trial(reref)
            if segs.shape[0] == 0:
                continue
            sb = subband_logpow(segs)
            lo.append(sb["gamma_lo_30_45"]); hi.append(sb["gamma_hi_45_75"])
            n = segs.shape[0]
            labels.append(np.full(n, cfg.TRIAL_LABELS[t_idx], dtype=np.int32))
            tids.append(np.full(n, sess_idx * cfg.scfg.N_TRIALS_PER_SESSION + t_idx, dtype=np.int32))
    np.savez_compressed(
        cfg.FEATURES_CACHE / f"{sid}_gamma.npz",
        gamma_lo_30_45=np.concatenate(lo), gamma_hi_45_75=np.concatenate(hi),
        seg_label=np.concatenate(labels), seg_trial_id=np.concatenate(tids),
    )


def main():
    manifest = pd.read_csv(cfg.MANIFEST_CSV)
    t0 = time.time()
    for _, row in manifest.iterrows():
        sid = int(row["subject_id"])
        process_subject(sid, [row["session_1_file"], row["session_2_file"], row["session_3_file"]])
        print(f"subj {sid:2d}: gamma sub-bands cached")
    print(f"gamma sub-band extraction done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
