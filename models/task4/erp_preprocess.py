"""
Task 4 ERP Stage 1: main_B-lineage preprocessing on the continuous ERP
recording, reusing task2's filter functions UNCHANGED (bandpass_filter,
notch_filter, detect_bad_channels -- imported directly, not reimplemented).

Deliberate deviations from task2's resting-state order, both required by the
user's spec for this step and documented in STATUS.md:
  1. average reference EXCLUDES bad channels from the averaging step (task2's
     resting-state avg_reference does not -- that pipeline is frozen, not
     touched; this is a new function for ERP only).
  2. bad-channel detection therefore has to run BEFORE referencing here
     (task2 runs it after), since excluding bad channels from the reference
     average requires already knowing which channels are bad.
Bad channels are still flag-only, never interpolated (main_B convention).
"""
import sys
from pathlib import Path

import numpy as np

import erp_config as ecfg

sys.path.insert(0, str(ecfg.TASK2_DIR))
from preprocess import bandpass_filter, notch_filter, detect_bad_channels  # noqa: E402


def avg_reference_excluding_bad(data, bad_channels):
    """data: (128, T). Average computed over good channels only, subtracted
    from ALL 128 channels (bad channels are re-referenced too, just not
    counted in the average -- they stay in the data, flagged not dropped)."""
    good_mask = np.ones(data.shape[0], dtype=bool)
    good_mask[list(bad_channels)] = False
    assert good_mask.sum() > 0, "all channels flagged bad -- refusing to reference against nothing"
    ref = data[good_mask].mean(axis=0, keepdims=True)
    return data - ref


def preprocess_continuous(data_128ch):
    """data_128ch: (128, T) raw EGI samples (E1..E128, Cz/E129 already dropped
    by the caller), in MNE's native units (Volts). Returns (referenced_data,
    bad_channels_list)."""
    filtered = bandpass_filter(data_128ch, ecfg.FS, ecfg.HPF, ecfg.LPF, ecfg.FILTER_ORDER)
    filtered = notch_filter(filtered, ecfg.FS, ecfg.NOTCH_FREQ, ecfg.NOTCH_Q)
    bad_channels = detect_bad_channels(
        filtered, ecfg.BAD_CH_LOGVAR_MAD_THRESH, ecfg.BAD_CH_MIN_MEAN_ABS_CORR
    )
    referenced = avg_reference_excluding_bad(filtered, bad_channels)
    return referenced, bad_channels
