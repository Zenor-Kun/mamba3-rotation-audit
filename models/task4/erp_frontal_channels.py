"""
Identifies the most-anterior (frontal/eye-proximal) channels among E1..E128
using the standard EGI GSN-HydroCel-129 montage positions -- used as an
eye-movement-artifact proxy since this dataset has no dedicated EOG channel.
Montage-based, not tuned to this data (no circularity with the artifact
flagging thresholds, which ARE data-driven from the trial distributions).
"""
import mne
import numpy as np

import erp_config as ecfg


def frontal_channel_indices(n_frontal=ecfg.N_FRONTAL_CHANNELS):
    mont = mne.channels.make_standard_montage("GSN-HydroCel-129")
    ch_pos = mont.get_positions()["ch_pos"]
    names = [f"E{i}" for i in range(1, ecfg.N_DATA_CH + 1)]  # E1..E128, same order as the modeling channels
    y = np.array([ch_pos[n][1] for n in names])  # anterior-posterior axis
    order = np.argsort(-y)
    idx = order[:n_frontal]
    return idx, [names[i] for i in idx]


if __name__ == "__main__":
    idx, names = frontal_channel_indices()
    print(f"frontal/eye-proximal channel subset (n={len(idx)}): {names}")
