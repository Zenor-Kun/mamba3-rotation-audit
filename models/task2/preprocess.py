"""
Task 2 Stage A — per-subject preprocessing (main_B only).

Order (locked, do not reorder):
  1. zero-phase bandpass 0.5-45 Hz on the FULL continuous [128, T] recording
     (before windowing -> no filter transients injected at window edges)
  2. re-reference (avg-reference placeholder for REST, see config.REREF_METHOD)
  3. data-driven bad-channel detection (flag only, no interpolation in main_B)
  4. cache the continuous preprocessed signal + bad_channels list

Windowing happens in windows.py (separate step, still fold-independent).
"""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

import config as cfg
from io_utils import load_raw_mat


def bandpass_filter(data, fs, hpf, lpf, order):
    sos = butter(order, [hpf, lpf], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(data, fs, freq, q):
    b, a = iirnotch(freq, q, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def avg_reference(data):
    return data - data.mean(axis=0, keepdims=True)


def detect_bad_channels(data, logvar_mad_thresh, min_mean_abs_corr):
    """data: (128, T) filtered+reref. Returns sorted list of bad channel indices."""
    n_ch = data.shape[0]
    var = data.var(axis=1)
    logvar = np.log(var + 1e-12)
    med = np.median(logvar)
    mad = np.median(np.abs(logvar - med)) * 1.4826 + 1e-12
    z = (logvar - med) / mad
    bad_var = set(np.where(np.abs(z) > logvar_mad_thresh)[0].tolist())

    corr = np.corrcoef(data)
    np.fill_diagonal(corr, np.nan)
    mean_abs_corr = np.nanmean(np.abs(corr), axis=1)
    bad_corr = set(np.where(mean_abs_corr < min_mean_abs_corr)[0].tolist())

    return sorted(bad_var | bad_corr)


def process_subject(sid, data_path):
    raw, fs = load_raw_mat(data_path)
    assert abs(fs - cfg.FS) < 1e-6, f"{sid}: fs={fs} != locked {cfg.FS}"
    data = raw[: cfg.N_DATA_CH, :]           # drop Cz row (129th), keep for record only
    cz = raw[cfg.CZ_ROW_IDX, :]              # noqa: F841 (kept for provenance, unused by avg-ref)

    filtered = bandpass_filter(data, fs, cfg.HPF, cfg.LPF, cfg.FILTER_ORDER)
    filtered = notch_filter(filtered, fs, cfg.NOTCH_FREQ, cfg.NOTCH_Q)
    reref = avg_reference(filtered)
    bad_channels = detect_bad_channels(
        reref, cfg.BAD_CH_LOGVAR_MAD_THRESH, cfg.BAD_CH_MIN_MEAN_ABS_CORR
    )

    out = cfg.PREPROC_CACHE / f"{sid}.npz"
    np.savez_compressed(
        out,
        data=reref.astype(np.float32),
        fs=fs,
        bad_channels=np.array(bad_channels, dtype=np.int64),
    )
    return reref.shape[1], bad_channels


def main():
    df = pd.read_csv(cfg.MANIFEST_IN, dtype={"subject_id": str})
    bad_ch_col = []
    t_lens = []
    print(f"Task 2 Stage A: preprocessing {len(df)} subjects (main_B only)")
    print(f"reref method = {cfg.REREF_METHOD}  (see config.py deviation note)")
    for _, row in df.iterrows():
        sid = row["subject_id"]
        T, bad_channels = process_subject(sid, row["data_path"])
        t_lens.append(T)
        bad_ch_col.append(json.dumps(bad_channels))
        flag = f" bad={bad_channels}" if bad_channels else ""
        print(f"  {sid}: T={T}{flag}")

    df["bad_channels"] = bad_ch_col
    df.to_csv(cfg.MANIFEST_OUT, index=False)
    print(f"\nwrote {cfg.MANIFEST_OUT}")
    print(f"T range: min={min(t_lens)} median={int(np.median(t_lens))} max={max(t_lens)}")
    n_with_bad = sum(1 for c in bad_ch_col if c != "[]")
    print(f"subjects with >=1 flagged bad channel: {n_with_bad} / {len(df)}")


if __name__ == "__main__":
    main()
