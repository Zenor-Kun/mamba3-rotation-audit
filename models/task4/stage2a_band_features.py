"""
Task 4 Stage 2A, Protection 3: ERP-band autopsy for any (channel_set,layer)
signal that exceeded balanced_acc > 0.55 in Protections 1+2. Three views of
the SAME kept-trial set (mad_k3-cleaned, same trials as the broadband
reference already computed):

  (a) time-domain broadband ERP -- already computed (Stage 1 main_B
      0.5-45Hz passband), this is the reference/baseline already in the
      Protections 1+2 report.
  (b) band-limited ERP (phase-locked): band-pass EACH KEPT TRIAL into the
      band, THEN average trials. A genuine phase-locked (evoked)
      oscillatory response survives trial-averaging in that band; a
      non-phase-locked (induced) or noise-only signal in that band
      averages toward zero.
  (c) band power (non-phase-locked): square each trial's band-filtered
      signal (instantaneous power), THEN average across trials. Captures
      power regardless of phase-locking -- classic band-power/EMG-style
      measure.

CAVEAT (recorded explicitly, not silently): the epoch is only 200 samples
(800ms) long. A full cycle at the delta band's low end (0.5Hz) is 2000ms --
more than 2x the epoch length. Delta-band features from this window
capture the slow trend/edge behavior, not resolved oscillatory content;
theta (4-8Hz, cycle 125-250ms) and alpha (8-13Hz, cycle 77-125ms) are
comfortably resolved within 800ms; gamma (30-45Hz) is well resolved.
"""
import numpy as np
from scipy.signal import butter, filtfilt

import erp_config as ecfg
import stage2a_config as cfg
from stage2a_data import load_subject_raw, flag_array
from stage2a_features import erp_roi_window_features

FILTER_ORDER = 4


def bandpass_trials(trials, lo, hi, fs):
    """trials: (n_trials, n_ch, n_samples). Zero-phase butterworth bandpass
    along the last axis, vectorized over the other two."""
    nyq = fs / 2.0
    b, a = butter(FILTER_ORDER, [lo / nyq, hi / nyq], btype="band")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if trials.shape[-1] <= padlen:
        padlen = trials.shape[-1] - 1
    return filtfilt(b, a, trials, axis=-1, padlen=padlen)


def subject_band_erp_power(sid, condition, flag_set="mad_k3"):
    """Returns {band_name: (band_erp_(128,200)_uv, band_power_(128,200)_uv2)}
    for one subject/condition, cached to disk."""
    cache_path = cfg.BAND_CACHE / f"{sid}_{condition}_{flag_set}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return {b: (d[f"erp_{b}"], d[f"power_{b}"]) for b in cfg.BANDS}

    raw = load_subject_raw(sid)
    cond_mask = raw["conditions"] == condition
    flags = flag_array(raw, flag_set)
    keep_mask = cond_mask & (~flags)
    trials_uv = raw["bc_epochs"][keep_mask].astype(np.float64) * 1e6  # (n_kept, 128, 200)

    out = {}
    save_kwargs = {}
    for band, (lo, hi) in cfg.BANDS.items():
        filtered = bandpass_trials(trials_uv, lo, hi, cfg.FS)  # (n_kept, 128, 200)
        band_erp = filtered.mean(axis=0).astype(np.float32)          # phase-locked
        band_power = (filtered ** 2).mean(axis=0).astype(np.float32)  # non-phase-locked
        out[band] = (band_erp, band_power)
        save_kwargs[f"erp_{band}"] = band_erp
        save_kwargs[f"power_{band}"] = band_power

    cfg.BAND_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **save_kwargs)
    return out


def subject_band_features(sid, condition, roi_dict, flag_set="mad_k3"):
    """Returns {band_name: {"erp": feat_vec, "power": feat_vec}}."""
    band_data = subject_band_erp_power(sid, condition, flag_set)
    out = {}
    for band, (erp, power) in band_data.items():
        out[band] = {
            "erp": erp_roi_window_features(erp, roi_dict),
            "power": erp_roi_window_features(power, roi_dict),
        }
    return out
