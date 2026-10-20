"""
SEED Task 2.6 helper: recompute Task 2 features on the CSD (current source
density / surface-Laplacian) transform of the signal.

CRITICAL ORDER (per spec, non-negotiable): CSD is a whole-head spatial
operator. It is computed ONCE on all 62 channels; channel-group analyses
later select channels from THIS 62-channel CSD result by mask. CSD is NEVER
recomputed on a channel subset (that would change the spatial operator and
make groups non-comparable).

Per trial: task2 preprocessing (50 Hz notch -> average reference excluding
bad channels) -> wrap in an MNE RawArray with the SEED montage -> ONE
compute_current_source_density over all 62 channels -> segment (same 800/400)
-> per-channel band log-power (theta/alpha/beta/gamma) + log-variance + Hjorth
mobility + a 55-75 Hz high-EMG-proxy band log-power.

Montage: 60 of 62 SEED names map directly to MNE 'standard_1005'. CB1/CB2
(the Neuroscan Chinese-cap occipito-inferior electrodes, absent from standard
10-05) are given the positions of OI1/OI2 -- the occipito-inferior 10-05 sites
sitting directly below O1/O2, matching where CB1/CB2 physically are. Documented
approximation; only affects those two local Laplacians.

Bad channels are kept in (flag-only, as in Task 2) so the CSD montage stays
full 62; bad-channel counts are low (median 4). Not dropped, for parallelism
with Task 2.

'high_emg_proxy' name is deliberate: SEED's ceiling is 75 Hz, so 55-75 Hz can
still contain a little neural high-gamma -- it is an EMG PROXY, not a pure EMG
band.
"""
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import seed2_config as cfg
from seed_preprocess_features import (
    load_session_trials, avg_reference_excluding_bad, segment_trial,
)

sys.path.insert(0, str(Path("/root/newcode/task2")))
from preprocess import notch_filter, detect_bad_channels  # noqa: E402

mne.set_log_level("ERROR")

CH_NAMES = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
            'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
            'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7',
            'P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8',
            'CB1','O1','OZ','O2','CB2']
CB_SUBSTITUTE = {"CB1": "OI1", "CB2": "OI2"}

FEATURES_CACHE_CSD = cfg.FEATURES_CACHE / "csd"
FEATURES_CACHE_CSD.mkdir(parents=True, exist_ok=True)

_FREQS = np.fft.rfftfreq(cfg.WIN_LEN, d=1.0 / cfg.FS)
_BAND_MASKS = {b: (_FREQS >= lo) & (_FREQS < hi) for b, (lo, hi) in cfg.BANDS.items()}
_HIGH_EMG_MASK = (_FREQS >= 55.0) & (_FREQS < 75.0)


def build_montage():
    std = mne.channels.make_standard_montage("standard_1005")
    sp = std.get_positions()
    ch_pos_std = sp["ch_pos"]
    up = {n.upper(): n for n in std.ch_names}
    seed_pos = {}
    for name in CH_NAMES:
        src = CB_SUBSTITUTE.get(name, name)
        seed_pos[name] = ch_pos_std[up[src.upper()]]
    return mne.channels.make_dig_montage(
        ch_pos=seed_pos, nasion=sp["nasion"], lpa=sp["lpa"], rpa=sp["rpa"], coord_frame="head")


_INFO = None
_MONTAGE = None


def get_info():
    global _INFO, _MONTAGE
    if _INFO is None:
        _MONTAGE = build_montage()
        _INFO = mne.create_info(CH_NAMES, sfreq=cfg.FS, ch_types="eeg")
        _INFO.set_montage(_MONTAGE)
    return _INFO


def csd_transform_trial(reref):
    """reref (62, T) -> CSD-transformed (62, T)."""
    raw = mne.io.RawArray(reref, get_info(), verbose="ERROR")
    csd = mne.preprocessing.compute_current_source_density(raw, verbose="ERROR")
    return csd.get_data()


def segment_features_csd(segs):
    X = np.fft.rfft(segs, axis=-1)
    psd = np.abs(X) ** 2
    n_seg, n_ch, _ = segs.shape
    band_logpow = np.empty((n_seg, len(cfg.BAND_ORDER), n_ch))
    for bi, b in enumerate(cfg.BAND_ORDER):
        band_logpow[:, bi, :] = np.log(psd[:, :, _BAND_MASKS[b]].sum(-1) + 1e-12)
    high_emg = np.log(psd[:, :, _HIGH_EMG_MASK].sum(-1) + 1e-12)
    var = segs.var(-1); logvar = np.log(var + 1e-12)
    dvar = np.diff(segs, axis=-1).var(-1)
    mobility = np.sqrt((dvar + 1e-12) / (var + 1e-12))
    return band_logpow, logvar, mobility, high_emg


def process_subject(sid, session_files):
    B, LV, MO, HE, LAB, TID = [], [], [], [], [], []
    for sess_idx, fname in enumerate(session_files):
        trials = load_session_trials(cfg.PREPROC_DIR / fname)
        concat = np.concatenate(trials, axis=1)
        bad = detect_bad_channels(concat, cfg.BAD_CH_LOGVAR_MAD_THRESH, cfg.BAD_CH_MIN_MEAN_ABS_CORR)
        for t_idx, trial in enumerate(trials):
            notched = notch_filter(trial, cfg.FS, cfg.NOTCH_FREQ, cfg.NOTCH_Q)
            reref = avg_reference_excluding_bad(notched, bad)
            csd = csd_transform_trial(reref)
            segs = segment_trial(csd)
            if segs.shape[0] == 0:
                continue
            b, lv, mo, he = segment_features_csd(segs)
            B.append(b); LV.append(lv); MO.append(mo); HE.append(he)
            n = segs.shape[0]
            LAB.append(np.full(n, cfg.TRIAL_LABELS[t_idx], dtype=np.int32))
            TID.append(np.full(n, sess_idx * cfg.scfg.N_TRIALS_PER_SESSION + t_idx, dtype=np.int32))
    np.savez_compressed(
        FEATURES_CACHE_CSD / f"{sid}.npz",
        band_logpow=np.concatenate(B), logvar=np.concatenate(LV),
        mobility=np.concatenate(MO), high_emg_logpow=np.concatenate(HE),
        seg_label=np.concatenate(LAB), seg_trial_id=np.concatenate(TID),
    )


def main():
    manifest = pd.read_csv(cfg.MANIFEST_CSV)
    t0 = time.time()
    for _, row in manifest.iterrows():
        sid = int(row["subject_id"])
        process_subject(sid, [row["session_1_file"], row["session_2_file"], row["session_3_file"]])
        print(f"subj {sid:2d}: CSD features cached ({time.time()-t0:.0f}s elapsed)")
    print(f"CSD feature extraction done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
