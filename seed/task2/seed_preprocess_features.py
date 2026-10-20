"""
SEED Task 2 stage A (fold-independent): preprocess -> segment -> per-segment
features, cached once per subject. No fold logic here (z-scoring is fold-
dependent and happens later in the LOSO LR step).

Per subject (all 3 sessions):
  1. Load each session .mat -> 15 trial arrays (62, samples).
  2. Bad-channel detection per SESSION on the trial-concatenated signal
     (more robust than per-trial; task2's UNCHANGED detect_bad_channels,
     same thresholds). Flag only -- never dropped/interpolated.
  3. Per trial: 50 Hz notch (task2's UNCHANGED notch_filter) -> average
     reference EXCLUDING that session's bad channels (all 62 channels stay
     in the data, bad ones just excluded from the reference average).
  4. Segment each preprocessed trial: 800-sample (4 s) windows, 400 hop
     (50% overlap), tail shorter than one window discarded.
  5. Per segment, per channel features:
       - band log-power in theta/alpha/beta/gamma  -> (n_band=4, n_ch=62)
       - log-variance (time-domain)                -> (n_ch=62)
       - Hjorth mobility  = sqrt(var(diff)/var(x)) -> (n_ch=62)
     plus a scalar gamma power fraction per segment (EMG-proxy autopsy).
  6. Cache to features_cache/<sid>.npz with segment->trial->label->fold
     bookkeeping so nothing leaks across the subject boundary later.

Run with base conda env (scipy/numpy/sklearn); never the mamba3 env.
"""
import sys
import time
from pathlib import Path

import json

import numpy as np
import pandas as pd
import scipy.io as sio

import seed2_config as cfg

sys.path.insert(0, str(Path("/root/newcode/task2")))
from preprocess import notch_filter, detect_bad_channels  # task2 UNCHANGED  # noqa: E402


def avg_reference_excluding_bad(data, bad_channels):
    """data (62, T). Reference = mean over good channels only, subtracted
    from ALL channels (bad channels stay in the data, flagged not dropped)."""
    good = np.ones(data.shape[0], dtype=bool)
    good[list(bad_channels)] = False
    assert good.sum() > 0
    return data - data[good].mean(axis=0, keepdims=True)


def load_session_trials(mat_path):
    m = sio.loadmat(mat_path, verify_compressed_data_integrity=False)
    trial_keys = [k for k in m.keys() if not k.startswith("__") and "_eeg" in k]
    trial_keys = sorted(trial_keys, key=lambda k: int(k.split("eeg")[1]))
    assert len(trial_keys) == cfg.scfg.N_TRIALS_PER_SESSION
    return [m[k].astype(np.float64) for k in trial_keys]  # each (62, samples)


def segment_trial(data):
    """data (62, T) -> (n_seg, 62, WIN_LEN); tail < WIN_LEN dropped."""
    T = data.shape[1]
    n_seg = (T - cfg.WIN_LEN) // cfg.STEP + 1 if T >= cfg.WIN_LEN else 0
    segs = np.empty((n_seg, data.shape[0], cfg.WIN_LEN), dtype=np.float64)
    for i in range(n_seg):
        s = i * cfg.STEP
        segs[i] = data[:, s:s + cfg.WIN_LEN]
    return segs


# precompute band bin masks over the rFFT frequency axis
_FREQS = np.fft.rfftfreq(cfg.WIN_LEN, d=1.0 / cfg.FS)
_BAND_MASKS = {b: (_FREQS >= lo) & (_FREQS < hi) for b, (lo, hi) in cfg.BANDS.items()}


def segment_features(segs):
    """segs (n_seg, 62, 800) -> dict of feature blocks.
    band_logpow (n_seg, 4, 62), logvar (n_seg, 62), mobility (n_seg, 62),
    gamma_frac (n_seg,)."""
    n_seg, n_ch, _ = segs.shape
    # PSD via rFFT (power = |X|^2 / (fs*N), one-sided) -- constants cancel in log/ratio
    X = np.fft.rfft(segs, axis=-1)
    psd = (np.abs(X) ** 2)  # (n_seg, 62, n_freq)

    band_pow = np.empty((n_seg, len(cfg.BAND_ORDER), n_ch), dtype=np.float64)
    for bi, b in enumerate(cfg.BAND_ORDER):
        band_pow[:, bi, :] = psd[:, :, _BAND_MASKS[b]].sum(axis=-1)
    band_logpow = np.log(band_pow + 1e-12)

    var = segs.var(axis=-1)                                  # (n_seg, 62)
    logvar = np.log(var + 1e-12)
    dvar = np.diff(segs, axis=-1).var(axis=-1)               # (n_seg, 62)
    mobility = np.sqrt((dvar + 1e-12) / (var + 1e-12))

    total_band = band_pow.sum(axis=1)                        # (n_seg, 62)
    gamma_idx = cfg.BAND_ORDER.index("gamma")
    gamma_frac = (band_pow[:, gamma_idx, :] / (total_band + 1e-12)).mean(axis=1)  # (n_seg,)

    return {"band_logpow": band_logpow, "logvar": logvar,
            "mobility": mobility, "gamma_frac": gamma_frac}


def process_subject(sid, session_files):
    all_band, all_logvar, all_mob, all_gf = [], [], [], []
    seg_trial_id, seg_label, seg_session = [], [], []
    bad_by_session = {}
    tail_drops = 0
    trial_seg_counts = []  # (session_idx, trial_idx_in_session, label, n_seg)

    for sess_idx, fname in enumerate(session_files):
        trials = load_session_trials(cfg.PREPROC_DIR / fname)
        concat = np.concatenate(trials, axis=1)  # (62, sum T) for robust bad-ch detection
        bad = detect_bad_channels(concat, cfg.BAD_CH_LOGVAR_MAD_THRESH, cfg.BAD_CH_MIN_MEAN_ABS_CORR)
        bad_by_session[sess_idx] = bad

        for t_idx, trial in enumerate(trials):
            notched = notch_filter(trial, cfg.FS, cfg.NOTCH_FREQ, cfg.NOTCH_Q)
            reref = avg_reference_excluding_bad(notched, bad)
            segs = segment_trial(reref)
            label = cfg.TRIAL_LABELS[t_idx]
            n_seg = segs.shape[0]
            trial_seg_counts.append((sess_idx, t_idx, label, n_seg))
            T = trial.shape[1]
            tail_drops += (T - cfg.WIN_LEN) % cfg.STEP if T >= cfg.WIN_LEN else T
            if n_seg == 0:
                continue
            f = segment_features(segs)
            all_band.append(f["band_logpow"]); all_logvar.append(f["logvar"])
            all_mob.append(f["mobility"]); all_gf.append(f["gamma_frac"])
            # global trial id unique within subject: sess*15 + t_idx
            gtid = sess_idx * cfg.scfg.N_TRIALS_PER_SESSION + t_idx
            seg_trial_id.append(np.full(n_seg, gtid, dtype=np.int32))
            seg_label.append(np.full(n_seg, label, dtype=np.int32))
            seg_session.append(np.full(n_seg, sess_idx, dtype=np.int32))

    out = {
        "band_logpow": np.concatenate(all_band),      # (N, 4, 62)
        "logvar": np.concatenate(all_logvar),         # (N, 62)
        "mobility": np.concatenate(all_mob),          # (N, 62)
        "gamma_frac": np.concatenate(all_gf),         # (N,)
        "seg_trial_id": np.concatenate(seg_trial_id),
        "seg_label": np.concatenate(seg_label),
        "seg_session": np.concatenate(seg_session),
        "n_bad_union": len(set().union(*[set(v) for v in bad_by_session.values()])),
        "bad_by_session": {int(k): [int(x) for x in v] for k, v in bad_by_session.items()},
        "tail_drops_samples": int(tail_drops),
        "trial_seg_counts": trial_seg_counts,
    }
    return out


def main():
    manifest = pd.read_csv(cfg.MANIFEST_CSV)
    t0 = time.time()
    summary = []
    for _, row in manifest.iterrows():
        sid = int(row["subject_id"])
        sess_files = [row["session_1_file"], row["session_2_file"], row["session_3_file"]]
        d = process_subject(sid, sess_files)
        np.savez_compressed(
            cfg.FEATURES_CACHE / f"{sid}.npz",
            band_logpow=d["band_logpow"], logvar=d["logvar"], mobility=d["mobility"],
            gamma_frac=d["gamma_frac"], seg_trial_id=d["seg_trial_id"],
            seg_label=d["seg_label"], seg_session=d["seg_session"],
        )
        n_seg = d["seg_label"].shape[0]
        per_class = {cfg.IDX_TO_NAME[cfg.LABEL_TO_IDX[c]]: int((d["seg_label"] == c).sum())
                     for c in (-1, 0, 1)}
        summary.append({"subject_id": sid, "n_segments": n_seg,
                        "n_bad_union": d["n_bad_union"], "bad_by_session": d["bad_by_session"],
                        "tail_drops_samples": d["tail_drops_samples"],
                        "segs_per_class": per_class})
        print(f"subj {sid:2d}: n_seg={n_seg} bad(union)={d['n_bad_union']} "
              f"segs/class={per_class}")
    with open(cfg.RESULTS_DIR / "seed_task2_stageA_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"feature extraction done in {time.time()-t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
