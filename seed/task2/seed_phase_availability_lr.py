"""
Phase-availability audit (layer-0): in what FORM does SEED's subject-independent
discriminative information exist? LR-only, NO Mamba. See PHASE_AVAIL_PREREG.md
(decision lines frozen BEFORE results).

Same complex STFT front-end as the Mamba variants (n_fft=128, hop=6, hann, center=False,
26 in-band bins). Three feature blocks fed to the identical LOSO LR pipeline as
seed_lr_loso.py (train-fold z-score, multinomial LR, segment->trial->subject bal_acc,
chance 0.333, subject = statistical unit):
  MAG   = log mean-power per (chan,bin)            [phase discarded]     1612 dims
  PHASE = mean unit-phasor [Re,Im] per (chan,bin)  [magnitude removed]   3224 dims
  FULL  = concat(MAG, PHASE)                        [nested superset]     4836 dims

Reads splits_seed_loso.json READ-ONLY. Archives to runs/<ts>_seed_phase_availability_lr/.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "task3"))
import seed_mamba_config as mc          # STFT front-end constants (shared with Mamba)
import seed2_config as cfg              # splits, labels, LR hyperparams (shared with LR baseline)

CHANCE = 1.0 / 3.0
DELTA = 0.02          # practical margin (pre-registered)
TOST_MARGIN = 0.03    # equivalence margin, consistent with main paper (pre-registered)
FEATCACHE = Path(__file__).resolve().parent / "phase_avail_featcache"
FEATCACHE.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KEPT = torch.tensor(mc.KEPT_BIN_IDX, dtype=torch.long)


def compute_blocks_for_subject(sid):
    """Compute MAG (N,1612) and PHASE (N,3224) blocks for one subject; cache to disk."""
    cache = FEATCACHE / f"{sid}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["mag"], d["phase"], d["label"], d["trial"]
    seg = np.load(mc.SEGMENTS_CACHE / f"{sid}.npz")
    raw = seg["raw_segments"]            # (N, 62, 800) float32
    label, trial = seg["seg_label"], seg["seg_trial_id"]
    N, C, T = raw.shape
    window = torch.hann_window(mc.STFT_N_FFT, device=DEV)
    kept = KEPT.to(DEV)
    mag_out = np.empty((N, C * mc.N_FREQ_BINS), dtype=np.float32)
    ph_out = np.empty((N, C * mc.N_FREQ_BINS * 2), dtype=np.float32)
    BS = 256                             # segments per GPU batch (256*62 rows)
    with torch.no_grad():
        for i in range(0, N, BS):
            xb = torch.from_numpy(raw[i:i + BS]).to(DEV)          # (b,62,800)
            b = xb.shape[0]
            xr = xb.reshape(b * C, T)
            spec = torch.stft(xr, n_fft=mc.STFT_N_FFT, hop_length=mc.STFT_HOP,
                              win_length=mc.STFT_N_FFT, window=window,
                              center=False, return_complex=True)   # (b*62, 65, 113)
            spec = spec.index_select(1, kept)                      # (b*62, 26, 113)
            power = (spec.real ** 2 + spec.imag ** 2)              # |Z|^2
            mag = torch.log(power.mean(dim=2) + 1e-8)              # (b*62, 26)  frame-mean power, log
            u = spec / (spec.abs() + 1e-8)                         # unit phasor (magnitude removed)
            R = u.mean(dim=2)                                      # (b*62, 26) complex resultant
            ph = torch.stack([R.real, R.imag], dim=-1)            # (b*62, 26, 2)
            mag_out[i:i + b] = mag.reshape(b, C * mc.N_FREQ_BINS).float().cpu().numpy()
            ph_out[i:i + b] = ph.reshape(b, C * mc.N_FREQ_BINS * 2).float().cpu().numpy()
    np.savez(cache, mag=mag_out, phase=ph_out, label=label, trial=trial)
    return mag_out, ph_out, label, trial


def block_matrix(store, sid, block):
    mag, ph = store[sid]["mag"], store[sid]["phase"]
    if block == "MAG":
        return mag
    if block == "PHASE":
        return ph
    return np.concatenate([mag, ph], axis=1)   # FULL


def labels_to_idx(y):
    return np.array([cfg.LABEL_TO_IDX[int(v)] for v in y], dtype=np.int64)


def run_block(store, splits, block):
    subj_bacc, seg_bacc = [], []
    for _, fold in splits.items():
        test_sid = int(fold["test"][0])
        train_ids = [int(s) for s in fold["train"]]
        Xtr = np.concatenate([block_matrix(store, s, block) for s in train_ids])
        ytr = labels_to_idx(np.concatenate([store[s]["label"] for s in train_ids]))
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-8
        Xtr = (Xtr - mu) / sd
        clf = LogisticRegression(C=cfg.LR_C, max_iter=cfg.LR_MAX_ITER)
        clf.fit(Xtr, ytr)
        Xte = (block_matrix(store, test_sid, block) - mu) / sd
        yte = labels_to_idx(store[test_sid]["label"])
        proba = clf.predict_proba(Xte)
        seg_bacc.append(balanced_accuracy_score(yte, proba.argmax(1)))
        trial_ids = store[test_sid]["trial"]
        t_true, t_pred = [], []
        for tid in np.unique(trial_ids):
            m = trial_ids == tid
            t_pred.append(proba[m].mean(0).argmax())
            t_true.append(yte[m][0])
        subj_bacc.append(balanced_accuracy_score(t_true, t_pred))
        print(f"  [{block:5s}] fold test S{test_sid:02d}: subj bal_acc {subj_bacc[-1]:.3f}")
    return np.array(subj_bacc), np.array(seg_bacc)


def vs_chance(vals):
    t_p = float(stats.ttest_1samp(vals, CHANCE).pvalue)
    try:
        w_p = float(stats.wilcoxon(vals - CHANCE).pvalue)
    except ValueError:
        w_p = 1.0
    mean = float(vals.mean())
    approx_chance = (t_p >= 0.05 and w_p >= 0.05 and mean <= CHANCE + DELTA)
    above_chance = (t_p < 0.05 and w_p < 0.05 and mean - CHANCE > DELTA)
    return {"mean": mean, "std": float(vals.std()), "t_p": t_p, "wilcoxon_p": w_p,
            "approx_chance": approx_chance, "above_chance": above_chance}


def paired(a, b):
    """a - b across subjects (paired)."""
    d = a - b
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        w_p = 1.0
    md = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    ci95 = (md - 1.96 * se, md + 1.96 * se)
    ci90 = (md - 1.645 * se, md + 1.645 * se)
    tost_equiv = (ci90[0] > -TOST_MARGIN and ci90[1] < TOST_MARGIN)
    return {"mean_diff": md, "t_p": t_p, "wilcoxon_p": w_p,
            "ci95": [ci95[0], ci95[1]], "ci90": [ci90[0], ci90[1]],
            "tost_equivalent_pm03": tost_equiv,
            "equiv_practical": (t_p >= 0.05 and abs(md) <= DELTA),
            "full_gt_mag": (t_p < 0.05 and md > DELTA)}


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_seed_phase_availability_lr"
    run_dir.mkdir(parents=True)
    t0 = time.time()

    splits = json.load(open(cfg.SPLITS_JSON))
    store = {}
    print(f"device={DEV}; computing STFT feature blocks per subject ...")
    for sid in range(1, cfg.scfg.N_SUBJECTS + 1):
        mag, ph, label, trial = compute_blocks_for_subject(sid)
        store[sid] = {"mag": mag, "phase": ph, "label": label, "trial": trial}
        print(f"  S{sid:02d}: N={len(label)} mag{mag.shape} phase{ph.shape}")

    res = {}
    perfold = {}
    for block in ["MAG", "PHASE", "FULL"]:
        sb, gb = run_block(store, splits, block)
        perfold[block] = sb
        res[block] = {"subject_level_balanced_acc": {
            "mean": float(sb.mean()), "std": float(sb.std()),
            "per_fold": [float(x) for x in sb]},
            "segment_level_mean": float(gb.mean()),
            "vs_chance": vs_chance(sb)}
        print(f"[{block:5s}] subject bal_acc = {sb.mean():.3f} +- {sb.std():.3f}  "
              f"(seg {gb.mean():.3f})  approx_chance={res[block]['vs_chance']['approx_chance']} "
              f"above_chance={res[block]['vs_chance']['above_chance']}")

    full_vs_mag = paired(perfold["FULL"], perfold["MAG"])
    phase_gt = res["PHASE"]["vs_chance"]["above_chance"]
    phase_approx_chance = res["PHASE"]["vs_chance"]["approx_chance"]
    mag_approx_full = full_vs_mag["equiv_practical"]

    verdict = {
        "A_magnitude_carries_signal": bool(mag_approx_full and phase_approx_chance),
        "B_phase_present_not_routed": bool(phase_gt),   # combine with established theta-null in text
        "C_phase_adds_increment": bool(full_vs_mag["full_gt_mag"]),
        "full_minus_mag": full_vs_mag,
    }

    report = {
        "prereg": "PHASE_AVAIL_PREREG.md (frozen before results)",
        "protocol": "LOSO 15-fold, subject-level 3-class balanced_acc (chance=0.333); "
                    "segments->trial (mean class-prob)->subject; subject = statistical unit.",
        "representation": {
            "stft_n_fft": mc.STFT_N_FFT, "stft_hop": mc.STFT_HOP,
            "fs_effective_hz": round(mc.EFFECTIVE_FRAME_RATE_HZ, 4),
            "kept_bins": mc.KEPT_BIN_IDX.tolist(),
            "kept_freqs_hz": [round(float(f), 3) for f in mc.KEPT_FREQS],
            "blocks": {"MAG": "log mean-power per (chan,bin), 1612d, phase discarded",
                       "PHASE": "mean unit-phasor [Re,Im] per (chan,bin), 3224d, magnitude removed",
                       "FULL": "concat(MAG,PHASE), 4836d"}},
        "thresholds": {"delta": DELTA, "tost_margin": TOST_MARGIN, "chance": CHANCE},
        "results": res,
        "verdict": verdict,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(cfg.RESULTS_DIR / "seed_phase_availability_lr.json", "w") as f:
        json.dump(report, f, indent=2)
    shutil.copy(cfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")
    prereg = Path(__file__).resolve().parent.parent / "PHASE_AVAIL_PREREG.md"
    if prereg.exists():
        shutil.copy(prereg, run_dir / "PHASE_AVAIL_PREREG.md")
    shutil.copy(Path(__file__).resolve(), run_dir / "seed_phase_availability_lr.py")

    print("\n=== VERDICT (pre-registered) ===")
    print(f"  MAG   {res['MAG']['subject_level_balanced_acc']['mean']:.3f}  "
          f"PHASE {res['PHASE']['subject_level_balanced_acc']['mean']:.3f}  "
          f"FULL  {res['FULL']['subject_level_balanced_acc']['mean']:.3f}")
    print(f"  FULL-MAG diff {full_vs_mag['mean_diff']:+.3f} (95% CI "
          f"[{full_vs_mag['ci95'][0]:+.3f},{full_vs_mag['ci95'][1]:+.3f}], t_p={full_vs_mag['t_p']:.3f})")
    print(f"  A={verdict['A_magnitude_carries_signal']}  B={verdict['B_phase_present_not_routed']}  "
          f"C={verdict['C_phase_adds_increment']}")
    print(f"elapsed {time.time()-t0:.1f}s; archived {run_dir}")
    return report


if __name__ == "__main__":
    main()
