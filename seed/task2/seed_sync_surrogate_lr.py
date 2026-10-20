"""
Circular-shift surrogate control for the PLV/wPLI cross-channel synchrony blocks
(RQ0, group 4, defensive control). Per SEGMENT and per CHANNEL, apply an
independent random circular time-shift to the raw signal BEFORE the STFT. This
preserves each channel's own magnitude spectrum (circular shift is phase-only on
the full-length signal) while destroying inter-channel phase alignment. Prediction
(positive control): PLV/wPLI collapse toward their null, confirming the synchrony
features encode genuine cross-channel phase rather than a per-channel spectral artifact.

TARGETED and NOT over-broad: single-channel spectra (hence MAG) are left essentially
intact; only cross-channel phase relations are broken. Same LOSO LR pipeline as the
real run; compares surrogate PLV/wPLI to the frozen real PLV/wPLI (same folds/subjects).
Reads splits READ-ONLY. Archives to runs/<ts>_seed_sync_surrogate_lr/.
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
import seed_mamba_config as mc
import seed2_config as cfg
import seed_sync_availability_lr as SY     # reuse constants + connectivity core

SURR_SEED = 20260713                        # fixed for reproducibility (pre-registered choice)
FEATCACHE = Path(__file__).resolve().parent / "sync_surrogate_featcache"
FEATCACHE.mkdir(exist_ok=True)
REAL_RUN_GLOB = "*_seed_sync_availability_lr"
DEV = SY.DEV
KEPT = SY.KEPT


def _connectivity_from_raw(raw_shifted):
    """raw_shifted: torch (b,62,800) on DEV -> (plv, wpli) numpy (b,7564) each.
    Identical STFT + PLV/wPLI core as the real run (SY.compute_sync_for_subject)."""
    b, C, T = raw_shifted.shape
    window = torch.hann_window(mc.STFT_N_FFT, device=DEV)
    kept = KEPT.to(DEV)
    eps = 1e-8
    xr = raw_shifted.reshape(b * C, T)
    spec = torch.stft(xr, n_fft=mc.STFT_N_FFT, hop_length=mc.STFT_HOP,
                      win_length=mc.STFT_N_FFT, window=window,
                      center=False, return_complex=True)
    spec = spec.index_select(1, kept)
    Fb, Tf = spec.shape[1], spec.shape[2]
    Z = spec.reshape(b, C, Fb, Tf).permute(0, 2, 1, 3)          # (b,26,62,113)
    U = Z / (Z.abs() + eps)
    plv = (torch.einsum('bfct,bfdt->bfcd', U, U.conj()) / Tf).abs()
    sumS = torch.einsum('bfct,bfdt->bfcd', Z, Z.conj())
    num = sumS.imag.abs()
    den = torch.zeros_like(num)
    for t in range(Tf):
        zt = Z[..., t]
        den += (zt.unsqueeze(-1) * zt.conj().unsqueeze(-2)).imag.abs()
    wpli = num / (den + eps)
    return (SY._band_average_uppertri(plv.float().cpu().numpy()),
            SY._band_average_uppertri(wpli.float().cpu().numpy()))


def compute_surrogate_for_subject(sid):
    cache = FEATCACHE / f"{sid}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["plv"], d["wpli"], d["label"], d["trial"]
    seg = np.load(mc.SEGMENTS_CACHE / f"{sid}.npz")
    raw = seg["raw_segments"]
    label, trial = seg["seg_label"], seg["seg_trial_id"]
    N, C, T = raw.shape
    rng = np.random.default_rng(SURR_SEED + sid)          # per-subject stream, fixed
    plv_out = np.empty((N, SY.N_SYNC), dtype=np.float32)
    wpli_out = np.empty((N, SY.N_SYNC), dtype=np.float32)
    tvec = torch.arange(T, device=DEV)
    BS = 32
    with torch.no_grad():
        for i in range(0, N, BS):
            xb = torch.from_numpy(raw[i:i + BS]).to(DEV)      # (b,62,800)
            b = xb.shape[0]
            shifts = torch.from_numpy(rng.integers(1, T, size=(b, C))).to(DEV)  # per seg,chan
            idx = (tvec[None, None, :] - shifts[..., None]) % T                 # (b,62,800)
            xs = torch.gather(xb, 2, idx)                                       # circular shift
            plv, wpli = _connectivity_from_raw(xs)
            plv_out[i:i + b] = plv
            wpli_out[i:i + b] = wpli
    np.savez(cache, plv=plv_out, wpli=wpli_out, label=label, trial=trial)
    return plv_out, wpli_out, label, trial


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_seed_sync_surrogate_lr"
    run_dir.mkdir(parents=True)
    t0 = time.time()

    splits = json.load(open(cfg.SPLITS_JSON))
    real_runs = sorted(cfg.RUNS_DIR.glob(REAL_RUN_GLOB))
    real = json.load(open(real_runs[-1] / "report.json"))
    real_plv = np.array(real["results"]["PLV"]["subject_level_balanced_acc"]["per_fold"])
    real_wpli = np.array(real["results"]["wPLI"]["subject_level_balanced_acc"]["per_fold"])

    store = {}
    print(f"device={DEV}; circular-shift surrogate (seed {SURR_SEED}); connectivity per subject ...",
          flush=True)
    for sid in range(1, cfg.scfg.N_SUBJECTS + 1):
        plv, wpli, label, trial = compute_surrogate_for_subject(sid)
        store[sid] = {"plv": plv, "wpli": wpli, "label": label, "trial": trial}
        print(f"  S{sid:02d}: N={len(label)}", flush=True)

    res, perfold = {}, {}
    for block in ["PLV", "wPLI"]:
        sb, gb = SY.run_block(store, splits, block)
        perfold[block] = sb
        res[block] = {"subject_level_balanced_acc": {
            "mean": float(sb.mean()), "std": float(sb.std()),
            "per_fold": [float(x) for x in sb]},
            "segment_level_mean": float(gb.mean()),
            "vs_chance": SY.vs_chance(sb)}
        print(f"[surr {block:4s}] {sb.mean():.3f} +- {sb.std():.3f} "
              f"(above_chance={res[block]['vs_chance']['above_chance']})", flush=True)

    # real - surrogate paired drop (same folds/subjects)
    drop = {"PLV": SY.paired(real_plv, perfold["PLV"]),
            "wPLI": SY.paired(real_wpli, perfold["wPLI"])}

    report = {
        "control": "per-segment per-channel random circular time-shift before STFT "
                   "(preserves per-channel spectrum, destroys cross-channel phase alignment)",
        "surrogate_seed": SURR_SEED,
        "real_run": real_runs[-1].name,
        "real_reference": {"PLV": float(real_plv.mean()), "wPLI": float(real_wpli.mean())},
        "surrogate_results": res,
        "real_minus_surrogate": drop,
        "thresholds": {"chance": SY.CHANCE, "delta": SY.DELTA, "tost_margin": SY.TOST_MARGIN},
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(cfg.RESULTS_DIR / "seed_sync_surrogate_lr.json", "w") as f:
        json.dump(report, f, indent=2)
    shutil.copy(cfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")
    shutil.copy(Path(__file__).resolve(), run_dir / "seed_sync_surrogate_lr.py")

    print("\n=== SURROGATE (positive control) ===", flush=True)
    print(f"  PLV  real {real_plv.mean():.3f} -> surrogate {res['PLV']['subject_level_balanced_acc']['mean']:.3f} "
          f"(drop {drop['PLV']['mean_diff']:+.3f}, t_p={drop['PLV']['t_p']:.3f})")
    print(f"  wPLI real {real_wpli.mean():.3f} -> surrogate {res['wPLI']['subject_level_balanced_acc']['mean']:.3f} "
          f"(drop {drop['wPLI']['mean_diff']:+.3f}, t_p={drop['wPLI']['t_p']:.3f})")
    print(f"elapsed {time.time()-t0:.1f}s; archived {run_dir}", flush=True)
    return report


if __name__ == "__main__":
    main()
