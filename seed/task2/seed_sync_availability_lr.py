"""
Cross-channel phase-synchrony availability (RQ0, group 4): does band-averaged
inter-channel phase synchrony (PLV / wPLI) carry SEED subject-independent
discriminative information, and does it add anything BEYOND the magnitude
representation? LR-only, NO Mamba. Companion to seed_phase_availability_lr.py.

Same complex STFT front-end as the Mamba variants (n_fft=128, hop=6, hann,
center=False, 26 in-band bins). Two connectivity blocks + one increment block,
fed to the identical LOSO LR pipeline (train-fold z-score, multinomial LR,
segment->trial->subject bal_acc, chance 0.333, subject = statistical unit):

  PLV      = |mean_t exp(i*dphi_ij)| per channel-pair, band-averaged   [7564d]
  wPLI     = |mean_t Im(S_ij)| / mean_t|Im(S_ij)| per pair, band-avg   [7564d]
  MAG_PLV  = concat(MAG, PLV)   -> paired increment test vs MAG-only    [9176d]

Upper-triangle of 62 channels = 1891 pairs; 4 canonical bands (theta/alpha/
beta/gamma per mc.KEPT_BIN_BAND) => 1891*4 = 7564 features per metric.

MAG-only per-fold subject values are loaded from the frozen phase-availability
run (same folds, same subjects) so MAG_PLV-vs-MAG is a valid paired increment.
Reads splits_seed_loso.json READ-ONLY. Archives to runs/<ts>_seed_sync_availability_lr/.
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
DELTA = 0.02          # practical margin (pre-registered, same as phase audit)
TOST_MARGIN = 0.03    # equivalence margin, consistent with main paper
FEATCACHE = Path(__file__).resolve().parent / "sync_avail_featcache"
FEATCACHE.mkdir(exist_ok=True)
PHASE_FEATCACHE = Path(__file__).resolve().parent / "phase_avail_featcache"  # reuse MAG from phase audit
PHASE_RUN = cfg.RUNS_DIR / "20260713-054351_seed_phase_availability_lr"      # frozen MAG per-fold
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KEPT = torch.tensor(mc.KEPT_BIN_IDX, dtype=torch.long)

C_CH = 62
IU = np.triu_indices(C_CH, k=1)                    # 1891 upper-tri pairs
N_PAIRS = IU[0].size
BANDS = ["theta", "alpha", "beta", "gamma"]
# indices (0..25) into the kept-bin axis for each band
BAND_BINS = {b: [i for i, lab in enumerate(mc.KEPT_BIN_BAND) if lab == b] for b in BANDS}
N_SYNC = N_PAIRS * len(BANDS)                       # 1891 * 4 = 7564


def _band_average_uppertri(mat_bins):
    """mat_bins: (B, 26, 62, 62) real -> (B, N_PAIRS*4) band-averaged upper-tri."""
    B = mat_bins.shape[0]
    out = np.empty((B, N_SYNC), dtype=np.float32)
    for k, band in enumerate(BANDS):
        m = mat_bins[:, BAND_BINS[band]].mean(axis=1)          # (B, 62, 62) avg over band bins
        out[:, k * N_PAIRS:(k + 1) * N_PAIRS] = m[:, IU[0], IU[1]]
    return out


def compute_sync_for_subject(sid):
    """PLV (N,7564) and wPLI (N,7564) connectivity blocks for one subject; cache to disk."""
    cache = FEATCACHE / f"{sid}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["plv"], d["wpli"], d["label"], d["trial"]
    seg = np.load(mc.SEGMENTS_CACHE / f"{sid}.npz")
    raw = seg["raw_segments"]                       # (N, 62, 800) float32
    label, trial = seg["seg_label"], seg["seg_trial_id"]
    N, C, T = raw.shape
    window = torch.hann_window(mc.STFT_N_FFT, device=DEV)
    kept = KEPT.to(DEV)
    plv_out = np.empty((N, N_SYNC), dtype=np.float32)
    wpli_out = np.empty((N, N_SYNC), dtype=np.float32)
    BS = 32                                          # segments per GPU batch
    eps = 1e-8
    with torch.no_grad():
        for i in range(0, N, BS):
            xb = torch.from_numpy(raw[i:i + BS]).to(DEV)       # (b,62,800)
            b = xb.shape[0]
            xr = xb.reshape(b * C, T)
            spec = torch.stft(xr, n_fft=mc.STFT_N_FFT, hop_length=mc.STFT_HOP,
                              win_length=mc.STFT_N_FFT, window=window,
                              center=False, return_complex=True)   # (b*62, 65, 113)
            spec = spec.index_select(1, kept)                      # (b*62, 26, 113)
            Fb, Tf = spec.shape[1], spec.shape[2]
            Z = spec.reshape(b, C, Fb, Tf).permute(0, 2, 1, 3)     # (b, 26, 62, 113)
            # --- PLV: |mean_t u_c conj(u_d)|  (unit phasors) ---
            U = Z / (Z.abs() + eps)                                # (b,26,62,113)
            M = torch.einsum('bfct,bfdt->bfcd', U, U.conj()) / Tf  # (b,26,62,62) complex
            plv = M.abs()                                          # (b,26,62,62)
            # --- wPLI: |sum_t Im(S)| / sum_t |Im(S)|,  S_cd = Z_c conj(Z_d) ---
            sumS = torch.einsum('bfct,bfdt->bfcd', Z, Z.conj())    # (b,26,62,62) = sum_t S
            num = sumS.imag.abs()                                  # |sum_t Im(S)|
            den = torch.zeros_like(num)
            for t in range(Tf):
                zt = Z[..., t]                                     # (b,26,62)
                cross_im = (zt.unsqueeze(-1) * zt.conj().unsqueeze(-2)).imag  # (b,26,62,62)
                den += cross_im.abs()
            wpli = num / (den + eps)                               # (b,26,62,62)
            plv_out[i:i + b] = _band_average_uppertri(plv.float().cpu().numpy())
            wpli_out[i:i + b] = _band_average_uppertri(wpli.float().cpu().numpy())
    np.savez(cache, plv=plv_out, wpli=wpli_out, label=label, trial=trial)
    return plv_out, wpli_out, label, trial


def load_mag_for_subject(sid):
    """MAG block (N,1612) from the frozen phase-availability feature cache (identical front-end)."""
    d = np.load(PHASE_FEATCACHE / f"{sid}.npz")
    return d["mag"]


def block_matrix(store, sid, block):
    s = store[sid]
    if block == "PLV":
        return s["plv"]
    if block == "wPLI":
        return s["wpli"]
    if block == "MAG_PLV":
        return np.concatenate([s["mag"], s["plv"]], axis=1)
    raise ValueError(block)


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
        print(f"  [{block:7s}] fold test S{test_sid:02d}: subj bal_acc {subj_bacc[-1]:.3f}", flush=True)
    return np.array(subj_bacc), np.array(seg_bacc)


def vs_chance(vals):
    t_p = float(stats.ttest_1samp(vals, CHANCE).pvalue)
    try:
        w_p = float(stats.wilcoxon(vals - CHANCE).pvalue)
    except ValueError:
        w_p = 1.0
    mean = float(vals.mean())
    return {"mean": mean, "std": float(vals.std()), "t_p": t_p, "wilcoxon_p": w_p,
            "approx_chance": bool(t_p >= 0.05 and w_p >= 0.05 and mean <= CHANCE + DELTA),
            "above_chance": bool(t_p < 0.05 and w_p < 0.05 and mean - CHANCE > DELTA)}


def paired(a, b):
    """a - b across subjects (paired). a=augmented, b=baseline."""
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
    return {"mean_diff": md, "t_p": t_p, "wilcoxon_p": w_p,
            "ci95": [ci95[0], ci95[1]], "ci90": [ci90[0], ci90[1]],
            "tost_equivalent_pm03": bool(ci90[0] > -TOST_MARGIN and ci90[1] < TOST_MARGIN),
            "equiv_practical": bool(t_p >= 0.05 and abs(md) <= DELTA),
            "aug_gt_base": bool(t_p < 0.05 and md > DELTA)}


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_seed_sync_availability_lr"
    run_dir.mkdir(parents=True)
    t0 = time.time()

    splits = json.load(open(cfg.SPLITS_JSON))
    mag_perfold = np.array(json.load(open(PHASE_RUN / "report.json"))
                           ["results"]["MAG"]["subject_level_balanced_acc"]["per_fold"])

    store = {}
    print(f"device={DEV}; computing PLV/wPLI connectivity per subject "
          f"({N_PAIRS} pairs x {len(BANDS)} bands = {N_SYNC}d) ...", flush=True)
    for sid in range(1, cfg.scfg.N_SUBJECTS + 1):
        plv, wpli, label, trial = compute_sync_for_subject(sid)
        store[sid] = {"plv": plv, "wpli": wpli, "mag": load_mag_for_subject(sid),
                      "label": label, "trial": trial}
        print(f"  S{sid:02d}: N={len(label)} plv{plv.shape} wpli{wpli.shape}", flush=True)

    res, perfold = {}, {}
    for block in ["PLV", "wPLI", "MAG_PLV"]:
        sb, gb = run_block(store, splits, block)
        perfold[block] = sb
        res[block] = {"subject_level_balanced_acc": {
            "mean": float(sb.mean()), "std": float(sb.std()),
            "per_fold": [float(x) for x in sb]},
            "segment_level_mean": float(gb.mean()),
            "vs_chance": vs_chance(sb)}
        print(f"[{block:7s}] subject bal_acc = {sb.mean():.3f} +- {sb.std():.3f}  "
              f"(seg {gb.mean():.3f})  above_chance={res[block]['vs_chance']['above_chance']}",
              flush=True)

    magplv_vs_mag = paired(perfold["MAG_PLV"], mag_perfold)

    report = {
        "prereg": "PHASE_AVAIL_PREREG.md group-4 (PLV/wPLI cross-channel synchrony)",
        "protocol": "LOSO 15-fold, subject-level 3-class balanced_acc (chance=0.333); "
                    "segments->trial (mean class-prob)->subject; subject = statistical unit. "
                    "MAG-only per-fold loaded from frozen phase-availability run for paired increment.",
        "representation": {
            "stft_n_fft": mc.STFT_N_FFT, "stft_hop": mc.STFT_HOP,
            "fs_effective_hz": round(mc.EFFECTIVE_FRAME_RATE_HZ, 4),
            "channel_pairs": N_PAIRS, "bands": BANDS, "sync_dim": N_SYNC,
            "blocks": {
                "PLV": "|mean_t exp(i*dphi_ij)| per pair, band-averaged, 7564d",
                "wPLI": "|mean_t Im(S_ij)|/mean_t|Im(S_ij)| per pair, band-averaged, 7564d",
                "MAG_PLV": "concat(MAG-1612, PLV-7564)=9176d, increment vs MAG-only"}},
        "thresholds": {"delta": DELTA, "tost_margin": TOST_MARGIN, "chance": CHANCE},
        "mag_only_reference": {"mean": float(mag_perfold.mean()),
                               "per_fold": [float(x) for x in mag_perfold],
                               "source": str(PHASE_RUN.name)},
        "results": res,
        "magplv_minus_mag": magplv_vs_mag,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(cfg.RESULTS_DIR / "seed_sync_availability_lr.json", "w") as f:
        json.dump(report, f, indent=2)
    shutil.copy(cfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")
    prereg = Path(__file__).resolve().parent.parent / "PHASE_AVAIL_PREREG.md"
    if prereg.exists():
        shutil.copy(prereg, run_dir / "PHASE_AVAIL_PREREG.md")
    shutil.copy(Path(__file__).resolve(), run_dir / "seed_sync_availability_lr.py")

    print("\n=== SYNC RESULT ===", flush=True)
    print(f"  PLV  {res['PLV']['subject_level_balanced_acc']['mean']:.3f}  "
          f"wPLI {res['wPLI']['subject_level_balanced_acc']['mean']:.3f}  "
          f"MAG {mag_perfold.mean():.3f}  MAG_PLV {res['MAG_PLV']['subject_level_balanced_acc']['mean']:.3f}")
    print(f"  MAG_PLV-MAG diff {magplv_vs_mag['mean_diff']:+.3f} (95% CI "
          f"[{magplv_vs_mag['ci95'][0]:+.3f},{magplv_vs_mag['ci95'][1]:+.3f}], "
          f"t_p={magplv_vs_mag['t_p']:.3f}, TOST_eq={magplv_vs_mag['tost_equivalent_pm03']})")
    print(f"elapsed {time.time()-t0:.1f}s; archived {run_dir}", flush=True)
    return report


if __name__ == "__main__":
    main()
