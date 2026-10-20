"""
Task 4 ERP Stage 1: main_B-lineage preprocessing -> cue-locked epoching ->
artifact flagging (NOT rejection). No modeling, no band-attribution here
(that's Stage 2). Run with the base conda env's Python (MNE 1.12.1) --
NEVER the locked mamba3 env.

Pipeline per subject:
  1. Read the EGI .raw with MNE (events_as_annotations=True for cue timing).
  2. Drop E129 (== montage's Cz vertex position, the reference row) -- model
     with E1..E128, same convention as resting-state's 128 data channels.
  3. main_B filtering (0.5-45Hz zero-phase bandpass + 50Hz notch, task2's
     UNCHANGED functions) on the continuous 128-channel signal.
  4. Data-driven bad-channel detection (task2's UNCHANGED function, same
     thresholds) on the FILTERED, PRE-REFERENCE signal (see erp_preprocess.py
     docstring for why the order differs from task2's resting-state pipeline).
  5. Average reference EXCLUDING bad channels from the averaging step (new
     function for ERP only -- task2's frozen resting-state avg_reference is
     untouched).
  6. Cue-locked epoching: for every fcue/hcue/scue event, extract samples
     [cue_sample-50 : cue_sample+150) -> (128, 200) raw epoch; baseline-
     correct using the first 50 samples (-200..0ms).
  7. Per-trial artifact metrics (global peak-to-peak on good channels, and
     on a frontal/eye-proximal channel subset) computed for ALL subjects'
     trials FIRST, then MAD-based thresholds derived from the POOLED
     distribution (blind to group/subject) BEFORE any trial is flagged --
     avoids the threshold itself being contaminated by group membership.
  8. Save per subject: raw epoch tensor, baseline-corrected epoch tensor,
     per-trial condition label (f/h/s), per-trial artifact flags (amplitude,
     eye-movement-proxy, combined), bad-channel list. NOTHING IS DROPPED.

All 53 subjects' epoch tensors + metrics kept in memory across the run
(~5GB total, comfortably fits in 31GB RAM) so thresholds can be derived from
the full pooled distribution before any per-subject file is written once,
final, no double I/O.
"""
import json
import time
from collections import Counter

import mne
import numpy as np
import pandas as pd
from scipy import stats

import erp_config as ecfg
from erp_preprocess import preprocess_continuous
from erp_frontal_channels import frontal_channel_indices

mne.set_log_level("WARNING")


def list_erp_files():
    files = sorted(ecfg.ERP_RAW_DIR.glob("*.raw"))
    out = {}
    for f in files:
        sid = f.name[:8]
        assert sid.isdigit() and len(sid) == 8
        out[sid] = f
    return out


def get_cue_events(raw):
    """Returns list of (sample_idx, condition_letter) sorted by time."""
    ann = raw.annotations
    onset, desc = ann.onset, ann.description
    events = []
    for letter, code in ecfg.CUE_EVENT_CODES.items():
        mask = desc == code
        for t in onset[mask]:
            s = int(round(t * ecfg.FS))
            events.append((s, letter))
    events.sort(key=lambda x: x[0])
    return events


def process_subject(sid, path):
    raw = mne.io.read_raw_egi(str(path), preload=True, events_as_annotations=True)
    eeg_names = [f"E{i}" for i in range(1, ecfg.N_DATA_CH + 1)]
    data_128 = raw.get_data(picks=eeg_names)  # (128, T), Volts
    n_times = data_128.shape[1]

    referenced, bad_channels = preprocess_continuous(data_128)
    good_mask = np.ones(ecfg.N_DATA_CH, dtype=bool)
    good_mask[bad_channels] = False

    cue_events = get_cue_events(raw)
    n_oob = 0
    raw_epochs, bc_epochs, conditions = [], [], []
    for s, letter in cue_events:
        start, end = s - ecfg.PRE_SAMPLES, s + ecfg.POST_SAMPLES
        if start < 0 or end > n_times:
            n_oob += 1
            continue
        raw_ep = referenced[:, start:end].astype(np.float32)  # (128, 200)
        baseline_mean = raw_ep[:, :ecfg.BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        bc_ep = raw_ep - baseline_mean
        raw_epochs.append(raw_ep)
        bc_epochs.append(bc_ep)
        conditions.append(letter)

    raw_epochs = np.stack(raw_epochs)  # (n_trials, 128, 200)
    bc_epochs = np.stack(bc_epochs)
    conditions = np.array(conditions)

    ptp_per_ch = np.ptp(bc_epochs, axis=2)  # (n_trials, 128)
    global_ptp = ptp_per_ch[:, good_mask].max(axis=1)  # (n_trials,)

    return {
        "subject_id": sid, "raw_epochs": raw_epochs, "bc_epochs": bc_epochs,
        "conditions": conditions, "bad_channels": bad_channels,
        "global_ptp": global_ptp, "ptp_per_ch": ptp_per_ch,
        "n_trials": raw_epochs.shape[0], "n_oob_trials_dropped": n_oob,
        "n_times_continuous": n_times,
    }


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ecfg.RUNS_DIR / f"{ts}_erp_stage1_preprocess_epoch_flag"
    run_dir.mkdir(parents=True)

    frontal_idx, frontal_names = frontal_channel_indices()
    print(f"frontal/eye-proximal channel subset (n={len(frontal_idx)}): {frontal_names}")

    files = list_erp_files()
    manifest = pd.read_csv(ecfg.MANIFEST_128REST, dtype={"subject_id": str})
    label_of = dict(zip(manifest["subject_id"], manifest["label"]))
    type_of = dict(zip(manifest["subject_id"], manifest["type"]))

    print(f"processing {len(files)} subjects...")
    t0 = time.time()
    subjects_data = {}
    log_lines = []
    for sid, path in files.items():
        r = process_subject(sid, path)
        r["frontal_ptp"] = r["ptp_per_ch"][:, frontal_idx].max(axis=1)
        subjects_data[sid] = r
        line = (f"{sid} ({type_of[sid]}): n_trials={r['n_trials']} "
                f"n_bad_channels={len(r['bad_channels'])} n_oob_dropped={r['n_oob_trials_dropped']}")
        print(line); log_lines.append(line)
    print(f"pass 1 (preprocess+epoch) done in {time.time() - t0:.1f}s")

    # ---- pool per-trial metrics across ALL subjects, derive MAD thresholds ----
    all_global_ptp = np.concatenate([d["global_ptp"] for d in subjects_data.values()])
    all_frontal_ptp = np.concatenate([d["frontal_ptp"] for d in subjects_data.values()])

    def mad_log_threshold(values, k):
        log_v = np.log(values + 1e-12)
        med = np.median(log_v)
        mad = np.median(np.abs(log_v - med)) * 1.4826 + 1e-12
        return med + k * mad

    amp_thresh_log = mad_log_threshold(all_global_ptp, ecfg.ARTIFACT_MAD_K)
    frontal_thresh_log = mad_log_threshold(all_frontal_ptp, ecfg.ARTIFACT_MAD_K)
    print(f"pooled amplitude threshold (log-ptp, k={ecfg.ARTIFACT_MAD_K}): {amp_thresh_log:.4f} "
          f"(ptp={np.exp(amp_thresh_log):.3e} V)")
    print(f"pooled frontal/eye threshold (log-ptp, k={ecfg.ARTIFACT_MAD_K}): {frontal_thresh_log:.4f} "
          f"(ptp={np.exp(frontal_thresh_log):.3e} V)")

    # ---- assign flags, write final per-subject files ----
    per_subject_report = []
    for sid, d in subjects_data.items():
        amp_flag = np.log(d["global_ptp"] + 1e-12) > amp_thresh_log
        eye_flag = np.log(d["frontal_ptp"] + 1e-12) > frontal_thresh_log
        artifact_flag = amp_flag | eye_flag

        out_path = ecfg.EPOCHS_CACHE / f"{sid}.npz"
        np.savez_compressed(
            out_path,
            raw_epochs=d["raw_epochs"], bc_epochs=d["bc_epochs"],
            conditions=d["conditions"], artifact_flag=artifact_flag,
            amp_flag=amp_flag, eye_flag=eye_flag,
            bad_channels=np.array(d["bad_channels"], dtype=np.int64),
            global_ptp=d["global_ptp"], frontal_ptp=d["frontal_ptp"],
        )

        cond_retention = {}
        for letter in ecfg.CUE_EVENT_CODES:
            m = d["conditions"] == letter
            n_cond = int(m.sum())
            n_clean = int((~artifact_flag[m]).sum())
            cond_retention[letter] = {
                "n_trials": n_cond, "n_clean": n_clean,
                "retention_rate": n_clean / n_cond if n_cond else None,
            }

        overall_artifact_rate = float(artifact_flag.mean())
        per_subject_report.append({
            "subject_id": sid, "label": int(label_of[sid]), "type": type_of[sid],
            "n_trials": d["n_trials"], "n_oob_trials_dropped": d["n_oob_trials_dropped"],
            "n_bad_channels": len(d["bad_channels"]), "bad_channels": d["bad_channels"],
            "n_amp_flagged": int(amp_flag.sum()), "n_eye_flagged": int(eye_flag.sum()),
            "n_artifact_flagged": int(artifact_flag.sum()),
            "overall_artifact_rate": overall_artifact_rate,
            "overall_retention_rate": 1.0 - overall_artifact_rate,
            "condition_retention": cond_retention,
        })

    # ---- MDD vs HC group test on per-subject artifact rate ----
    mdd_rates = [r["overall_artifact_rate"] for r in per_subject_report if r["type"] == "MDD"]
    hc_rates = [r["overall_artifact_rate"] for r in per_subject_report if r["type"] == "HC"]
    u_stat, p_value = stats.mannwhitneyu(mdd_rates, hc_rates, alternative="two-sided")

    splits_check = json.load(open(ecfg.SPLITS_JSON))
    split_ids = set()
    for seed_splits in splits_check.values():
        for fold_def in seed_splits.values():
            split_ids.update(fold_def["train"]); split_ids.update(fold_def["test"])

    aggregate = {
        "n_subjects": len(per_subject_report),
        "n_trials_per_subject": {"min": min(r["n_trials"] for r in per_subject_report),
                                  "median": float(np.median([r["n_trials"] for r in per_subject_report])),
                                  "max": max(r["n_trials"] for r in per_subject_report)},
        "amp_threshold_log_ptp": float(amp_thresh_log), "amp_threshold_ptp_volts": float(np.exp(amp_thresh_log)),
        "frontal_threshold_log_ptp": float(frontal_thresh_log), "frontal_threshold_ptp_volts": float(np.exp(frontal_thresh_log)),
        "frontal_channels": frontal_names,
        "overall_artifact_rate": {
            "min": float(min(r["overall_artifact_rate"] for r in per_subject_report)),
            "median": float(np.median([r["overall_artifact_rate"] for r in per_subject_report])),
            "max": float(max(r["overall_artifact_rate"] for r in per_subject_report)),
        },
        "mdd_vs_hc_artifact_rate": {
            "mdd_mean": float(np.mean(mdd_rates)), "mdd_median": float(np.median(mdd_rates)), "n_mdd": len(mdd_rates),
            "hc_mean": float(np.mean(hc_rates)), "hc_median": float(np.median(hc_rates)), "n_hc": len(hc_rates),
            "mannwhitney_u": float(u_stat), "p_value": float(p_value),
        },
        "splits_128rest_subject_set_matches_erp": split_ids == set(files.keys()),
    }

    report = {"aggregate": aggregate, "per_subject": per_subject_report}
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(run_dir / "stage1.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(ecfg.RESULTS_DIR / "erp_stage1_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n=== ERP Stage 1 summary (n={len(per_subject_report)} subjects) ===")
    print(f"trials/subject: {aggregate['n_trials_per_subject']}")
    print(f"overall artifact rate: {aggregate['overall_artifact_rate']}")
    print(f"MDD vs HC artifact rate: MDD mean={aggregate['mdd_vs_hc_artifact_rate']['mdd_mean']:.3f} "
          f"HC mean={aggregate['mdd_vs_hc_artifact_rate']['hc_mean']:.3f} "
          f"Mann-Whitney p={aggregate['mdd_vs_hc_artifact_rate']['p_value']:.4f}")
    print(f"splits_128rest.json subject set matches ERP: {aggregate['splits_128rest_subject_set_matches_erp']}")
    print(f"per-subject epoch tensors written to {ecfg.EPOCHS_CACHE}/<sid>.npz")
    print(f"archived to {run_dir}")
    print(f"total elapsed: {time.time() - t0:.1f}s")
    return report


if __name__ == "__main__":
    main()
