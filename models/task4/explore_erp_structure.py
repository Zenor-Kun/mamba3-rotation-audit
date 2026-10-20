"""
Task 4 (ERP direction), step 1: STRUCTURE-ONLY exploration of the 128-channel
ERP .raw (EGI RAW) files -- the ERP analogue of Task 0's resting-state .mat
exploration. NO preprocessing, NO modeling here -- just: what does this data
actually look like, does it have the same subjects/labels as the resting-
state cohort, and can Task 1's manifest/splits be reused.

MNE is required to read EGI .raw files and is NOT installed in the locked
mamba3 env (that env's torch/mamba stack must not be touched, per project
rule). MNE 1.12.1 is already present in the base conda env
(/root/miniconda3/bin/python) -- this script must be run with THAT
interpreter, not mamba3's.

Reports, per subject and aggregated:
  - n_channels (EEG vs stim/event channels), sampling rate, total duration
  - event/trigger codes present and their counts (the ERP-specific structure
    resting-state never had)
  - trial count (using 'bgin' as the trial-boundary marker -- one per trial,
    matches condition-cue counts summed) and per-trial timing (fix->cue->
    isi->dot->wrp offsets, bgin-to-bgin period)
  - cross-check: same 53 subjects as the resting-state manifest? same
    MDD/HC labels?
  - whether Task 1's subject manifest / splits_128rest.json can be reused
    as-is for ERP (same subject universe -> yes, no need to re-derive
    subject-independent folds)
"""
import json
import time
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd

ERP_DIR = Path("/root/MODMA/EEG_128channels_ERP_lanzhou_2015")
TASK1_DIR = Path("/root/newcode/task1")
TASK4_DIR = Path("/root/newcode/task4")
RESULTS_DIR = TASK4_DIR / "results"
RUNS_DIR = TASK4_DIR / "runs"

mne.set_log_level("WARNING")


def list_erp_files():
    files = sorted(ERP_DIR.glob("*.raw"))
    out = {}
    for f in files:
        sid = f.name[:8]
        assert sid.isdigit() and len(sid) == 8, f"unexpected filename pattern: {f.name}"
        assert sid not in out, f"duplicate ERP file for subject {sid}: {out[sid]} vs {f}"
        out[sid] = f
    return out


def explore_subject(sid, path):
    raw = mne.io.read_raw_egi(str(path), preload=False, events_as_annotations=True)
    ch_types = raw.get_channel_types()
    type_counts = dict(Counter(ch_types))

    ann = raw.annotations
    onset = ann.onset
    desc = ann.description
    order = np.argsort(onset)
    onset, desc = onset[order], desc[order]
    event_counts = dict(Counter(desc))

    n_trials_bgin = event_counts.get("bgin", 0)
    cue_codes = [c for c in event_counts if c.endswith("cue")]
    n_trials_by_cue_sum = sum(event_counts.get(c, 0) for c in cue_codes)

    # per-trial timing: for each bgin, find the offsets of the next fix/cue/isi/dot/wrp
    # of the SAME trial (condition prefix inferred from the fix event right after bgin)
    bgin_idx = np.where(desc == "bgin")[0]
    trial_periods = np.diff(onset[bgin_idx]) if len(bgin_idx) > 1 else np.array([])

    sub_event_offsets = {"fix_after_bgin": [], "cue_after_fix": [], "isi_after_cue": [],
                          "dot_after_isi": [], "wrp_after_dot": []}
    for bi in bgin_idx[:50]:  # sample the first 50 trials -- timing structure is stereotyped, no need for all
        t_bgin = onset[bi]
        # look at the next few events after this bgin
        following = [(onset[j], desc[j]) for j in range(bi + 1, min(bi + 7, len(desc)))]
        fix = next((t for t, d in following if d.endswith("fix")), None)
        cue = next((t for t, d in following if d.endswith("cue")), None)
        isi = next((t for t, d in following if d.endswith("isi")), None)
        dot = next((t for t, d in following if d.endswith("dot")), None)
        wrp = next((t for t, d in following if d.endswith("wrp")), None)
        if fix is not None:
            sub_event_offsets["fix_after_bgin"].append(fix - t_bgin)
        if fix is not None and cue is not None:
            sub_event_offsets["cue_after_fix"].append(cue - fix)
        if cue is not None and isi is not None:
            sub_event_offsets["isi_after_cue"].append(isi - cue)
        if isi is not None and dot is not None:
            sub_event_offsets["dot_after_isi"].append(dot - isi)
        if dot is not None and wrp is not None:
            sub_event_offsets["wrp_after_dot"].append(wrp - dot)

    return {
        "subject_id": sid,
        "n_channels_total": len(raw.ch_names),
        "channel_type_counts": type_counts,
        "sfreq": float(raw.info["sfreq"]),
        "n_times": int(raw.n_times),
        "duration_sec": float(raw.n_times / raw.info["sfreq"]),
        "event_counts": event_counts,
        "n_trials_bgin": int(n_trials_bgin),
        "n_trials_by_cue_sum": int(n_trials_by_cue_sum),
        "cue_condition_counts": {c: event_counts.get(c, 0) for c in sorted(cue_codes)},
        "trial_period_bgin_to_bgin_median_sec": float(np.median(trial_periods)) if len(trial_periods) else None,
        "trial_period_bgin_to_bgin_min_max": [float(trial_periods.min()), float(trial_periods.max())] if len(trial_periods) else None,
        "sub_event_offsets_median_sec": {k: (float(np.median(v)) if v else None) for k, v in sub_event_offsets.items()},
    }


def cross_check_labels():
    erp_info = pd.read_excel(ERP_DIR / "subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx")
    erp_info["subject_id"] = erp_info["subject id"].astype(str).str.zfill(8)
    erp_labels = dict(zip(erp_info["subject_id"], erp_info["type"]))

    rest = pd.read_csv(TASK1_DIR / "manifest_128rest.csv", dtype={"subject_id": str})
    rest_labels = dict(zip(rest["subject_id"], rest["type"]))

    erp_ids = set(erp_labels)
    rest_ids = set(rest_labels)
    mismatches = [(sid, erp_labels[sid], rest_labels[sid]) for sid in (rest_ids & erp_ids)
                  if erp_labels[sid] != rest_labels[sid]]
    return {
        "n_erp_subjects": len(erp_ids),
        "n_resting_subjects": len(rest_ids),
        "same_subject_set": erp_ids == rest_ids,
        "resting_minus_erp": sorted(rest_ids - erp_ids),
        "erp_minus_resting": sorted(erp_ids - rest_ids),
        "n_label_mismatches": len(mismatches),
        "label_mismatches": mismatches,
        "erp_label_distribution": dict(pd.Series(list(erp_labels.values())).value_counts()),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{ts}_explore_erp_structure"
    run_dir.mkdir(parents=True)

    files = list_erp_files()
    print(f"found {len(files)} ERP .raw files (one per subject, no duplicates)")

    per_subject = []
    log_lines = []
    t0 = time.time()
    for sid, path in files.items():
        r = explore_subject(sid, path)
        per_subject.append(r)
        line = (f"{sid}: sfreq={r['sfreq']} n_ch={r['n_channels_total']} "
                f"duration={r['duration_sec']:.1f}s n_trials(bgin)={r['n_trials_bgin']} "
                f"n_trials(cue_sum)={r['n_trials_by_cue_sum']} "
                f"trial_period_median={r['trial_period_bgin_to_bgin_median_sec']:.3f}s")
        print(line); log_lines.append(line)

    label_check = cross_check_labels()

    # aggregate consistency checks
    sfreqs = sorted(set(r["sfreq"] for r in per_subject))
    n_ch_totals = sorted(set(r["n_channels_total"] for r in per_subject))
    durations = np.array([r["duration_sec"] for r in per_subject])
    n_trials_bgin = np.array([r["n_trials_bgin"] for r in per_subject])
    all_event_code_sets = [frozenset(r["event_counts"].keys()) for r in per_subject]
    unique_event_code_sets = set(all_event_code_sets)

    splits_path = TASK1_DIR / "splits_128rest.json"
    with open(splits_path) as f:
        splits = json.load(f)
    split_subject_ids = set()
    for seed_splits in splits.values():
        for fold_def in seed_splits.values():
            split_subject_ids.update(fold_def["train"])
            split_subject_ids.update(fold_def["test"])
    erp_ids = set(files.keys())
    splits_reusable = split_subject_ids == erp_ids

    aggregate = {
        "n_subjects": len(per_subject),
        "sfreq_values_seen": sfreqs,
        "n_channels_total_values_seen": n_ch_totals,
        "channel_type_counts_example": per_subject[0]["channel_type_counts"],
        "duration_sec": {"min": float(durations.min()), "median": float(np.median(durations)),
                          "max": float(durations.max())},
        "n_trials_bgin": {"min": int(n_trials_bgin.min()), "median": float(np.median(n_trials_bgin)),
                           "max": int(n_trials_bgin.max()),
                           "all_equal": bool(n_trials_bgin.min() == n_trials_bgin.max())},
        "n_unique_event_code_sets_across_subjects": len(unique_event_code_sets),
        "event_code_set_example": sorted(next(iter(unique_event_code_sets))),
        "label_cross_check": label_check,
        "splits_128rest_json_subject_set_matches_erp": splits_reusable,
        "splits_128rest_json_n_subjects": len(split_subject_ids),
    }

    report = {"aggregate": aggregate, "per_subject": per_subject}
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(run_dir / "explore.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    out_json = RESULTS_DIR / "erp_structure_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n=== Task 4 ERP structure exploration summary (n={len(per_subject)} subjects) ===")
    print(f"sfreq: {sfreqs}   n_channels_total: {n_ch_totals}   "
          f"channel types (subject 0): {aggregate['channel_type_counts_example']}")
    print(f"duration_sec: min={aggregate['duration_sec']['min']:.1f} "
          f"median={aggregate['duration_sec']['median']:.1f} max={aggregate['duration_sec']['max']:.1f}")
    print(f"n_trials (bgin marker): min={aggregate['n_trials_bgin']['min']} "
          f"median={aggregate['n_trials_bgin']['median']} max={aggregate['n_trials_bgin']['max']} "
          f"all_equal={aggregate['n_trials_bgin']['all_equal']}")
    print(f"n_unique_event_code_sets_across_subjects={aggregate['n_unique_event_code_sets_across_subjects']} "
          f"(1 means every subject has the identical event-code vocabulary)")
    print(f"event codes (one subject): {aggregate['event_code_set_example']}")
    print(f"label cross-check: {label_check['n_label_mismatches']} mismatches out of "
          f"{label_check['n_erp_subjects']} subjects; same_subject_set={label_check['same_subject_set']}")
    print(f"splits_128rest.json subject set matches ERP subject set: {splits_reusable}")
    print(f"\narchived to {run_dir}")
    print(f"wrote {out_json}")
    print(f"total elapsed: {time.time() - t0:.1f}s")
    return report


if __name__ == "__main__":
    main()
