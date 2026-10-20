"""
SEED Task 1: subject-level manifest + subject-independent LOSO splits.

Reuses MODMA's split DISCIPLINE (subject is the atomic unit; splits frozen
to a JSON; zero-leak assertions), adapted to SEED's two dataset-specific
leakage risks:
  Leak #1 -- session leak: each subject has 3 recording sessions. All 3
    sessions of a subject MUST stay together in one fold, else the model
    trains and tests on the same person across sessions.
  Leak #2 -- trial/segment leak: labels are TRIAL-level (15 trials/session),
    and later stages will cut each trial into many segments. Every trial and
    every future segment of a subject inherits that subject's fold. Splitting
    on trials or segments would put the same subject on both sides.
Both are prevented structurally by making the split unit = subject: a fold
assignment is over the 15 subject IDs only, so all sessions/trials/segments
follow automatically.

Protocol: Leave-One-Subject-Out (LOSO), 15 folds. Each fold: test = exactly
1 subject, train = the other 14. With only 15 subjects, LOSO gives higher
statistical power than 5-fold. Frozen to splits_seed_loso.json.

This step ONLY builds the manifest + splits + assertions. It does NOT read
EEG signal content, does NOT preprocess, does NOT model. (It reads .mat
variable NAMES/COUNTS to verify 15 trials/session -- structure, not signal.)
Everything archived immutably to seed/runs/<ts>_seed_task1_manifest_splits/.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import seed_config as scfg


def read_gender():
    gender = {}
    for line in scfg.GENDER_FILE.read_text().strip().splitlines():
        sid, g = line.split()
        gender[int(sid)] = g
    assert len(gender) == scfg.N_SUBJECTS, f"expected {scfg.N_SUBJECTS} gender rows, got {len(gender)}"
    return gender


def list_sessions_by_subject():
    """subject_id (int) -> sorted list of session .mat filenames (by date)."""
    by_subj = {}
    for f in sorted(scfg.PREPROC_DIR.glob("*.mat")):
        if f.name == "label.mat":
            continue
        sid = int(f.name.split("_")[0])
        by_subj.setdefault(sid, []).append(f.name)
    for sid in by_subj:
        # filenames are <sid>_<YYYYMMDD>.mat -> sort by the date field chronologically
        by_subj[sid] = sorted(by_subj[sid], key=lambda n: n.split("_")[1])
    return by_subj


def count_trials(mat_path):
    """Number of trial variables (<initials>_eegN) in a session file -- keys
    only, no signal values read into analysis."""
    m = sio.loadmat(mat_path, verify_compressed_data_integrity=False)
    trial_keys = [k for k in m.keys() if not k.startswith("__") and "_eeg" in k]
    return len(trial_keys), sorted(trial_keys, key=lambda k: int(k.split("eeg")[1]))


def build_manifest(log):
    gender = read_gender()
    by_subj = list_sessions_by_subject()

    label_m = sio.loadmat(scfg.LABEL_MAT)["label"].ravel().tolist()
    assert label_m == scfg.TRIAL_LABELS, f"label.mat {label_m} != expected {scfg.TRIAL_LABELS}"

    class_counts = {name: scfg.TRIAL_LABELS.count(v) for v, name in scfg.LABEL_NAMES.items()}
    log.append(f"trial labels (per session): {scfg.TRIAL_LABELS}")
    log.append(f"per-session class counts: {class_counts}")

    rows = []
    subjects = sorted(by_subj.keys())
    assert subjects == list(range(1, scfg.N_SUBJECTS + 1)), f"subjects not 1..15: {subjects}"
    for sid in subjects:
        sessions = by_subj[sid]
        assert len(sessions) == scfg.N_SESSIONS, f"subject {sid}: {len(sessions)} sessions != {scfg.N_SESSIONS}"
        n_trials_per_session = []
        for sess_name in sessions:
            n_tr, _ = count_trials(scfg.PREPROC_DIR / sess_name)
            assert n_tr == scfg.N_TRIALS_PER_SESSION, \
                f"{sess_name}: {n_tr} trials != {scfg.N_TRIALS_PER_SESSION}"
            n_trials_per_session.append(n_tr)
        row = {
            "subject_id": sid,
            "gender": gender[sid],
            "session_1_file": sessions[0],
            "session_2_file": sessions[1],
            "session_3_file": sessions[2],
            "n_trials_per_session": n_trials_per_session[0],  # all 15, verified equal
            "n_sessions": scfg.N_SESSIONS,
            "n_trials_total": scfg.N_SESSIONS * scfg.N_TRIALS_PER_SESSION,
            "n_pos_per_session": class_counts["positive"],
            "n_neu_per_session": class_counts["neutral"],
            "n_neg_per_session": class_counts["negative"],
        }
        rows.append(row)
        log.append(f"subject {sid:2d} ({gender[sid]}): sessions={sessions} "
                   f"trials/session={n_trials_per_session}")
    return pd.DataFrame(rows), class_counts


def build_loso_splits(subjects, log):
    """15-fold LOSO. fold_k -> {test:[subj_k], train:[the other 14]}."""
    splits = {}
    for i, sid in enumerate(subjects):
        train = [s for s in subjects if s != sid]
        splits[f"fold_{i}"] = {"test": [sid], "train": train}
    log.append(f"built LOSO: {len(splits)} folds, test=1 subject each")
    return splits


def assert_zero_leak(splits, subjects):
    all_subjects = set(subjects)
    assert len(splits) == scfg.N_SUBJECTS, f"{len(splits)} folds != {scfg.N_SUBJECTS}"
    test_union = set()
    for name, fold in splits.items():
        test, train = fold["test"], fold["train"]
        # 1. each fold test is exactly 1 subject
        assert len(test) == 1, f"{name}: test has {len(test)} subjects, must be 1"
        # 2. no subject on both sides
        assert set(test).isdisjoint(set(train)), f"{name}: subject leaks across train/test"
        # 3. train+test partition all 15 subjects exactly
        assert set(test) | set(train) == all_subjects, f"{name}: train+test != all subjects"
        assert len(train) == scfg.N_SUBJECTS - 1, f"{name}: train has {len(train)} != 14"
        # 4. no duplicate subject within a fold
        assert len(set(train)) == len(train), f"{name}: duplicate subject in train"
        test_union.update(test)
    # 5. 15 folds cover all 15 subjects, no repeat, no omission
    assert test_union == all_subjects, f"test folds don't cover all subjects: {test_union}"
    return "all zero-leak assertions PASSED"


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = scfg.RUNS_DIR / f"{ts}_seed_task1_manifest_splits"
    run_dir.mkdir(parents=True)
    log = []

    manifest_df, class_counts = build_manifest(log)
    subjects = manifest_df["subject_id"].tolist()

    splits = build_loso_splits(subjects, log)
    assert_msg = assert_zero_leak(splits, subjects)
    log.append(assert_msg)

    # write manifest + frozen splits to the canonical seed/ locations
    manifest_df.to_csv(scfg.MANIFEST_CSV, index=False)
    with open(scfg.SPLITS_JSON, "w") as f:
        json.dump(splits, f, indent=2)

    # immutable archive copies + log + a machine-readable summary
    manifest_df.to_csv(run_dir / "manifest_seed.csv", index=False)
    shutil.copy(scfg.SPLITS_JSON, run_dir / "splits_seed_loso.json")
    summary = {
        "n_subjects": scfg.N_SUBJECTS, "n_sessions_per_subject": scfg.N_SESSIONS,
        "n_trials_per_session": scfg.N_TRIALS_PER_SESSION, "n_channels": scfg.N_CHANNELS,
        "trial_labels": scfg.TRIAL_LABELS, "label_names": scfg.LABEL_NAMES,
        "per_session_class_counts": class_counts,
        "protocol": "LOSO (leave-one-subject-out), 15 folds, test=1 subject each",
        "split_unit": "subject (all 3 sessions + all 15 trials + all future segments follow the subject)",
        "leaks_prevented": ["session-leak (3 sessions/subject kept together)",
                            "trial/segment-leak (trial-level labels; segments inherit subject fold)"],
        "zero_leak_assertions": assert_msg,
        "manifest_csv": str(scfg.MANIFEST_CSV), "splits_json": str(scfg.SPLITS_JSON),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(run_dir / "task1.log", "w") as f:
        f.write("\n".join(log) + "\n")

    print("\n".join(log))
    print(f"\n=== SEED Task 1 done ===")
    print(f"manifest: {scfg.MANIFEST_CSV} ({len(manifest_df)} subjects)")
    print(f"splits:   {scfg.SPLITS_JSON} ({len(splits)} LOSO folds)")
    print(f"{assert_msg}")
    print(f"archived: {run_dir}")
    return summary


if __name__ == "__main__":
    main()
