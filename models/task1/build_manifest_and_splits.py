"""
Task 1: subject manifest + subject-independent stratified K-fold splits.
Pure metadata — does not read/parse any signal payloads.

Locked decisions (see conversation spec, do not silently change):
  - two independent subsets: 128rest (53 subjects, primary) and 3ch (55 subjects, replication)
  - subject_id is derived from directory/file names (8-digit, zero-padded), never fuzzy-matched
    against the xlsx "subject id" column
  - label: MDD -> 1, HC -> 0, taken from each subset's own xlsx `type` column
  - split = subject-independent stratified K=5 fold x seeds [42, 1337, 2024, 7, 91]
    stratification key = label only; sex/age are reported per fold but NOT stratified on
  - sample unit = one subject (53 / 55), windows are introduced in Task 2 and must inherit
    the subject's fold assignment
"""
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
import yaml
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/root/MODMA")
OUT = Path("/root/newcode/task1")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1337, 2024, 7, 91]
N_FOLDS = 5

SUBJECT_ID_RE = re.compile(r"^(\d{8})")

# files confirmed (Task 0 full sweep) to lack an Impedances_0 key
MISSING_IMPEDANCE_128REST = {"02030009", "02030014", "02030020"}
SHORT_RECORD_3CH = {"02020018"}


def read_subjects_xlsx(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else None for h in rows[0]]
    col = {name: i for i, name in enumerate(header) if name}

    def get(row, name, default=None):
        i = col.get(name)
        if i is None or i >= len(row):
            return default
        v = row[i]
        return default if v is None else v

    out = {}
    for row in rows[1:]:
        sid_raw = get(row, "subject id")
        if sid_raw is None:
            continue
        sid = str(sid_raw).strip()
        if not re.fullmatch(r"\d{8}", sid):
            continue  # never fuzzy-match malformed ids (e.g. missing leading zero)
        out[sid] = {
            "type": get(row, "type"),
            "age": get(row, "age", np.nan),
            "gender": get(row, "gender"),
            "education_years": get(row, "education（years）", np.nan),
            "phq9": get(row, "PHQ-9", np.nan),
            "gad7": get(row, "GAD-7", np.nan),
            "ctq_sf": get(row, "CTQ-SF", np.nan),
            "les": get(row, "LES", np.nan),
            "ssrs": get(row, "SSRS", np.nan),
            "psqi": get(row, "PSQI", np.nan),
        }
    return out


def scan_subject_ids(dir_path, glob_pattern):
    files = {}
    for f in sorted(dir_path.glob(glob_pattern)):
        m = SUBJECT_ID_RE.match(f.name)
        if not m:
            raise ValueError(f"cannot extract 8-digit subject id from filename: {f.name}")
        sid = m.group(1)
        if sid in files:
            raise ValueError(f"duplicate subject id {sid} from files {files[sid].name} and {f.name}")
        files[sid] = f
    return files


def build_manifest_128rest():
    d = ROOT / "EEG_128channels_resting_lanzhou_2015"
    files = scan_subject_ids(d, "*.mat")
    meta = read_subjects_xlsx(d / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")

    missing_meta = sorted(set(files) - set(meta))
    if missing_meta:
        raise ValueError(f"128rest: subjects present as files but missing xlsx metadata: {missing_meta}")

    rows = []
    for sid in sorted(files):
        m = meta[sid]
        typ = m["type"]
        label = {"MDD": 1, "HC": 0}.get(typ)
        if label is None:
            raise ValueError(f"128rest subject {sid}: unrecognized type {typ!r}")
        rows.append({
            "subject_id": sid,
            "label": label,
            "type": typ,
            "age": m["age"],
            "gender": m["gender"],
            "education_years": m["education_years"],
            "phq9": m["phq9"],
            "gad7": m["gad7"],
            "ctq_sf": m["ctq_sf"],
            "les": m["les"],
            "ssrs": m["ssrs"],
            "psqi": m["psqi"],
            "data_path": str(files[sid]),
            "n_data_channels": 128,
            "ref_channel_idx": 128,
            "has_impedance": sid not in MISSING_IMPEDANCE_128REST,
            "bad_channels": np.nan,  # placeholder, filled in Task 2 after reading signals
            "known_issue": "missing_impedance_qc" if sid in MISSING_IMPEDANCE_128REST else "",
        })
    return pd.DataFrame(rows)


def build_manifest_3ch():
    d = ROOT / "EEG_3channels_resting_lanzhou_2015"
    files = scan_subject_ids(d, "*_still.txt")
    meta = read_subjects_xlsx(d / "subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx")

    missing_meta = sorted(set(files) - set(meta))
    if missing_meta:
        raise ValueError(f"3ch: subjects present as files but missing xlsx metadata: {missing_meta}")

    rows = []
    for sid in sorted(files):
        m = meta[sid]
        typ = m["type"]
        label = {"MDD": 1, "HC": 0}.get(typ)
        if label is None:
            raise ValueError(f"3ch subject {sid}: unrecognized type {typ!r}")
        rows.append({
            "subject_id": sid,
            "label": label,
            "type": typ,
            "age": m["age"],
            "gender": m["gender"],
            "education_years": m["education_years"],
            "phq9": m["phq9"],
            "gad7": m["gad7"],
            "ctq_sf": m["ctq_sf"],
            "les": m["les"],
            "ssrs": m["ssrs"],
            "psqi": m["psqi"],
            "data_path": str(files[sid]),
            "n_data_channels": 3,
            "ref_channel_idx": np.nan,   # not an EGI net recording, no online reference row
            "has_impedance": np.nan,     # no impedance metadata for this subset
            "bad_channels": np.nan,      # placeholder, filled in Task 2
            "known_issue": "short_record_18204_rows" if sid in SHORT_RECORD_3CH else "",
        })
    return pd.DataFrame(rows)


def make_splits(df):
    """Subject-independent stratified K-fold, stratified on label only."""
    subject_ids = df["subject_id"].to_numpy()
    labels = df["label"].to_numpy()

    splits = {}
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        folds = {}
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(subject_ids, labels)):
            folds[str(fold_idx)] = {
                "train": sorted(subject_ids[train_idx].tolist()),
                "test": sorted(subject_ids[test_idx].tolist()),
            }
        splits[str(seed)] = folds
    return splits


def assert_no_leakage(splits, all_ids):
    from collections import Counter
    all_ids = set(all_ids)
    for seed, folds in splits.items():
        test_union = []
        for fold, d in folds.items():
            tr, te = set(d["train"]), set(d["test"])
            assert tr.isdisjoint(te), f"[s{seed} f{fold}] train/test 被试重叠"
            assert tr | te == all_ids, f"[s{seed} f{fold}] 覆盖不全/多余"
            test_union += d["test"]
        c = Counter(test_union)
        assert set(c) == all_ids and max(c.values()) == 1, f"[s{seed}] 测试折重叠/遗漏"
    print("PASS: 零泄漏，每 seed 下 5 折构成全体被试的一个划分")


def fmt_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return f"{x.mean():.2f}" if len(x) else "nan"


def write_split_report(name, df, splits, fh):
    idx = df.set_index("subject_id")
    fh.write(f"\n{'='*70}\n{name}\n{'='*70}\n")
    for seed, folds in splits.items():
        for fold, d in folds.items():
            tr_ids, te_ids = d["train"], d["test"]
            tr = idx.loc[tr_ids]
            te = idx.loc[te_ids]
            tr_mdd, tr_hc = int((tr["label"] == 1).sum()), int((tr["label"] == 0).sum())
            te_mdd, te_hc = int((te["label"] == 1).sum()), int((te["label"] == 0).sum())
            tr_sex = tr["gender"].value_counts().to_dict()
            te_sex = te["gender"].value_counts().to_dict()
            fh.write(
                f"seed={seed:>5} fold={fold}: "
                f"train n={len(tr_ids):>3} (MDD={tr_mdd},HC={tr_hc}) "
                f"age_mean={fmt_mean(tr['age'])} sex={tr_sex} | "
                f"test n={len(te_ids):>3} (MDD={te_mdd},HC={te_hc}) "
                f"age_mean={fmt_mean(te['age'])} sex={te_sex}\n"
            )


def main():
    df_128 = build_manifest_128rest()
    df_3ch = build_manifest_3ch()

    df_128.to_csv(OUT / "manifest_128rest.csv", index=False)
    df_3ch.to_csv(OUT / "manifest_3ch.csv", index=False)

    print("manifest_128rest.csv:", len(df_128), "rows, MDD =", int((df_128.label == 1).sum()),
          "HC =", int((df_128.label == 0).sum()))
    print("manifest_3ch.csv:", len(df_3ch), "rows, MDD =", int((df_3ch.label == 1).sum()),
          "HC =", int((df_3ch.label == 0).sum()))

    assert len(df_128) == 53
    assert len(df_3ch) == 55
    assert (df_128.n_data_channels == 128).all()
    assert set(df_128.loc[~df_128.has_impedance, "subject_id"]) == MISSING_IMPEDANCE_128REST
    assert df_3ch.loc[df_3ch.subject_id == "02020018", "known_issue"].iloc[0] == "short_record_18204_rows"

    splits_128 = make_splits(df_128)
    splits_3ch = make_splits(df_3ch)

    with open(OUT / "splits_128rest.json", "w") as f:
        json.dump(splits_128, f, indent=2, ensure_ascii=False)
    with open(OUT / "splits_3ch.json", "w") as f:
        json.dump(splits_3ch, f, indent=2, ensure_ascii=False)

    assert_no_leakage(splits_128, df_128["subject_id"].tolist())
    assert_no_leakage(splits_3ch, df_3ch["subject_id"].tolist())

    with open(OUT / "split_report.txt", "w") as fh:
        write_split_report("manifest_128rest", df_128, splits_128, fh)
        write_split_report("manifest_3ch", df_3ch, splits_3ch, fh)
    print("split_report.txt written")

    preproc_config = {
        "main_B": {"hpf": 0.5, "lpf": 45, "filter": "zero_phase_iir", "reref": "REST", "asr": False, "interp_bad": False},
        "ctrl_A": {"hpf": 1.0, "lpf": 40, "filter": "zero_phase_iir", "reref": "REST", "asr": True, "interp_bad": True},
        "fs": 250,
        "ref_channel_idx": 128,
        "inject_freq_max": 40,
        "theta_to_hz": 39.79,
    }
    with open(OUT / "preproc_config.yaml", "w") as f:
        yaml.dump(preproc_config, f, default_flow_style=False, sort_keys=False)
    print("preproc_config.yaml written")

    # scale-field silent-drop check: every xlsx scale column must survive into the manifest
    for df, name in [(df_128, "128rest"), (df_3ch, "3ch")]:
        for col in ["phq9", "gad7", "ctq_sf", "les", "ssrs", "psqi"]:
            assert col in df.columns, f"{name}: missing scale column {col}"
            n_nan = df[col].isna().sum()
            print(f"{name}.{col}: {df[col].notna().sum()} non-null, {n_nan} NaN")


if __name__ == "__main__":
    main()
