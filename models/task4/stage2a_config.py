"""Task 4 Stage 2A config: ERP signal confirmation + EOG control + ERP-band
autopsy, LR/linear only. No Mamba, no full-dim features."""
import json
from pathlib import Path

import erp_config as ecfg

FS = ecfg.FS  # 250
PRE_SAMPLES = ecfg.PRE_SAMPLES  # 50 (t=0 at sample index 50)
EPOCH_LEN = ecfg.EPOCH_LEN  # 200

# ---- time windows (ms), converted to sample-index [lo, hi) ----
def _idx(t_ms):
    return int(round((t_ms / 1000.0) * FS)) + PRE_SAMPLES

TIME_WINDOWS_MS = [(0, 100), (100, 200), (200, 350), (350, 596)]
TIME_WINDOWS = [(f"w{i+1}_{lo}_{hi}", _idx(lo), _idx(hi)) for i, (lo, hi) in enumerate(TIME_WINDOWS_MS)]
# -> [("w1_0_100",50,75), ("w2_100_200",75,100), ("w3_200_350",100,138), ("w4_350_596",138,200)]

CONDITIONS = ["f", "h", "s"]
CONTRASTS = [("f_minus_h", "f", "h"), ("s_minus_h", "s", "h"), ("f_minus_s", "f", "s")]

MIN_TRIAL_THRESHOLD = 20  # per (subject,condition), under the mad_k3 flag set specifically

BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
}

SIGNAL_THRESHOLD = 0.55  # balanced_acc above which Protection 3 (band autopsy) is triggered

SEEDS_FULL = [42, 1337, 2024, 7, 91]
N_FOLDS = 5

TASK4_DIR = ecfg.TASK4_DIR
ERP_CACHE = TASK4_DIR / "stage2a_erp_cache"
BAND_CACHE = TASK4_DIR / "stage2a_band_cache"
RESULTS_DIR = ecfg.RESULTS_DIR
RUNS_DIR = ecfg.RUNS_DIR
for d in (ERP_CACHE, BAND_CACHE):
    d.mkdir(parents=True, exist_ok=True)

EXCLUDED_STAGE1C = ["02010030", "02010034", "02030009", "02030020"]

FLAG_SET_ALL_TRIALS = "all_trials"
FLAG_SETS = ["mad_k3", "abs_100uv", "ptp_100uv", FLAG_SET_ALL_TRIALS]


def load_splits():
    with open(ecfg.SPLITS_JSON) as f:
        return json.load(f)
