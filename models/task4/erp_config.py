"""
Task 4 ERP Stage 1 config: preprocessing + cue-locked epoching + artifact
flagging (no rejection, no modeling). Reuses task2's main_B filter/bad-
channel functions UNCHANGED where the operation is identical; the average
reference differs deliberately (excludes bad channels from the averaging
step -- task2's resting-state avg_reference does not, since that pipeline
is frozen and not being touched here).

MNE (base conda env, /root/miniconda3/bin/python) is required to read the
EGI .raw files -- NOT installed in the locked mamba3 env, which this script
never touches.
"""
import sys
from pathlib import Path

TASK1_DIR = Path("/root/newcode/task1")
TASK2_DIR = Path("/root/newcode/task2")
TASK4_DIR = Path("/root/newcode/task4")
ERP_RAW_DIR = Path("/root/MODMA/EEG_128channels_ERP_lanzhou_2015")

sys.path.insert(0, str(TASK2_DIR))
import config as task2_cfg  # noqa: E402  (main_B constants + filter/bad-channel functions' home)

EPOCHS_CACHE = TASK4_DIR / "epochs_cache"
RESULTS_DIR = TASK4_DIR / "results"
RUNS_DIR = TASK4_DIR / "runs"
for d in (EPOCHS_CACHE, RESULTS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- main_B lineage (identical constants to task2/resting-state) ----
FS = task2_cfg.FS  # 250
HPF = task2_cfg.HPF  # 0.5
LPF = task2_cfg.LPF  # 45.0
FILTER_ORDER = task2_cfg.FILTER_ORDER  # 4
NOTCH_FREQ = task2_cfg.NOTCH_FREQ  # 50.0
NOTCH_Q = task2_cfg.NOTCH_Q  # 30.0
N_DATA_CH = task2_cfg.N_DATA_CH  # 128 -- E1..E128, modeling channels
CZ_CH_NAME = "E129"  # the 129th EGI channel == montage's "Cz" position, dropped (reference row)

# bad-channel detection: SAME thresholds as task2, applied on the filtered
# (pre-reference) continuous ERP signal instead of post-reference -- see
# STATUS.md for why the order differs (avg reference here must exclude bad
# channels, so detection has to happen before referencing).
BAD_CH_LOGVAR_MAD_THRESH = task2_cfg.BAD_CH_LOGVAR_MAD_THRESH  # 5.0
BAD_CH_MIN_MEAN_ABS_CORR = task2_cfg.BAD_CH_MIN_MEAN_ABS_CORR  # 0.05

# ---- cue-locked epoching ----
EPOCH_TMIN_SEC = -0.200
EPOCH_TMAX_SEC = 0.600
PRE_SAMPLES = int(round(0.200 * FS))   # 50
POST_SAMPLES = int(round(0.600 * FS))  # 150
EPOCH_LEN = PRE_SAMPLES + POST_SAMPLES  # 200
BASELINE_SAMPLES = PRE_SAMPLES  # first 50 samples of the epoch = the -200..0ms baseline window

CUE_EVENT_CODES = {"f": "fcue", "h": "hcue", "s": "scue"}  # condition letter -> event code

# ---- artifact flagging (data-driven, MAD-based -- same style/k as bad-channel
# detection, applied on POOLED per-trial peak-to-peak metrics across ALL
# subjects so the threshold itself can't be contaminated by group membership) ----
ARTIFACT_MAD_K = 5.0
N_FRONTAL_CHANNELS = 10  # most-anterior montage channels, eye-movement proxy (no dedicated EOG channel exists)

SPLITS_JSON = TASK1_DIR / "splits_128rest.json"
MANIFEST_128REST = TASK1_DIR / "manifest_128rest.csv"
