"""
SEED Task 2 config: preprocessing + segmentation + LR signal-confirmation.
Isolated under seed/; reuses task2's UNCHANGED filter/bad-channel functions
where identical, and MODMA's split/archive discipline. Never touches MODMA.

Preprocessing note: SEED Preprocessed_EEG is ALREADY bandpassed 0-75 Hz by
the dataset providers -- we do NOT re-bandpass (that would double-filter).
We add a 50 Hz notch (China mains), then average-reference EXCLUDING flagged
bad channels (same convention as task4 ERP), then data-driven bad-channel
flagging (flag only, never dropped/interpolated).
"""
import sys
from pathlib import Path

SEED_CODE_DIR = Path("/root/newcode/seed")
TASK2_DIR = SEED_CODE_DIR / "task2"
MODMA_TASK2_DIR = Path("/root/newcode/task2")

sys.path.insert(0, str(SEED_CODE_DIR))
import seed_config as scfg  # noqa: E402

FEATURES_CACHE = TASK2_DIR / "features_cache"
RESULTS_DIR = TASK2_DIR / "results"
RUNS_DIR = scfg.RUNS_DIR
for d in (FEATURES_CACHE, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

PREPROC_DIR = scfg.PREPROC_DIR
MANIFEST_CSV = scfg.MANIFEST_CSV
SPLITS_JSON = scfg.SPLITS_JSON      # read-only, frozen

# ---- signal params ----
FS = 200                 # SEED Preprocessed_EEG downsampled to 200 Hz (confirmed: 185-265s clips)
N_CH = scfg.N_CHANNELS   # 62
BANDPASS_NOTE = "already 0-75 Hz by dataset provider; not re-bandpassed here"

# mains notch (China grid = 50 Hz), same params as MODMA task2
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0

# bad-channel detection: SAME thresholds/function as MODMA task2, flag-only
BAD_CH_LOGVAR_MAD_THRESH = 5.0
BAD_CH_MIN_MEAN_ABS_CORR = 0.05

# ---- segmentation ----
WIN_SEC = 4.0
WIN_LEN = int(WIN_SEC * FS)     # 800
STEP = WIN_LEN // 2             # 400, 50% overlap
assert WIN_LEN == 800 and STEP == 400

# ---- band definitions (theta/alpha/beta/gamma; gamma capped at 45 to sit
# clear of the 50 Hz notch even though signal extends to 75 Hz) ----
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_ORDER = ["theta", "alpha", "beta", "gamma"]

# ---- labels (SEED standard 3-class), from seed_config ----
TRIAL_LABELS = scfg.TRIAL_LABELS            # [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}          # negative/neutral/positive -> 0/1/2 for the classifier
IDX_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}
N_CLASSES = 3
CHANCE = 1.0 / 3.0

# ---- LR ----
LR_MAX_ITER = 2000
LR_C = 1.0
