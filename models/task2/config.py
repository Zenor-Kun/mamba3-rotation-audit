"""
Task 2 locked config. Do not silently change these numbers — they were
computed from Task 0 measured stats (128 ch, fs=250, T in [75089, 83876])
and from the conversation spec. main_B is the only pipeline Task 2 runs;
ctrl_A is interface-only (never executed here).
"""
from pathlib import Path

TASK1_DIR = Path("/root/newcode/task1")
TASK2_DIR = Path("/root/newcode/task2")
MANIFEST_IN = TASK1_DIR / "manifest_128rest.csv"
SPLITS_JSON = TASK1_DIR / "splits_128rest.json"

PREPROC_CACHE = TASK2_DIR / "preproc_cache"
WINDOWS_CACHE = TASK2_DIR / "windows_cache"
RESULTS_DIR = TASK2_DIR / "results"
FIGS_DIR = TASK2_DIR / "figs"
MANIFEST_OUT = TASK2_DIR / "manifest_128rest_task2.csv"

FS = 250
N_DATA_CH = 128          # channels 0..127; row 128 (129th) is Cz, dropped from the data tensor
CZ_ROW_IDX = 128

# ---- main_B (the only pipeline Task 2 activates) ----
HPF = 0.5
LPF = 45.0
FILTER_ORDER = 4         # butterworth, per-side order for SOS bandpass (zero-phase via sosfiltfilt)

# Re-reference method actually run in main_B. NOTE (approved deviation, 2026-07-07):
# spec calls for REST (Yao 2001), which needs a lead-field matrix (head model + 128-ch
# 3D montage). Neither is available in this environment and installing MNE was ruled
# out to avoid touching the configured torch/mamba env, so this substitutes average
# reference across the 128 data channels. Swap this out for true REST once a montage
# is available; nothing downstream depends on the string value itself.
REREF_METHOD = "avg_reference"
REREF_DEVIATION_NOTE = (
    "substitutes REST (Yao 2001, lead-field-based); no electrode montage/head model "
    "available in this environment and installing MNE was ruled out -- see STATUS.md"
)

# mains-line notch (China grid = 50 Hz). The 0.5-45 Hz bandpass alone leaves a residual
# 50 Hz peak (only 5 Hz past the LPF cutoff -- a 4th-order Butterworth doesn't fully
# reject it there). Applied zero-phase, same stage as the bandpass, before windowing.
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0            # ~1.67 Hz -3dB bandwidth

# bad-channel detection thresholds (flag only, never interpolate in main_B)
BAD_CH_LOGVAR_MAD_THRESH = 5.0
BAD_CH_MIN_MEAN_ABS_CORR = 0.05

# ---- windowing ----
WIN_SEC = 4.0
WIN_LEN = int(WIN_SEC * FS)      # 1000
STEP = WIN_LEN // 2              # 500, 50% overlap
assert WIN_LEN == 1000 and STEP == 500

# ctrl window (interface only, not run in Task 2)
CTRL_WIN_SEC = 2.0
CTRL_WIN_LEN = int(CTRL_WIN_SEC * FS)   # 500
CTRL_STEP = CTRL_WIN_LEN // 2            # 250

# ---- SSM sequence-length lock: no stride / no pooling before the SSM ----
SEQ_LEN = WIN_LEN                # 1000, fed straight into Mamba-3, effective fs stays 250
EFFECTIVE_FS = FS

def assert_no_stride_or_pool(seq_len_in, seq_len_out, stage_name=""):
    assert seq_len_in == SEQ_LEN, f"{stage_name}: input seq_len {seq_len_in} != locked SEQ_LEN {SEQ_LEN}"
    assert seq_len_out == seq_len_in, (
        f"{stage_name}: seq_len changed {seq_len_in} -> {seq_len_out}; "
        f"stride/pooling before the SSM is forbidden by spec"
    )

# ---- splits ----
SEED = 42                # Task 2 only runs this one seed (V1 link-check, not a sweep)
N_FOLDS = 5

# ---- V1 model ----
D_MODEL = 128
D_STATE = 128
EXPAND = 2
HEADDIM = 64
N_LAYERS = 2              # 2-4 allowed by spec; kept small, only ~6300 train windows / 42 subjects
IS_MIMO = False           # SISO first, per spec
CHUNK_SIZE = 64
DROPOUT = 0.3             # window-level training badly overfits subject identity with this few subjects

# ---- training ----
BATCH_SIZE = 32
EPOCHS = 40               # upper bound; actual stop point picked by inner early-stopping (train-only)
EARLY_STOP_PATIENCE = 6
INNER_VAL_FRAC = 0.15     # carved out of TRAIN subjects only, never touches the held-out test fold
LR = 3e-4
WEIGHT_DECAY = 5e-2
DEVICE = "cuda:0"

for d in (PREPROC_CACHE, WINDOWS_CACHE, RESULTS_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)
