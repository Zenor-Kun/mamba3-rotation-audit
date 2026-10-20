"""
SEED Mamba Stage 1 config: rebuild V1 (V1_full only, no theta-switch ablation)
+ band localization. Runs in the mamba3 conda env (torch + mamba_ssm); NEVER
modifies that env. Isolated under seed/; MODMA untouched.

Front-end (lessons from MODMA): phase-preserving complex STFT, NO giant flatten
FC. Per (channel, frame) feature = freq_bins x (real, imag); channels folded
into the batch dim (weight-shared small projection), frames are the Mamba
sequence axis. seq_len (frames) >= 100 so the SSM is actually exercised.

Mamba-3 activated near the official recommended config (lessons from MODMA
where SISO/fp32/short-seq never activated): is_mimo=True, mimo_rank=4,
dtype=bfloat16, d_state=128, headdim=64, d_model moderate.
"""
import sys
from pathlib import Path

import numpy as np

SEED_CODE_DIR = Path("/root/newcode/seed")
TASK3_DIR = SEED_CODE_DIR / "task3"
sys.path.insert(0, str(SEED_CODE_DIR))
sys.path.insert(0, str(SEED_CODE_DIR / "task2"))
import seed_config as scfg          # noqa: E402
import seed2_config as s2cfg        # noqa: E402  (FS, bands, labels, preprocessing constants)

SEGMENTS_CACHE = TASK3_DIR / "segments_cache"
RESULTS_DIR = TASK3_DIR / "results"
RUNS_DIR = scfg.RUNS_DIR
for d in (SEGMENTS_CACHE, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SPLITS_JSON = scfg.SPLITS_JSON     # read-only, frozen LOSO
MANIFEST_CSV = scfg.MANIFEST_CSV

# ---- signal / segmentation (same as Task 2) ----
FS = s2cfg.FS                       # 200
N_CH = s2cfg.N_CH                   # 62
WIN_LEN = s2cfg.WIN_LEN            # 800 (4 s)
STEP = s2cfg.STEP                  # 400 (50% overlap)

# ---- complex STFT front-end ----
STFT_N_FFT = 128
STFT_HOP = 6
STFT_WINDOW = "hann"
STFT_FRAMES_RAW = (WIN_LEN - STFT_N_FFT) // STFT_HOP + 1   # 113
# The MIMO Mamba-3 tilelang kernel requires seq_len divisible by chunk_size, and
# on a 3090 only chunk_size=8 fits shared memory for this MIMO config (see below).
# 113 is prime, so the last STFT frame is dropped -> 112 = 14*8 frames fed to the
# SSM. This is a front-end framing choice (the final 4 s-window tail frame), NOT a
# stride/pool inside the SSM; effective frame rate + theta->Hz mapping unchanged.
N_FRAMES = 112
SEQ_LEN = N_FRAMES                                    # 112 >= 100, divisible by chunk_size 8
EFFECTIVE_FRAME_RATE_HZ = FS / STFT_HOP               # 200/6 = 33.333 Hz
# theta->Hz for this front-end: f = phi * (EFFECTIVE_FRAME_RATE_HZ / (2*pi)) = phi * 5.3052
THETA_TO_HZ_CONST = EFFECTIVE_FRAME_RATE_HZ / (2 * np.pi)

# STFT freq bins (rfft, n_fft=128, fs=200): freq[k] = k * FS / N_FFT
_ALL_FREQS = np.arange(STFT_N_FFT // 2 + 1) * (FS / STFT_N_FFT)   # 65 bins, 0..100 Hz, 1.5625 spacing
BANDS = s2cfg.BANDS                 # theta/alpha/beta/gamma edges
BAND_ORDER = s2cfg.BAND_ORDER

# keep only bins that fall in one of the 4 bands (theta..gamma); record band per bin
def _band_of(f):
    for b, (lo, hi) in BANDS.items():
        if lo <= f < hi:
            return b
    return None

KEPT_BIN_IDX = np.array([k for k, f in enumerate(_ALL_FREQS) if _band_of(f) is not None])  # k in 3..28
KEPT_FREQS = _ALL_FREQS[KEPT_BIN_IDX]
KEPT_BIN_BAND = [_band_of(f) for f in KEPT_FREQS]
N_FREQ_BINS = len(KEPT_BIN_IDX)                    # 26
N_FREQ_X2 = N_FREQ_BINS * 2                        # 52 = per-(channel,frame) feature dim
# indices (within the kept array, 0..N_FREQ_BINS-1) of gamma bins -> zeroed in the drop-gamma condition
GAMMA_KEPT_IDX = np.array([j for j, b in enumerate(KEPT_BIN_BAND) if b == "gamma"])
NONGAMMA_KEPT_IDX = np.array([j for j, b in enumerate(KEPT_BIN_BAND) if b != "gamma"])

# ---- labels (3-class) ----
LABEL_TO_IDX = s2cfg.LABEL_TO_IDX   # {-1:0, 0:1, 1:2}
IDX_TO_NAME = s2cfg.IDX_TO_NAME
N_CLASSES = 3
CHANCE = 1.0 / 3.0
TRIAL_LABELS = s2cfg.TRIAL_LABELS

# ---- Mamba-3 (V1_full = stock Mamba3 with full theta; MIMO/bf16) ----
# Target ("official recommended"): is_mimo=True, mimo_rank=4, d_state=128,
# headdim=64, bf16. That exact config's BACKWARD kernel needs ~122 KB dynamic
# shared memory, over the RTX 3090's ~100 KB opt-in limit, so it cannot train
# on this hardware. Minimal hardware-forced deviation (kept "close to" spec):
# headdim 64->32, everything else exact (is_mimo=True, mimo_rank=4, d_state=128).
# d_state (the SSM state dim -- the quantity the complex-state/rhythm thesis is
# about) is kept at the full 128; only the head decomposition is halved
# (nheads = d_model*expand/headdim = 8). chunk_size=8 (min allowed; 16 overflows
# smem). bf16 via autocast (casting the module to bf16 breaks the kernel's
# float32-bias expectation). All verified empirically on the 3090.
D_MODEL = 128
D_STATE = 128
HEADDIM = 32
EXPAND = 2
MIMO_RANK = 4
IS_MIMO = True
CHUNK_SIZE = 8
N_LAYERS = 2
DROPOUT = 0.3

# ---- training ----
BATCH_SEGMENTS = 32                 # segments/step; x62 channels = ~1984 sequences/step
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
INNER_VAL_SUBJECTS = 2              # carved from the 14 train subjects (subject-level, no leak)
LR = 3e-4
WEIGHT_DECAY = 5e-2
DTYPE = "bfloat16"
# compute-driven subsample: cap training segments per trial (evenly spaced).
# EVAL/inner-val use ALL segments (aggregation quality unaffected). This bounds
# the channel-folded step count; documented as a compute choice, not tuning.
TRAIN_SEG_PER_TRIAL_CAP = 24

N_SUBJECTS = scfg.N_SUBJECTS


def assert_no_stride_or_pool(seq_len_in, seq_len_out, stage_name=""):
    assert seq_len_in == SEQ_LEN, f"{stage_name}: seq_len {seq_len_in} != SEQ_LEN {SEQ_LEN}"
    assert seq_len_out == seq_len_in, (
        f"{stage_name}: seq_len {seq_len_in}->{seq_len_out}; no stride/pool beyond STFT hop allowed in the SSM")
