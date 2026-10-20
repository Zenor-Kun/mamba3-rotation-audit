"""
Task 3 / Probe 1 (mechanism separability) config.

Everything data/protocol-related is imported unchanged from Task 2 (config,
windows.py) -- same splits_128rest.json, same main_B+notch preprocessing, same
4s/1000-sample windows, same train-fold z-score, same window-level train /
subject-level aggregation. Probe 1 changes exactly one thing: how the Mamba-3
rotation angle (theta) is computed, across 4 variants. Everything else
(layers, d_model, lr, dropout, epochs, early stopping) must be IDENTICAL
across variants -- that's what makes the ablation fair.
"""
import sys
from pathlib import Path

TASK2_DIR = Path("/root/newcode/task2")
TASK3_DIR = Path("/root/newcode/task3")
sys.path.insert(0, str(TASK2_DIR))  # exposes task2's `config` and `windows` modules unchanged

import config as task2_cfg  # noqa: E402  (task2's config.py -- the shared protocol)

RESULTS_DIR = TASK3_DIR / "results"
RUNS_DIR = TASK3_DIR / "runs"
FIGS_DIR = TASK3_DIR / "figs"
for d in (RESULTS_DIR, RUNS_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- shared architecture, IDENTICAL across all 4 variants (this is what "fair
# ablation" means -- only the theta switch differs) ----
D_MODEL = task2_cfg.D_MODEL          # 128
D_STATE = task2_cfg.D_STATE          # 128
EXPAND = task2_cfg.EXPAND            # 2
HEADDIM = task2_cfg.HEADDIM          # 64
N_LAYERS = task2_cfg.N_LAYERS        # 2
CHUNK_SIZE = task2_cfg.CHUNK_SIZE    # 64
DROPOUT = task2_cfg.DROPOUT          # 0.3

# ---- shared training recipe, IDENTICAL across all 4 variants ----
BATCH_SIZE = task2_cfg.BATCH_SIZE
EPOCHS = task2_cfg.EPOCHS
EARLY_STOP_PATIENCE = task2_cfg.EARLY_STOP_PATIENCE
INNER_VAL_FRAC = task2_cfg.INNER_VAL_FRAC
LR = task2_cfg.LR
WEIGHT_DECAY = task2_cfg.WEIGHT_DECAY
DEVICE = task2_cfg.DEVICE

# ---- V4 real-only counterpart (Mamba2), width chosen so 2-layer total param
# count matches V1's 277,009 within +/-5% (see param_match.py for the search) ----
V4_D_STATE = 128
V4_HEADDIM = 8
V4_EXPAND = 2
V4_CHUNK_SIZE = 64

# ---- variants ----
VARIANTS = ["V1_full", "V2_theta_zero", "V3_theta_frozen_random", "V4_real_only"]

# ---- seeds: seed 42 first for a directional read; only extend to the full
# 5-seed sweep after seed42 clears the pre-registered decision line ----
SEED_FIRST = 42
SEEDS_FULL = [42, 1337, 2024, 7, 91]

N_FOLDS = task2_cfg.N_FOLDS
