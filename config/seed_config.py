"""
SEED dataset config -- a NEW, fully isolated battlefield from the MODMA work.

Isolation rules (per user directive):
  - All SEED code + products live under /root/newcode/seed/ ONLY. MODMA's
    task1..task4 (and their runs/, splits_128rest.json) are frozen archived
    assets -- never touched or overwritten from here.
  - SEED has its own runs/ (/root/newcode/seed/runs/) and its own frozen
    split file (splits_seed_loso.json). No cross-contamination with MODMA.

Dataset facts (confirmed by structure exploration, task0_explore/):
  - Source: /root/SEED/SEED/. Use Preprocessed_EEG/ (phase-preserving time
    series). ExtractedFeatures/ (DE features) is NOT used -- we need raw
    phase-bearing signal for the complex-state / rhythm story.
  - 15 subjects x 3 sessions = 45 .mat files, plus label.mat.
  - Each session .mat has exactly 15 trial variables named "<initials>_eeg1"
    .. "<initials>_eeg15" (initials prefix varies per subject); 62 channels.
  - Trial-level labels (SAME 15-length array for every session), from
    Preprocessed_EEG/readme.txt and label.mat:
        [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    with 1=positive, 0=neutral, -1=negative -> 5 trials per class per
    session (5 pos / 5 neu / 5 neg).
"""
from pathlib import Path

SEED_ROOT = Path("/root/SEED/SEED")
PREPROC_DIR = SEED_ROOT / "Preprocessed_EEG"
GENDER_FILE = SEED_ROOT / "subject-id-gender-seed.txt"
LABEL_MAT = PREPROC_DIR / "label.mat"

SEED_CODE_DIR = Path("/root/newcode/seed")
RUNS_DIR = SEED_CODE_DIR / "runs"
TASK1_DIR = SEED_CODE_DIR / "task1"
SPLITS_JSON = SEED_CODE_DIR / "splits_seed_loso.json"
MANIFEST_CSV = SEED_CODE_DIR / "manifest_seed.csv"
for d in (RUNS_DIR, TASK1_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_SUBJECTS = 15
N_SESSIONS = 3
N_TRIALS_PER_SESSION = 15
N_CHANNELS = 62

# trial-level labels, identical for all sessions (SEED standard)
TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LABEL_NAMES = {1: "positive", 0: "neutral", -1: "negative"}
