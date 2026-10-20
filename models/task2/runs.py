"""
Immutable run archive. Every training/eval run gets its own timestamped
directory under runs/ (never deleted or overwritten). results/*.json stays as
a "latest" convenience pointer, but runs/ is the record of truth going forward.
"""
import json
import time

import config as cfg

RUNS_DIR = cfg.TASK2_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def new_run_dir(tag):
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = RUNS_DIR / f"{ts}_{tag}"
    assert not d.exists(), f"run dir collision, refusing to reuse: {d}"
    d.mkdir(parents=True)
    return d


def config_snapshot():
    return {k: v for k, v in vars(cfg).items()
            if k.isupper() and isinstance(v, (int, float, str, bool, type(None)))}


def write_snapshot(run_dir):
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(config_snapshot(), f, indent=2, default=str)
