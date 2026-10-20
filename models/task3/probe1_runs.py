"""Immutable run archive for Task 3 (same convention as task2/runs.py):
every run gets its own timestamped dir under runs/, never deleted or
overwritten. results/*.json is a "latest" pointer only."""
import json
import time

import probe1_config as pcfg

RUNS_DIR = pcfg.RUNS_DIR


def new_run_dir(tag):
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = RUNS_DIR / f"{ts}_{tag}"
    assert not d.exists(), f"run dir collision, refusing to reuse: {d}"
    d.mkdir(parents=True)
    return d


def config_snapshot():
    return {k: v for k, v in vars(pcfg).items()
            if k.isupper() and isinstance(v, (int, float, str, bool, list, type(None)))}


def write_snapshot(run_dir):
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(config_snapshot(), f, indent=2, default=str)
