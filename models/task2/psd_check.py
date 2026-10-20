"""
Task 2 verification: PSD before vs after main_B preprocessing, for one subject.
Sanity check only (supplement-grade figure), not a probe/ablation.
"""
import json
import numpy as np
import pandas as pd
from scipy.signal import welch

import config as cfg
from io_utils import load_raw_mat

SUBJECT_ID = "02010002"


def main():
    df = pd.read_csv(cfg.MANIFEST_IN, dtype={"subject_id": str})
    row = df[df.subject_id == SUBJECT_ID].iloc[0]

    raw, fs = load_raw_mat(row["data_path"])
    raw_data = raw[: cfg.N_DATA_CH, :]

    cached = np.load(cfg.PREPROC_CACHE / f"{SUBJECT_ID}.npz")
    proc_data = cached["data"]

    nperseg = cfg.WIN_LEN  # 1000, matches the classification window -> 0.25 Hz resolution
    f_before, psd_before = welch(raw_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    f_after, psd_after = welch(proc_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)

    psd_before_db = 10 * np.log10(psd_before.mean(axis=0) + 1e-20)
    psd_after_db = 10 * np.log10(psd_after.mean(axis=0) + 1e-20)

    assert np.allclose(f_before, f_after)

    out = {
        "subject_id": SUBJECT_ID,
        "freqs": f_before.tolist(),
        "psd_before_db": psd_before_db.tolist(),
        "psd_after_db": psd_after_db.tolist(),
        "hpf": cfg.HPF,
        "lpf": cfg.LPF,
    }
    out_path = cfg.FIGS_DIR / "psd_check.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {out_path}")

    # quick numeric sanity: passband should have more power than sub-0.5Hz / above-45Hz after filtering
    band_lo = (f_before >= 0.5) & (f_before <= 45)
    band_below = f_before < 0.5
    band_above = f_before > 45
    print(f"after-filter mean dB: passband={psd_after_db[band_lo].mean():.1f}  "
          f"<0.5Hz={psd_after_db[band_below].mean():.1f}  "
          f">45Hz={psd_after_db[band_above].mean():.1f}")
    print(f"before-filter mean dB: passband={psd_before_db[band_lo].mean():.1f}  "
          f"<0.5Hz={psd_before_db[band_below].mean():.1f}  "
          f">45Hz={psd_before_db[band_above].mean():.1f}")


if __name__ == "__main__":
    main()
