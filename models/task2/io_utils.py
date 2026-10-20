import scipy.io as sio
import numpy as np


def load_raw_mat(path):
    """Returns (data, fs) where data is (129, T) float64: rows 0..127 = EEG channels,
    row 128 = Cz (the online reference, stored as the 129th row)."""
    d = sio.loadmat(path)
    data_key = None
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if k == "samplingRate" or k == "Impedances_0":
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 129:
            data_key = k
            break
    if data_key is None:
        raise ValueError(f"{path}: could not find a (129, T) data array among keys {list(d.keys())}")
    data = d[data_key].astype(np.float64)
    fs = float(np.asarray(d["samplingRate"]).squeeze())
    return data, fs
