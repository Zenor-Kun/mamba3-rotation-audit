"""
Task 4 Stage 2A: channel ROI partition + channel-set variants for the
frontal/EOG spatial control. Montage-based (GSN-HydroCel-129 3D positions),
not tuned to this dataset's data -- same non-circularity principle as
erp_frontal_channels.py, which this module reuses directly for the
near-eye 10-channel cluster.

ROI partition (frontal / central / parietal_occipital): tertile split of
E1..E128 by the montage's anterior-posterior (y) coordinate, most-anterior
third = frontal, most-posterior third = parietal_occipital, remainder =
central. This is a coarse, 3-way, montage-derived partition -- not the
literature 10-20 lobe boundaries, but adequate for a low-dimensional
ROI x time-window confirmatory feature set (see stage2a_config.py).

Channel-set variants (Protection 2, frontal/EOG spatial control):
  - all:                 the 3-ROI partition above, unmodified.
  - exclude_frontal_eog: same 3-ROI partition, but the near-eye 10-channel
                          cluster (a SUBSET of the frontal tertile, not all
                          of it) is removed from the frontal ROI only;
                          central/parietal_occipital unchanged.
  - frontal_only:        a single ROI containing ONLY the near-eye
                          10-channel cluster (not the whole frontal
                          tertile) -- this is the literal "frontal/near-eye
                          idx~5-25 cluster" the spec refers to.
  - posterior_only:      a single ROI containing only the parietal_occipital
                          tertile.
"""
import numpy as np

import erp_config as ecfg
from erp_frontal_channels import frontal_channel_indices


Z_PERIPHERAL_THRESH = -0.03  # excludes the EGI net's bottom ring (mastoid/cheek/neck sites)


def roi_partition():
    """Tertile split of E1..E128 by anterior-posterior (y) montage position,
    RESTRICTED to upper-scalp channels (z >= Z_PERIPHERAL_THRESH) -- a naive
    y-only ranking over all 128 channels pulls the EGI net's bottom ring
    (mastoid/cheek/neck sites, e.g. E113-E128, E43/44/48/49/etc, which sit at
    moderate y but strongly negative z) into the "frontal" tertile, which is
    anatomically wrong. Peripheral-ring channels are left OUT of all 3 ROIs
    entirely (not assigned anywhere) rather than force-assigned.
    Returns {roi_name: np.ndarray of channel indices (0-based, E1=0..E128=127)}."""
    import mne
    mont = mne.channels.make_standard_montage("GSN-HydroCel-129")
    ch_pos = mont.get_positions()["ch_pos"]
    names = [f"E{i}" for i in range(1, ecfg.N_DATA_CH + 1)]
    y = np.array([ch_pos[n][1] for n in names])
    z = np.array([ch_pos[n][2] for n in names])
    upper = np.where(z >= Z_PERIPHERAL_THRESH)[0]
    order = upper[np.argsort(-y[upper])]  # most anterior first, upper-scalp only
    n = len(order)
    third = n // 3
    frontal = np.sort(order[:third])
    central = np.sort(order[third:2 * third])
    parietal_occipital = np.sort(order[2 * third:])
    return {"frontal": frontal, "central": central, "parietal_occipital": parietal_occipital}


def near_eye_indices():
    idx, _names = frontal_channel_indices()
    return np.sort(np.asarray(idx))


def channel_set_variants():
    """Returns {variant_name: {roi_name: channel_idx_array}}."""
    roi = roi_partition()
    eye = near_eye_indices()

    exclude_frontal = np.sort(np.setdiff1d(roi["frontal"], eye))

    return {
        "all": {
            "frontal": roi["frontal"], "central": roi["central"],
            "parietal_occipital": roi["parietal_occipital"],
        },
        "exclude_frontal_eog": {
            "frontal": exclude_frontal, "central": roi["central"],
            "parietal_occipital": roi["parietal_occipital"],
        },
        "frontal_only": {"frontal_eog": eye},
        "posterior_only": {"parietal_occipital": roi["parietal_occipital"]},
    }


if __name__ == "__main__":
    roi = roi_partition()
    for name, idx in roi.items():
        print(f"{name}: n={len(idx)} channels={[f'E{i+1}' for i in idx]}")
    eye = near_eye_indices()
    print(f"\nnear_eye (10): {[f'E{i+1}' for i in eye]}")
    variants = channel_set_variants()
    for vname, rois in variants.items():
        sizes = {r: len(idx) for r, idx in rois.items()}
        print(f"variant={vname}: roi sizes={sizes}")
