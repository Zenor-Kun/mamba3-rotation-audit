"""
Task 4 Stage 2A, Protection 3: ERP-band autopsy, run ONLY on the 4
(channel_set, layer) cells that exceeded balanced_acc > 0.55 in
Protections 1+2 (frontal_only/h, all/s, posterior_only/s, frontal_only/s).
No contrasts or "overall" hit threshold, so only single conditions (h, s)
are needed. LR only, 5seed x 5fold, same primary pool (n=49), mad_k3
flag_set -- identical CV protocol to Protections 1+2.
"""
import json
import time

import pandas as pd

import erp_config as ecfg
import stage2a_config as cfg
from erp_roi import channel_set_variants
from stage2a_data import determine_primary_pool
from stage2a_eval import run_band_cv, summarize_points

FLAGS_DIR = ecfg.TASK4_DIR / "artifact_flags"

HITS = [
    {"channel_set": "frontal_only", "layer": "h", "broadband_balanced_acc": 0.5917, "broadband_auc": 0.6456},
    {"channel_set": "all", "layer": "s", "broadband_balanced_acc": 0.5830, "broadband_auc": 0.5722},
    {"channel_set": "posterior_only", "layer": "s", "broadband_balanced_acc": 0.5770, "broadband_auc": 0.6019},
    {"channel_set": "frontal_only", "layer": "s", "broadband_balanced_acc": 0.5673, "broadband_auc": 0.5870},
]


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.RUNS_DIR / f"{ts}_stage2a_protection3_band_autopsy"
    run_dir.mkdir(parents=True)

    manifest = pd.read_csv(ecfg.MANIFEST_128REST, dtype={"subject_id": str})
    label_of = dict(zip(manifest["subject_id"], manifest["label"]))
    splits = cfg.load_splits()

    all_sids = sorted(p.stem for p in FLAGS_DIR.glob("*.npz"))
    stage1c_pool = sorted(set(all_sids) - set(cfg.EXCLUDED_STAGE1C))
    primary_ids, _ = determine_primary_pool(stage1c_pool)
    variants = channel_set_variants()

    results = []
    log_lines = []
    t0 = time.time()
    for hit in HITS:
        vname, layer = hit["channel_set"], hit["layer"]
        roi_dict = variants[vname]
        cell = {"channel_set": vname, "condition": layer,
                "broadband": {"balanced_acc": hit["broadband_balanced_acc"], "auc": hit["broadband_auc"]},
                "bands": {}}
        for band in cfg.BANDS:
            for repr_kind in ("erp", "power"):
                points = run_band_cv(primary_ids, layer, "mad_k3", roi_dict, band, repr_kind, label_of, splits)
                summ = summarize_points(points)
                cell["bands"].setdefault(band, {})[repr_kind] = summ
                line = (f"[{vname}][{layer}][{band}][{repr_kind}] "
                        f"balanced_acc={summ['balanced_acc_mean']:.3f}+/-{summ['balanced_acc_std']:.3f} "
                        f"auc={summ['auc_mean']:.3f}+/-{summ['auc_std']:.3f}")
                print(line); log_lines.append(line)
        results.append(cell)
    print(f"protection 3 done in {time.time()-t0:.1f}s")

    report = {"primary_pool_n": len(primary_ids), "hits_examined": HITS, "band_autopsy": results,
              "bands_hz": cfg.BANDS,
              "epoch_length_caveat": "200 samples/800ms epoch: delta (0.5-4Hz, cycle 250-2000ms) is NOT "
                                      "well resolved within this window; theta/alpha/beta/gamma are."}
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(run_dir / "protection3.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(cfg.RESULTS_DIR / "erp_stage2a_protection3_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"archived to {run_dir}")
    return report


if __name__ == "__main__":
    main()
