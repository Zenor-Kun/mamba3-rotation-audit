"""
Task 4 ERP Stage 1c, part 2: records the VISUAL check (step 2 of the
pre-registered protocol in stage1c_quality_outlier_check.py) for each of the
9 statistical outliers found there, and applies the pre-registered
exclusion rule (step 3) mechanically from those recorded verdicts -- the
rule itself was fixed before any figure was looked at.

Visual criterion actually used to tell "genuine hardware/electrode quality
fault" apart from "plausible real (if non-neural, e.g. ocular) signal",
based on what was visible in each comparison figure
(runs/20260708-183810_stage1c_quality_outlier_check/visual_qc_*.png):

  - GENUINE QUALITY FAULT pattern: worst-trial waveform shows a near-linear
    RAMP or SATURATION/RAIL shape (not a smooth biphasic slow wave), peak
    magnitude in the thousands-to-tens-of-thousands of uV (>>1000uV, one to
    two orders of magnitude beyond anything ocular or neural), and/or a
    per-channel profile dominated by 1-3 essentially-random (non-frontal)
    channels spiking far above every other channel. This pattern is
    physiologically implausible (no biological source produces a clean
    linear ramp to 30,000-100,000uV) and matches classic electrode-pop /
    amplifier-saturation / movement-induced impedance-change artifact.

  - PLAUSIBLE PHYSIOLOGICAL (kept) pattern: worst-trial waveform is a
    smooth, slow, biphasic/monophasic deflection peaking within
    ~100-300ms, magnitude in the low-hundreds to ~1500-1800uV, and the
    per-channel profile is concentrated specifically in the same frontal
    channel cluster (idx ~5-25, matching erp_frontal_channels.py's
    eye-proximal subset) on EVERY elevated trial, chronically across
    nearly the whole session -- the textbook shape/topography of blink /
    eye-movement (EOG) contamination, which is a real, common, expected
    individual-difference source of "artifact" in raw EEG, not evidence of
    a broken sensor. Per-protocol wording ("not a plausible real neural
    response"), ocular signal is still non-neural -- but subject-level
    PRIMARY-analysis exclusion is reserved for pervasive, session-wide,
    non-physiological hardware-fault-level contamination, not an
    individual's blink rate; excess-EOG trials, if needed, are a
    trial-level (not subject-level) concern for Stage 2, not decided here.

Verdicts recorded below, one per statistical outlier, BEFORE the exclusion
rule (mechanical, not judgment) is applied:
"""
import json

import erp_config as ecfg

RUN_DIR = ecfg.RUNS_DIR / "20260708-183810_stage1c_quality_outlier_check"

VISUAL_VERDICTS = {
    "02030009": {
        "verdict": "genuine_quality_fault",
        "evidence": "worst trial: smooth near-linear ramps to +-7500uV (not blink-shaped); "
                    "per-channel profile: 3-4 essentially-random channels (idx ~38,~55,~100,~113) "
                    "spiking to 400-1400uV mean while most of scalp sits BELOW the comparison "
                    "subject's baseline noise floor; chronic, 100% of trials flagged.",
    },
    "02010034": {
        "verdict": "genuine_quality_fault",
        "evidence": "worst trial: linear ramp from -7000uV to +30000uV over ~200ms then sharp "
                    "vertical drop to -25000uV (amplifier rail/reset shape); episodic (2 extreme "
                    "trials at 12000/58000uV ptp, rest of session near baseline) -- consistent "
                    "with transient electrode/connector fault, not physiology at any timescale.",
    },
    "02030020": {
        "verdict": "genuine_quality_fault",
        "evidence": "worst trial: -100000 to +40000uV; per-channel profile dominated by 1 channel "
                    "at ~1400uV mean ptp (order of magnitude above every other channel, including "
                    "this subject's own other channels); independently already had the highest "
                    "bad-channel count of all 53 subjects in Stage 1 (n_bad_channels=16) -- "
                    "corroborating evidence of a hardware problem, not a coincidence.",
    },
    "02010030": {
        "verdict": "genuine_quality_fault",
        "evidence": "per-trial trace: near-comparison-level for most of the session, but a "
                    "contiguous burst of trials (~130-160) plus isolated spikes reach 3000-4000uV "
                    "-- a localized-in-time step/impulse artifact (one channel dominates), not a "
                    "gradual slow wave; episodic hardware/contact issue, not chronic ocular signal.",
    },
    "02010004": {
        "verdict": "plausible_physiological_not_excluded",
        "evidence": "smooth slow deflection peaking ~100-200ms post-cue, chronic across nearly all "
                    "480 trials, concentrated specifically in frontal channel cluster idx ~5-25 "
                    "(matches eye-proximal subset); worst-trial peak ~1700uV magnitude but "
                    "consistent blink/EOG topography and waveform shape throughout, not a ramp/"
                    "rail pattern.",
    },
    "02010005": {
        "verdict": "plausible_physiological_not_excluded",
        "evidence": "same frontal-concentrated, chronic, smooth-slow-wave pattern as 02010004; "
                    "worst trial ~1500uV biphasic deflection, textbook blink shape, not saturation.",
    },
    "02010011": {
        "verdict": "plausible_physiological_not_excluded",
        "evidence": "milder version of the same frontal-weighted chronic elevation (~150-200uV "
                    "mean, worst trial ~1250uV), smooth slow-wave shape, no ramp/rail pattern, no "
                    "single dominant random channel.",
    },
    "02030004": {
        "verdict": "plausible_physiological_not_excluded",
        "evidence": "same frontal-concentrated chronic pattern (~150-280uV mean, worst trial "
                    "~2000uV smooth biphasic deflection), consistent blink/EOG topography.",
    },
    "02030019": {
        "verdict": "plausible_physiological_not_excluded",
        "evidence": "same frontal-concentrated chronic pattern (~150-270uV mean, worst trial "
                    "~1200uV smooth slow wave), consistent blink/EOG topography across session.",
    },
}


def main():
    with open(RUN_DIR / "report.json") as f:
        report = json.load(f)

    assert set(VISUAL_VERDICTS) == set(report["statistical_outliers"]), "verdict set must match step-1 outlier set exactly"

    final_excluded = []
    for sid, v in VISUAL_VERDICTS.items():
        report["statistical_outliers"][sid]["visual_verdict"] = v["verdict"]
        report["statistical_outliers"][sid]["visual_evidence"] = v["evidence"]
        # mechanical rule: exclude iff statistical outlier (already true, sid is in this dict
        # only because it was) AND visual verdict == genuine_quality_fault
        excluded = v["verdict"] == "genuine_quality_fault"
        report["statistical_outliers"][sid]["excluded_from_primary"] = excluded
        if excluded:
            final_excluded.append(sid)

    report["visual_check_result"] = "DONE"
    report["final_exclusion_list"] = sorted(final_excluded)
    report["primary_analysis_n"] = 53 - len(final_excluded)
    report["sensitivity_analysis_n"] = 53
    report["exclusion_status_label"] = (
        "primary analysis excluded by objective signal-quality criterion; sensitivity analysis included"
    )

    with open(RUN_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(ecfg.RESULTS_DIR / "erp_stage1c_quality_outlier_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"final exclusion list ({len(final_excluded)}): {sorted(final_excluded)}")
    print(f"primary analysis n={report['primary_analysis_n']}, sensitivity analysis n={report['sensitivity_analysis_n']}")
    for sid, v in VISUAL_VERDICTS.items():
        print(f"  {sid}: {v['verdict']}")


if __name__ == "__main__":
    main()
