"""Unified tip measurement: same tip IDs across Methods 1–3; hard validity first."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sem_analysis.arch_detection import ValidatedArch, detect_validated_arches
from sem_analysis.edge_probability import preprocess_sem
from sem_analysis.methods.fixed_distance_circle import (
    measure_method1_multi,
    method1_to_dict,
)
from sem_analysis.methods.inscribed_angle import (
    inscribed_angle_to_dict,
    measure_inscribed_angle,
)
from sem_analysis.methods.projected_tip_distance import (
    measure_projected_tip_distance,
    projected_tip_distance_to_dict,
)
from sem_analysis.protocol import get_protocol
from sem_analysis.roi import MeasurementROI, extract_measurement_roi
from sem_analysis.stats_summary import summarize_values, tip_confidence


@dataclass
class TipMeasurement:
    tip_id: int
    apex_x_px: float
    apex_y_px: float
    nm_per_px: float
    border_valid: bool
    left_branch_valid: bool
    right_branch_valid: bool
    fit_residual_px: float
    window_valid: bool
    hard_valid: bool
    method1: dict = field(default_factory=dict)
    method2: dict = field(default_factory=dict)
    method3: dict = field(default_factory=dict)
    method1_valid: bool = False
    method2_valid: bool = False
    method3_valid: bool = False
    confidence: float = 0.0
    rejection_reason: str | None = None

    def to_row(self) -> dict:
        row = {
            "tip_id": self.tip_id,
            "apex_x_px": self.apex_x_px,
            "apex_y_px": self.apex_y_px,
            "nm_per_px": self.nm_per_px,
            "border_valid": self.border_valid,
            "left_branch_valid": self.left_branch_valid,
            "right_branch_valid": self.right_branch_valid,
            "fit_residual_px": self.fit_residual_px,
            "window_valid": self.window_valid,
            "hard_valid": self.hard_valid,
            "method1_valid": self.method1_valid,
            "method2_valid": self.method2_valid,
            "method3_valid": self.method3_valid,
            "confidence": self.confidence if self.hard_valid else 0.0,
            "rejection_reason": self.rejection_reason,
            "projected_tip_distance_nm": self.method2.get("distance_l_nm"),
            "angle_D100_deg": self.method3.get("angle_degrees"),
        }
        for label, data in (self.method1.get("radii_by_l") or {}).items():
            row[f"{label}_nm"] = data.get("projected_radius_nm") or data.get("radius_nm")
        # Ensure R columns exist
        for lab in ("R25", "R50", "R100", "R200"):
            row.setdefault(f"{lab}_nm", None)
        return row


def _hard_valid(arch: ValidatedArch) -> tuple[bool, str | None]:
    if not arch.window_valid:
        return False, arch.rejection_reason or "incomplete_measurement_window"
    if not arch.border_valid:
        return False, "touches_border"
    if not arch.left_branch_valid or not arch.right_branch_valid:
        return False, "missing_branch"
    if not arch.valid:
        return False, arch.rejection_reason or "arch_invalid"
    if arch.fit_residual_px > 2.0:
        return False, "fit_residual_too_high"
    return True, None


def measure_all_tips(
    roi: MeasurementROI,
    nm_per_px: float,
    config: dict,
    image_id: str = "",
) -> tuple[list[TipMeasurement], dict]:
    """Detect validated arches and run Methods 1–3 on the same tip IDs."""
    proto = get_protocol(config)
    edge_maps = preprocess_sem(roi.image)
    arches = detect_validated_arches(roi.image, nm_per_px, config, edge_maps=edge_maps)

    # Map ROI coords → original image coords for reporting
    ox, oy = roi.offset_x, roi.offset_y

    tips: list[TipMeasurement] = []
    for arch in arches:
        if arch.tip_id < 0 and not arch.valid:
            # Diagnostic reject without tip_id — skip unified table or include?
            continue
        if arch.tip_id < 0:
            continue

        hard_ok, reason = _hard_valid(arch)
        apex = np.array([arch.apex_x_px, arch.apex_y_px], dtype=float)
        left = arch.left_smooth if arch.left_smooth is not None else arch.left_raw
        right = arch.right_smooth if arch.right_smooth is not None else arch.right_raw

        tm = TipMeasurement(
            tip_id=arch.tip_id,
            apex_x_px=arch.apex_x_px + ox,
            apex_y_px=arch.apex_y_px + oy,
            nm_per_px=nm_per_px,
            border_valid=arch.border_valid,
            left_branch_valid=arch.left_branch_valid,
            right_branch_valid=arch.right_branch_valid,
            fit_residual_px=arch.fit_residual_px,
            window_valid=arch.window_valid,
            hard_valid=hard_ok,
            rejection_reason=reason,
            confidence=0.0,
        )

        if not hard_ok:
            tips.append(tm)
            continue

        # Method 1
        m1 = measure_method1_multi(
            apex, left, right, nm_per_px, proto["method1_distances_nm"], tip_id=arch.tip_id
        )
        radii_by_l = {k: method1_to_dict(v) for k, v in m1.items()}
        primary_label = f"R{int(round(proto['method1_primary_nm']))}"
        primary = m1.get(primary_label) or next(iter(m1.values()), None)
        tm.method1 = {
            "radii_by_l": radii_by_l,
            **(method1_to_dict(primary) if primary else {}),
        }
        tm.method1_valid = bool(primary and primary.valid)

        # Method 2
        m2 = measure_projected_tip_distance(
            nm_per_pixel=nm_per_px,
            fit_band_nm=proto["method2_fit_band_nm"],
            apex=apex,
            left=left,
            right=right,
            tip_id=arch.tip_id,
            min_flank_points=int(
                config.get("measurement_methods", {})
                .get("projected_tip_distance", {})
                .get("min_flank_points", 5)
            ),
        )
        if m2:
            # Shift drawable coords to original image
            d = projected_tip_distance_to_dict(m2)
            if d.get("tip_point"):
                d["tip_point"] = [d["tip_point"][0] + ox, d["tip_point"][1] + oy]
            if d.get("convergence_point"):
                d["convergence_point"] = [
                    d["convergence_point"][0] + ox,
                    d["convergence_point"][1] + oy,
                ]
            tm.method2 = d
            tm.method2_valid = bool(m2.valid)

        # Method 3 Interpretation A
        m3 = measure_inscribed_angle(
            circle_diameter_nm=proto["method3_circle_diameter_nm"],
            nm_per_pixel=nm_per_px,
            apex=apex,
            left=left,
            right=right,
            tip_id=arch.tip_id,
        )
        if m3:
            d3 = inscribed_angle_to_dict(m3)
            d3["tip_point"] = [d3["tip_point"][0] + ox, d3["tip_point"][1] + oy]
            d3["circle_center"] = [d3["circle_center"][0] + ox, d3["circle_center"][1] + oy]
            tm.method3 = d3
            tm.method3_valid = bool(m3.valid)

        # Confidence only after hard validity
        weights = config.get("measurement_methods", {}).get("confidence_weights") or {
            "edge": 0.25,
            "continuity": 0.20,
            "fit": 0.20,
            "symmetry": 0.15,
            "stability": 0.20,
        }
        # Remap to tip_confidence keys
        stab = 1.0
        if primary and primary.stability_s is not None:
            stab = float(max(0.0, 1.0 - primary.stability_s / 0.20))
        tm.confidence = tip_confidence(
            edge_score=arch.edge_score,
            symmetry_score=1.0 if arch.left_branch_valid and arch.right_branch_valid else 0.0,
            fit_score=max(0.0, 1.0 - arch.fit_residual_px / 2.0),
            continuity_score=1.0 if tm.method1_valid and tm.method2_valid else 0.5,
            consensus_score=stab,
            weights={
                "edge": weights.get("edge", 0.25),
                "continuity": weights.get("continuity", 0.20),
                "fit": weights.get("fit", 0.20),
                "symmetry": weights.get("symmetry", 0.15),
                "consensus": weights.get("stability", 0.20),
            },
        )
        tips.append(tm)

    # Summaries — accepted hard-valid tips only; methods separate
    accepted = [t for t in tips if t.hard_valid]
    def _vals(getter):
        return [v for v in (getter(t) for t in accepted) if v is not None]

    summary = {
        "image_id": image_id,
        "n_detected_arches": len(arches),
        "n_hard_valid": len(accepted),
        "protocol": proto,
        "nm_per_px": nm_per_px,
        "tilt_note": "Measurements are projected (tilt metadata stored; no blind 2× correction).",
        "fixed_distance_circle": {
            "headline": "median",
            **summarize_values(
                _vals(lambda t: (t.method1.get("projected_radius_nm") or t.method1.get("radius_nm"))
                      if t.method1_valid else None)
            ),
            "count": sum(1 for t in accepted if t.method1_valid),
            "per_curve": [
                {
                    "peak_id": t.tip_id,
                    "peak_location": [t.apex_x_px, t.apex_y_px],
                    "confidence": t.confidence,
                    **t.method1,
                }
                for t in accepted
                if t.method1_valid
            ],
        },
        "projected_tip_distance": {
            "headline": "median",
            **summarize_values(_vals(lambda t: t.method2.get("distance_l_nm") if t.method2_valid else None)),
            "median_distance_l_nm": None,
            "count": sum(1 for t in accepted if t.method2_valid),
            "per_curve": [
                {
                    "peak_id": t.tip_id,
                    "peak_location": [t.apex_x_px, t.apex_y_px],
                    "confidence": t.confidence,
                    **t.method2,
                }
                for t in accepted
                if t.method2_valid
            ],
        },
        "inscribed_angle": {
            "headline": "median",
            **summarize_values(_vals(lambda t: t.method3.get("angle_degrees") if t.method3_valid else None)),
            "median_angle_deg": None,
            "count": sum(1 for t in accepted if t.method3_valid),
            "per_curve": [
                {
                    "peak_id": t.tip_id,
                    "peak_location": [t.apex_x_px, t.apex_y_px],
                    "confidence": t.confidence,
                    **t.method3,
                }
                for t in accepted
                if t.method3_valid
            ],
        },
        "tip_validation": {
            "n_detected_candidates": len(arches),
            "n_accepted": len(accepted),
        },
    }
    summary["fixed_distance_circle"]["median_radius_nm"] = summary["fixed_distance_circle"].get("median")
    summary["fixed_distance_circle"]["mean_radius_nm"] = summary["fixed_distance_circle"].get("mean")
    summary["projected_tip_distance"]["median_distance_l_nm"] = summary["projected_tip_distance"].get("median")
    summary["projected_tip_distance"]["mean_distance_l_nm"] = summary["projected_tip_distance"].get("mean")
    summary["inscribed_angle"]["median_angle_deg"] = summary["inscribed_angle"].get("median")
    summary["inscribed_angle"]["mean_angle_deg"] = summary["inscribed_angle"].get("mean")

    return tips, summary


def tips_to_dataframe(tips: list[TipMeasurement], image_id: str = "") -> pd.DataFrame:
    rows = []
    for t in tips:
        row = t.to_row()
        row["image_id"] = image_id
        rows.append(row)
    cols = [
        "image_id", "tip_id", "apex_x_px", "apex_y_px", "nm_per_px",
        "border_valid", "left_branch_valid", "right_branch_valid", "fit_residual_px",
        "R25_nm", "R50_nm", "R100_nm", "R200_nm",
        "projected_tip_distance_nm", "angle_D100_deg",
        "method1_valid", "method2_valid", "method3_valid",
        "confidence", "rejection_reason", "hard_valid", "window_valid",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols] if len(df) else pd.DataFrame(columns=cols)
