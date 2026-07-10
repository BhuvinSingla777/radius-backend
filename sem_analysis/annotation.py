"""Brainstorming method visualization — separate per-method annotated outputs."""

from __future__ import annotations

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sem_analysis.deduction import FilteredDetection
from sem_analysis.edge_detection import EdgePeakResult
from sem_analysis.methods.fixed_distance_circle import FixedDistanceCircleResult
from sem_analysis.methods.inscribed_angle import InscribedAngleResult
from sem_analysis.methods.projected_tip_distance import ProjectedTipDistanceResult
from sem_analysis.radius_computation import RadiusResult

# BGR colors
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)


def _draw_scale_bar(ax, nm_per_pixel: float, scale_bar_nm: float, position: str = "bottom-right") -> None:
    bar_px = scale_bar_nm / nm_per_pixel
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    margin = 20
    x_start = xlim[1] - bar_px - margin if "right" in position else xlim[0] + margin
    y_pos = ylim[0] + margin if "bottom" in position else ylim[1] - margin - 10
    ax.plot([x_start, x_start + bar_px], [y_pos, y_pos], "w-", linewidth=3)
    ax.text(
        x_start + bar_px / 2, y_pos + 5, f"{scale_bar_nm:.0f} nm",
        color="white", ha="center", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
    )


def _save_annotated(img: np.ndarray, output_path: str, nm_per_pixel: float, config: dict) -> None:
    cfg = config.get("annotation", {})
    scale_bar_nm = cfg.get("scale_bar_nm", 50)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    _draw_scale_bar(ax, nm_per_pixel, scale_bar_nm, cfg.get("scale_bar_position", "bottom-right"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.get("output_dpi", 300), bbox_inches="tight")
    plt.close()


def _base_image(image: np.ndarray) -> np.ndarray:
    uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(uint8, cv2.COLOR_GRAY2BGR)


def _draw_line(img: np.ndarray, coords: list[float], color: tuple, thickness: int = 2) -> None:
    if len(coords) < 4:
        return
    x1, y1, x2, y2 = (int(v) for v in coords[:4])
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _draw_dot(img: np.ndarray, point: tuple[float, float] | list[float], color: tuple, radius: int = 4) -> None:
    cv2.circle(img, (int(point[0]), int(point[1])), radius, color, -1, cv2.LINE_AA)


def _draw_vertical_l(img: np.ndarray, coords: list[float], label: str = "l") -> None:
    if len(coords) < 4:
        return
    x, y_top, _, y_bottom = (int(v) for v in coords[:4])
    y_top, y_bottom = int(min(y_top, y_bottom)), int(max(y_top, y_bottom))
    cv2.line(img, (x, y_top), (x, y_bottom), RED, 2, cv2.LINE_AA)
    tick = 8
    cv2.line(img, (x - tick, y_top), (x + tick, y_top), RED, 2, cv2.LINE_AA)
    cv2.line(img, (x - tick, y_bottom), (x + tick, y_bottom), RED, 2, cv2.LINE_AA)
    cv2.putText(img, label, (x + 10, (y_top + y_bottom) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1, cv2.LINE_AA)


def _draw_method1_curve(img: np.ndarray, curve: dict, index: int) -> None:
    """Method 1: blue dots, red scan line, cyan inscribed circle, R label."""
    tip = curve.get("tip_point")
    left = curve.get("intersection_left")
    right = curve.get("intersection_right")
    center = curve.get("center")
    radius_px = curve.get("radius_px")
    scan_line = curve.get("scan_line", [])

    if tip:
        _draw_dot(img, tip, BLUE)
    if left:
        _draw_dot(img, left, BLUE, 3)
    if right:
        _draw_dot(img, right, BLUE, 3)
    if scan_line:
        _draw_line(img, scan_line, RED, 2)
    if center and radius_px:
        cv2.circle(img, (int(center[0]), int(center[1])), max(3, int(radius_px)), CYAN, 1, cv2.LINE_AA)

    peak = curve.get("peak_location") or tip
    if peak and curve.get("radius_nm") is not None:
        px, py = int(peak[0]), int(peak[1])
        label = f"R={curve['radius_nm']:.1f}nm"
        ly = py - 6 if index % 2 == 0 else py + 12
        cv2.putText(img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, YELLOW, 1, cv2.LINE_AA)


def _draw_method2_curve(img: np.ndarray, curve: dict, index: int) -> None:
    """Method 2: yellow tangents, red vertical l, blue tip dot/arc."""
    _draw_line(img, curve.get("left_line", []), YELLOW, 2)
    _draw_line(img, curve.get("right_line", []), YELLOW, 2)
    _draw_vertical_l(img, curve.get("vertical_l_line", []), "l")

    tip = curve.get("tip_point")
    if tip:
        _draw_dot(img, tip, BLUE, 5)

    arc_center = curve.get("tip_arc_center")
    arc_angles = curve.get("tip_arc_angles")
    arc_r = curve.get("tip_radius_px")
    if arc_center and arc_angles and arc_r:
        cx, cy = int(arc_center[0]), int(arc_center[1])
        start_deg = math.degrees(arc_angles[0])
        end_deg = math.degrees(arc_angles[1])
        cv2.ellipse(img, (cx, cy), (int(arc_r), int(arc_r)), 0, start_deg, end_deg, BLUE, 2, cv2.LINE_AA)

    peak = curve.get("peak_location") or tip
    if peak and curve.get("distance_l_nm") is not None:
        px, py = int(peak[0]), int(peak[1])
        label = f"l={curve['distance_l_nm']:.1f}nm"
        ly = py - 6 if index % 2 == 0 else py + 12
        cv2.putText(img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, YELLOW, 1, cv2.LINE_AA)


def _draw_method3_curve(img: np.ndarray, curve: dict, index: int) -> None:
    """Method 3: fixed circle, tangent lines, theta label."""
    center = curve.get("circle_center")
    radius_px = curve.get("circle_radius_px")
    if center and radius_px:
        cv2.circle(img, (int(center[0]), int(center[1])), max(3, int(radius_px)), CYAN, 1, cv2.LINE_AA)

    tip = curve.get("tip_point")
    if tip:
        _draw_dot(img, tip, BLUE, 4)

    left = curve.get("intersection_left")
    right = curve.get("intersection_right")
    if left:
        _draw_dot(img, left, BLUE, 3)
    if right:
        _draw_dot(img, right, BLUE, 3)

    _draw_line(img, curve.get("left_tangent_line", []), YELLOW, 2)
    _draw_line(img, curve.get("right_tangent_line", []), YELLOW, 2)

    peak = curve.get("peak_location") or tip
    if peak and curve.get("angle_degrees") is not None:
        px, py = int(peak[0]), int(peak[1])
        label = f"θ={curve['angle_degrees']:.1f}°"
        ly = py - 6 if index % 2 == 0 else py + 12
        cv2.putText(img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, YELLOW, 1, cv2.LINE_AA)


def _draw_all_serration_curves(img: np.ndarray, radius_results: list[RadiusResult]) -> None:
    """Draw cyan fitted circle + R label at every detected serration peak."""
    for i, r in enumerate(radius_results):
        cx, cy = int(r.center[0]), int(r.center[1])
        r_px = max(3, int(r.radius_px))
        cv2.circle(img, (cx, cy), r_px, CYAN, 1, cv2.LINE_AA)

        if r.peak_location:
            px, py = int(r.peak_location[0]), int(r.peak_location[1])
        else:
            px, py = cx, cy - r_px
        cv2.circle(img, (px, py), 3, BLUE, -1, cv2.LINE_AA)

        if r.tangent_lines:
            for p1, p2 in r.tangent_lines:
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(img, pt1, pt2, MAGENTA, 1, cv2.LINE_AA)

        label = f"R={r.radius_nm:.1f}nm"
        if r.opening_angle_deg is not None:
            label += f" A={r.opening_angle_deg:.0f}°"
        lx = px + 6
        ly = py - 4 if i % 2 == 0 else py + 14
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, YELLOW, 1, cv2.LINE_AA)


def annotate_method1_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Method 1 annotated image — fixed-distance inscribed circle per curve."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_method1_curve(img, curve, i)
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_method2_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Method 2 annotated image — projected tip distance per curve."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_method2_curve(img, curve, i)
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_method3_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Method 3 annotated image — inscribed angle per curve."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        _draw_method3_curve(img, curve, i)
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_research_image(
    image: np.ndarray,
    per_curve: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Research-grade osculating circle annotated image."""
    img = _base_image(image)
    for i, curve in enumerate(per_curve):
        if curve.get("rejected"):
            continue
        _draw_line(img, curve.get("left_line", []), YELLOW, 2)
        _draw_line(img, curve.get("right_line", []), YELLOW, 2)
        _draw_vertical_l(img, curve.get("vertical_l_line", []), "l")

        va = curve.get("virtual_apex")
        if va:
            _draw_dot(img, va, MAGENTA, 4)

        tip = curve.get("physical_tip")
        if tip:
            _draw_dot(img, tip, BLUE, 5)

        center = curve.get("center")
        r_px = curve.get("radius_px")
        if center and r_px:
            cv2.circle(img, (int(center[0]), int(center[1])), max(3, int(r_px)), CYAN, 1, cv2.LINE_AA)

        peak = curve.get("peak_location") or tip
        if peak and curve.get("radius_um") is not None:
            px, py = int(peak[0]), int(peak[1])
            conf = curve.get("confidence_score", 0)
            label = f"R={curve['radius_um']:.2f}um C={conf*100:.0f}%"
            ly = py - 6 if i % 2 == 0 else py + 12
            cv2.putText(img, label, (px + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.3, YELLOW, 1, cv2.LINE_AA)

    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_validated_tips(
    image: np.ndarray,
    tips: list[dict],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    """Draw hard-valid protocol tip apexes (arch-first pipeline overview)."""
    img = _base_image(image)
    for tip in tips:
        loc = tip.get("peak_location") or [tip.get("apex_x_px"), tip.get("apex_y_px")]
        if not loc or loc[0] is None or loc[1] is None:
            continue
        px, py = float(loc[0]), float(loc[1])
        valid = tip.get("hard_valid", True)
        color = CYAN if valid else RED
        _draw_dot(img, (px, py), color, radius=5)
        tip_id = tip.get("tip_id", tip.get("peak_id", "?"))
        cv2.putText(
            img,
            f"tip {tip_id}",
            (int(px) + 8, int(py) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    if output_path:
        _save_annotated(img, output_path, nm_per_pixel, config)
    return img


def annotate_image(
    image: np.ndarray,
    detections: list[FilteredDetection],
    edge_results: list[EdgePeakResult],
    nm_per_pixel: float,
    config: dict,
    output_path: str | None = None,
    brainstorming_raw: dict | None = None,
    all_radii: list[RadiusResult] | None = None,
    show_secondary_methods: bool = False,
) -> np.ndarray:
    """Render Hough bulk serration curve annotation (cyan circles)."""
    annotated = _base_image(image)

    for edge in edge_results:
        if edge.hough_lines:
            for line in edge.hough_lines:
                if line and len(line) >= 4:
                    x1, y1, x2, y2 = (int(v) for v in line[:4])
                    cv2.line(annotated, (x1, y1), (x2, y2), GREEN, 1, cv2.LINE_AA)

    radii = all_radii or []
    if not radii:
        for det in detections:
            if det.passed:
                radii.extend(det.radius_results)
    _draw_all_serration_curves(annotated, radii)

    if output_path:
        _save_annotated(annotated, output_path, nm_per_pixel, config)

    return annotated
