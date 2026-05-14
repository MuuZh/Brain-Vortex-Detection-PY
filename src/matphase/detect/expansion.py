"""
MATLAB-style spiral expansion utilities.

This module reconstructs the post-detection expansion step used in the
original MATLAB pipeline. After initial curl thresholding identifies spiral
cores, the MATLAB code:
    1. Compares phase vectors against center-originated vectors to keep only
       voxels whose angles fall within +/-45° of the ideal 90° difference.
    2. Grows concentric radii outward from each spiral center until the
       number of excluded voxels exceeds expansion_threshold * radius.
    3. Applies an additional phase-difference filter between raw and
       smoothed phase volumes to ensure compatibility.

The helpers in this module re-create those behaviors so Python detections
match MATLAB footprints and vortex sizes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from matphase.detect.phase_field import angle_subtract
from matphase.detect.spirals import SpiralDetectionResult, SpiralPattern


def compute_phase_alignment_mask(
    raw_phase: np.ndarray,
    smooth_phase: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Return mask where |raw_phase - smooth_phase| <= threshold.

    Parameters
    ----------
    raw_phase : np.ndarray
        Phase prior to spatial smoothing (e.g., temporal-only Hilbert output).
    smooth_phase : np.ndarray
        Phase after spatial filtering (DoG/Hilbert).
    threshold : float
        Maximum absolute angular difference in radians (MATLAB default pi/6).

    Returns
    -------
    np.ndarray
        Boolean mask with True where voxels pass the phase-difference filter.
    """
    if raw_phase.shape != smooth_phase.shape:
        raise ValueError(
            f"Phase arrays must share shape, got {raw_phase.shape} vs {smooth_phase.shape}"
        )
    if threshold is None:
        return np.isfinite(raw_phase) & np.isfinite(smooth_phase)

    diff = np.abs(angle_subtract(raw_phase, smooth_phase))
    mask = np.isfinite(diff) & (diff <= threshold)
    return mask


def expand_spiral_patterns(
    detection_result: SpiralDetectionResult,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray],
    phase_alignment_mask: Optional[np.ndarray],
    angle_center: float,
    angle_half_width: float,
    expansion_threshold: float,
    radius_min: float,
    radius_max: float,
    radius_step: float,
    center_patch_radius: int = 1,
    show_progress: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[int, Dict[int, np.ndarray]], Dict[int, Dict[int, float]]]:
    """
    Expand spiral detections using MATLAB radius + angle constraints.

    Parameters
    ----------
    detection_result : SpiralDetectionResult
        Initial detection result produced from curl thresholding.
    vx, vy : np.ndarray
        Normalized phase-gradient components (vector field), shape (rows, cols, frames).
    valid_mask : np.ndarray or None
        Boolean mask of valid cortical voxels (same shape as vx/vy) or None.
    phase_alignment_mask : np.ndarray or None
        Boolean mask enforcing raw-vs-smooth phase compatibility per voxel/time.
    angle_center : float
        Ideal angle difference (radians) between radial vectors and phase vectors (pi/2).
    angle_half_width : float
        Allowed deviation (radians) around the ideal angle (pi/4).
    expansion_threshold : float
        MATLAB's expansion_threshold parameter (voxels excluded <= threshold * radius).
    radius_min, radius_max, radius_step : float
        Radius sweep configuration (grid units).
    center_patch_radius : int
        Radius (pixels) of a forced-on neighborhood around spiral centers.
    show_progress : bool, optional
        If True, show a tqdm progress bar while iterating spiral seeds.

    Returns
    -------
    masks : dict[str, np.ndarray]
        Directional binary masks ('ccw', 'cw').
    pattern_slices : dict[int, dict[int, np.ndarray]]
        Per-pattern, per-time expanded slices.
    expansion_radii : dict[int, dict[int, float]]
        Per-pattern, per-time expansion radii used.
    """
    if vx.shape != vy.shape:
        raise ValueError(f"vx and vy must share shape, got {vx.shape} vs {vy.shape}")

    rows, cols, frames = vx.shape
    masks = {
        "ccw": np.zeros((rows, cols, frames), dtype=bool),
        "cw": np.zeros((rows, cols, frames), dtype=bool),
    }
    pattern_slices: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    expansion_radii: Dict[int, Dict[int, float]] = defaultdict(dict)
    if detection_result.num_patterns == 0:
        return masks, pattern_slices, expansion_radii

    grid_y, grid_x = np.indices((rows, cols), dtype=float)

    angle_low = max(angle_center - angle_half_width, 0.0)
    angle_high = min(angle_center + angle_half_width, np.pi)

    valid_mask = (
        np.asarray(valid_mask, dtype=bool) if valid_mask is not None else None
    )
    phase_alignment_mask = (
        np.asarray(phase_alignment_mask, dtype=bool)
        if phase_alignment_mask is not None
        else None
    )

    radius_min = max(radius_min, 0.0)
    radius_max = max(radius_max, radius_min)
    threshold = float(expansion_threshold if expansion_threshold is not None else 1.0)

    pattern_iterable = detection_result.patterns
    if show_progress and detection_result.num_patterns > 5:
        pattern_iterable = tqdm(
            pattern_iterable,
            desc="Spiral expansion",
            unit="pattern",
            total=detection_result.num_patterns,
        )

    for pattern in pattern_iterable:
        rotation = pattern.rotation_direction
        if rotation not in {"ccw", "cw"}:
            continue
        for idx, abs_time in enumerate(pattern.absolute_times):
            if not (0 <= abs_time < frames):
                continue
            center = pattern.weighted_centroids[idx]
            if np.any(~np.isfinite(center)):
                continue

            mask_slice, used_radius = _expand_single_pattern(
                center=center,
                grid_x=grid_x,
                grid_y=grid_y,
                vx_frame=vx[:, :, abs_time],
                vy_frame=vy[:, :, abs_time],
                valid_slice=valid_mask[:, :, abs_time] if valid_mask is not None else None,
                compatibility_slice=(
                    phase_alignment_mask[:, :, abs_time]
                    if phase_alignment_mask is not None
                    else None
                ),
                angle_low=angle_low,
                angle_high=angle_high,
                expansion_threshold=threshold,
                radius_min=radius_min,
                radius_max=radius_max,
                radius_step=radius_step,
                center_patch_radius=center_patch_radius,
            )
            if mask_slice is None or not np.any(mask_slice):
                continue
            pattern_slices[pattern.pattern_id][int(abs_time)] = mask_slice.copy()
            expansion_radii[pattern.pattern_id][int(abs_time)] = float(used_radius)
            masks[rotation][:, :, abs_time] |= mask_slice

    return masks, pattern_slices, expansion_radii


def _expand_single_pattern(
    *,
    center: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    vx_frame: np.ndarray,
    vy_frame: np.ndarray,
    valid_slice: Optional[np.ndarray],
    compatibility_slice: Optional[np.ndarray],
    angle_low: float,
    angle_high: float,
    expansion_threshold: float,
    radius_min: float,
    radius_max: float,
    radius_step: float,
    center_patch_radius: int,
) -> Tuple[Optional[np.ndarray], float]:
    """Return boolean mask of expanded spiral footprint for a single frame and the radius used."""
    rows, cols = vx_frame.shape
    cx, cy = float(center[0]), float(center[1])

    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None, 0.0
    if cx < 0 or cx >= cols or cy < 0 or cy >= rows:
        return None, 0.0

    center_vx = np.round(grid_x - cx)
    center_vy = np.round(grid_y - cy)
    center_abs = np.hypot(center_vx, center_vy)
    center_abs[center_abs == 0] = 1.0
    if valid_slice is not None:
        center_abs = center_abs.copy()
        center_abs[~valid_slice] = np.nan

    vector_complex = vx_frame + 1j * vy_frame
    vector_angles = np.angle(vector_complex)
    radial_angles = np.angle(center_vx + 1j * center_vy)
    angle_diff = np.abs(angle_subtract(radial_angles, vector_angles))

    angle_filtered = angle_diff.copy()
    angle_filtered[(angle_filtered < angle_low) | (angle_filtered > angle_high)] = np.nan
    angle_filtered[~np.isfinite(vector_angles)] = np.nan

    if valid_slice is not None:
        angle_filtered[~valid_slice] = np.nan
    if compatibility_slice is not None:
        angle_filtered[~compatibility_slice] = np.nan

    if center_patch_radius >= 0:
        cx_round = int(round(cx))
        cy_round = int(round(cy))
        x0 = max(cx_round - center_patch_radius, 0)
        x1 = min(cx_round + center_patch_radius, cols - 1)
        y0 = max(cy_round - center_patch_radius, 0)
        y1 = min(cy_round + center_patch_radius, rows - 1)
        angle_filtered[y0 : y1 + 1, x0 : x1 + 1] = np.pi / 2.0

    if not np.any(np.isfinite(angle_filtered)):
        return None, 0.0

    radii = np.arange(max(radius_min, 0.0), radius_max + radius_step, radius_step)
    best_mask = None
    used_radius = radius_max
    for radius in radii:
        radius_filter = center_abs.copy()
        radius_filter[radius_filter > radius] = np.nan
        temp1 = radius_filter / radius_filter
        temp1_count = float(np.nansum(temp1))

        temp2 = radius_filter * angle_filtered
        temp2_mask = temp2 / temp2
        temp2_count = float(np.nansum(temp2_mask))

        if temp1_count - temp2_count > expansion_threshold * radius:
            best_mask = np.nan_to_num(temp2_mask, nan=0.0).astype(bool)
            used_radius = radius
            break

    if best_mask is None:
        radius_filter = center_abs.copy()
        radius_filter[radius_filter > radius_max] = np.nan
        temp2 = radius_filter * angle_filtered
        best_mask = np.nan_to_num(temp2 / temp2, nan=0.0).astype(bool)
        used_radius = radius_max

    if not np.any(best_mask):
        cx_round = int(round(cx))
        cy_round = int(round(cy))
        if 0 <= cx_round < cols and 0 <= cy_round < rows:
            fallback_mask = np.zeros_like(best_mask, dtype=bool)
            fallback_mask[cy_round, cx_round] = True
            return fallback_mask, 0.0
        return None, 0.0

    return best_mask.astype(bool), used_radius


def apply_expanded_masks_to_detection(
    base_result: SpiralDetectionResult,
    pattern_masks: Dict[int, Dict[int, np.ndarray]],
    amplitude_field: np.ndarray,
    expansion_radii: Optional[Dict[int, Dict[int, float]]] = None,
) -> SpiralDetectionResult:
    """
    Replace pattern voxels/metrics with expanded masks while keeping IDs/durations.

    Parameters
    ----------
    base_result : SpiralDetectionResult
        Original detection result.
    pattern_masks : Dict[int, Dict[int, np.ndarray]]
        Per-pattern, per-time expanded masks.
    amplitude_field : np.ndarray
        Amplitude field for computing powers/centroids.
    expansion_radii : Optional[Dict[int, Dict[int, float]]]
        Per-pattern, per-time expansion radii. If provided, will be stored in pattern objects.

    Returns
    -------
    SpiralDetectionResult
        Updated result with expanded pattern data.
    """
    if base_result.num_patterns == 0 or not pattern_masks:
        return base_result

    rows, cols, frames = amplitude_field.shape
    new_labeled = np.zeros((rows, cols, frames), dtype=int)
    updated_patterns: list[SpiralPattern] = []

    for pattern in base_result.patterns:
        slices = pattern_masks.get(pattern.pattern_id)
        if not slices:
            continue

        duration = len(pattern.absolute_times)
        centroids = np.zeros_like(pattern.centroids)
        weighted_centroids = np.zeros_like(pattern.weighted_centroids)
        instantaneous_sizes = np.zeros_like(pattern.instantaneous_sizes)
        instantaneous_powers = np.zeros_like(pattern.instantaneous_powers)
        instantaneous_peak_amps = np.zeros_like(pattern.instantaneous_peak_amps)
        instantaneous_widths = np.zeros_like(pattern.instantaneous_widths)
        radii = np.zeros(duration, dtype=float)
        voxel_indices_list: list[np.ndarray] = []
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        # Get expansion radii for this pattern if available
        pattern_radii = expansion_radii.get(pattern.pattern_id, {}) if expansion_radii else {}

        for idx, abs_time in enumerate(pattern.absolute_times):
            mask_slice = slices.get(int(abs_time))
            if mask_slice is None:
                continue

            ys, xs = np.where(mask_slice)
            if ys.size == 0:
                continue

            # Store expansion radius if available
            if int(abs_time) in pattern_radii:
                radii[idx] = pattern_radii[int(abs_time)]

            instantaneous_sizes[idx] = ys.size
            amps = np.abs(amplitude_field[:, :, abs_time])
            weights = amps[ys, xs]
            instantaneous_powers[idx] = float(np.sum(weights))
            instantaneous_peak_amps[idx] = (
                float(np.max(weights)) if weights.size > 0 else 0.0
            )
            centroids[idx] = [float(xs.mean()), float(ys.mean())]
            if weights.size > 0 and float(np.sum(weights)) > 0:
                weighted_centroids[idx] = [
                    float(np.average(xs, weights=weights)),
                    float(np.average(ys, weights=weights)),
                ]
            else:
                weighted_centroids[idx] = centroids[idx]

            width = (
                (xs.max() - xs.min() + 1) + (ys.max() - ys.min() + 1)
            ) / 2.0
            instantaneous_widths[idx] = float(width)

            t_coords = np.full_like(xs, abs_time)
            indices = np.ravel_multi_index(
                (ys, xs, t_coords), dims=amplitude_field.shape
            )
            voxel_indices_list.append(indices)
            new_labeled[ys, xs, abs_time] = pattern.pattern_id
            all_x.append(xs)
            all_y.append(ys)

        if not voxel_indices_list:
            continue
        voxel_indices = np.concatenate(voxel_indices_list).astype(int, copy=False)
        total_size = int(np.sum(instantaneous_sizes))
        if total_size == 0:
            continue

        concat_x = np.concatenate(all_x)
        concat_y = np.concatenate(all_y)
        bbox = (
            int(concat_x.min()),
            int(concat_x.max()),
            int(concat_y.min()),
            int(concat_y.max()),
            pattern.start_time,
            pattern.end_time,
        )

        updated_patterns.append(
            SpiralPattern(
                pattern_id=pattern.pattern_id,
                duration=pattern.duration,
                start_time=pattern.start_time,
                end_time=pattern.end_time,
                absolute_times=pattern.absolute_times,
                total_size=total_size,
                centroids=centroids,
                weighted_centroids=weighted_centroids,
                instantaneous_sizes=instantaneous_sizes,
                instantaneous_powers=instantaneous_powers,
                instantaneous_peak_amps=instantaneous_peak_amps,
                instantaneous_widths=instantaneous_widths,
                bounding_box=bbox,
                rotation_direction=pattern.rotation_direction,
                curl_sign=pattern.curl_sign,
                voxel_indices=voxel_indices,
                expansion_radii=radii if expansion_radii else None,
                compatibility_ratios=None,  # Will be filled later by surrogate filtering
            )
        )

    num_patterns = len(updated_patterns)
    if num_patterns > 0:
        durations = np.array([p.duration for p in updated_patterns], dtype=float)
        sizes = np.array([p.total_size for p in updated_patterns], dtype=float)
        stats = {
            "total_patterns": num_patterns,
            "avg_duration": float(np.mean(durations)),
            "avg_size": float(np.mean(sizes)),
            "total_voxels": int(np.sum(sizes)),
            "max_duration": int(np.max(durations)),
            "max_size": int(np.max(sizes)),
        }
    else:
        stats = {
            "total_patterns": 0,
            "avg_duration": 0.0,
            "avg_size": 0.0,
            "total_voxels": 0,
            "max_duration": 0,
            "max_size": 0,
        }

    return SpiralDetectionResult(
        patterns=updated_patterns,
        num_patterns=num_patterns,
        labeled_volume=new_labeled,
        input_shape=base_result.input_shape,
        detection_params=base_result.detection_params,
        statistics=stats,
        rotation_direction=base_result.rotation_direction,
        curl_sign=base_result.curl_sign,
    )
