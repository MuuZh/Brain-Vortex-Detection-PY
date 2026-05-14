"""
MATLAB-style compatibility ratio calculation for spiral filtering.

This module implements the compatibility ratio metric used in the MATLAB pipeline
to filter detected spirals based on surrogate data. The compatibility ratio measures
how well voxels within a spiral conform to the ideal spiral pattern by checking:
1. Vector angle alignment: phase gradient vectors should be perpendicular (±45°)
   to radial vectors from the spiral center
2. Phase coherence: smoothed and unsmoothed phases should agree within π/6

The MATLAB code (spiral_detection_surfilt.m lines 1239-1491) calculates per-spiral,
per-frame compatibility ratios and builds radius-specific distributions from surrogate
data. Real spirals are then filtered by comparing their ratios to the 95th percentile
threshold for their specific radius.

References:
    matcode/spiral_detection_surfilt.m: Lines 1239-1650
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from matphase.detect.phase_field import angle_subtract


def compute_compatibility_ratio_for_frame(
    center_x: float,
    center_y: float,
    vx_frame: np.ndarray,
    vy_frame: np.ndarray,
    phase_alignment_mask: np.ndarray,
    expansion_radius: float,
    *,
    angle_threshold_low: float = np.pi / 2 - np.pi / 4,  # 45 degrees below 90
    angle_threshold_high: float = np.pi / 2 + np.pi / 4,  # 45 degrees above 90
    max_radius_bins: int = 180,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute compatibility ratio for a single spiral frame.

    This function replicates MATLAB lines 1239-1336 logic for calculating the
    compatibility ratio of voxels within a spiral's expansion radius.

    Parameters
    ----------
    center_x, center_y : float
        Spiral center coordinates (weighted centroid).
    vx_frame, vy_frame : np.ndarray
        Normalized phase gradient vectors at this frame, shape (rows, cols).
    phase_alignment_mask : np.ndarray
        Boolean mask where |raw_phase - smooth_phase| <= π/6, shape (rows, cols).
    expansion_radius : float
        The radius (in pixels) used for this spiral at this frame.
    angle_threshold_low, angle_threshold_high : float
        Bounds for acceptable angle difference between radial and phase vectors.
        MATLAB default: π/2 ± π/4 (90° ± 45°).
    max_radius_bins : int
        Maximum number of distance bins (MATLAB uses 0:1:180).

    Returns
    -------
    compatibility_ratio : float
        Ratio of compatible voxels to total voxels within expansion_radius.
    compatible_voxel_count_by_radius : np.ndarray
        Array of compatible voxel counts for each radius bin (length max_radius_bins).
    total_voxel_count_by_radius : np.ndarray
        Array of total voxel counts for each radius bin (length max_radius_bins).

    Notes
    -----
    - Compatible voxels must pass BOTH:
      1. Vector angle alignment (within angle thresholds)
      2. Phase coherence (phase_alignment_mask = True)
    - MATLAB code: spiral_detection_surfilt.m lines 1250-1336
    """
    rows, cols = vx_frame.shape

    # Create grid coordinates (MATLAB lines 1263-1268)
    y_grid, x_grid = np.indices((rows, cols), dtype=float)

    # Radial vectors from spiral center (MATLAB lines 1269-1275)
    center_vx = np.round(x_grid - center_x)
    center_vy = np.round(y_grid - center_y)
    distance_matrix = np.hypot(center_vx, center_vy)
    distance_matrix[distance_matrix == 0] = 1.0  # Avoid division by zero at center

    # Phase vector angles (MATLAB lines 1252-1254)
    phase_vector_angle = np.angle(vx_frame + 1j * vy_frame)

    # Radial vector angles (MATLAB line 1271)
    radial_vector_angle = np.angle(center_vx + 1j * center_vy)

    # Angle difference (MATLAB lines 1279-1291)
    angle_diff = radial_vector_angle - phase_vector_angle
    # Wrap to [-π, π]
    angle_diff = np.where(angle_diff > np.pi, angle_diff - 2 * np.pi, angle_diff)
    angle_diff = np.where(angle_diff < -np.pi, angle_diff + 2 * np.pi, angle_diff)
    angle_diff = np.abs(angle_diff)

    # Apply angle threshold filter (MATLAB lines 1298-1302)
    angle_filtered = angle_diff.copy()
    angle_filtered[(angle_filtered < angle_threshold_low) | (angle_filtered > angle_threshold_high)] = np.nan
    angle_filtered[~np.isfinite(phase_vector_angle)] = np.nan

    # Apply phase alignment mask (MATLAB line 1312: temp1_raw_phase_sur_filt_anglealign_2d)
    # Only voxels that pass BOTH angle alignment AND phase coherence are compatible
    phase_coherent_and_angle_aligned = np.isfinite(angle_filtered) & phase_alignment_mask

    # Compute per-radius bins (MATLAB lines 1315-1333)
    bins = np.arange(0, max_radius_bins + 1, 1)  # [0, 1, 2, ..., 180]
    compatible_voxel_count = np.zeros(max_radius_bins, dtype=float)
    total_voxel_count = np.zeros(max_radius_bins, dtype=float)

    for bin_id in range(len(bins) - 1):
        bin_min = bins[bin_id]
        bin_max = bins[bin_id + 1]

        # Voxels within this distance bin
        in_bin = (distance_matrix >= bin_min) & (distance_matrix < bin_max)

        # Total voxels in this radius bin
        total_voxel_count[bin_id] = np.sum(in_bin)

        # Compatible voxels (angle aligned AND phase coherent)
        compatible_voxel_count[bin_id] = np.sum(in_bin & phase_coherent_and_angle_aligned)

    # Compute compatibility ratio for voxels within expansion_radius
    # MATLAB lines 1543-1565: sum compatible/total for radius 0 to expansion_radius
    radius_limit = int(np.round(expansion_radius))
    radius_limit = min(radius_limit, max_radius_bins - 1)

    total_within_radius = np.sum(total_voxel_count[:radius_limit + 1])
    compatible_within_radius = np.sum(compatible_voxel_count[:radius_limit + 1])

    if total_within_radius > 0:
        compatibility_ratio = compatible_within_radius / total_within_radius
    else:
        compatibility_ratio = 0.0

    return compatibility_ratio, compatible_voxel_count, total_voxel_count


def build_surrogate_compatibility_distributions(
    surrogate_patterns_by_radius: Dict[int, List[float]],
    max_radius: int = 180,
) -> Dict[int, np.ndarray]:
    """
    Build radius-specific compatibility ratio distributions from surrogate spirals.

    This replicates MATLAB lines 1496-1532 where surrogate compatibility ratios
    are aggregated by radius to create per-radius distributions.

    Parameters
    ----------
    surrogate_patterns_by_radius : Dict[int, List[float]]
        Mapping of radius -> list of compatibility ratios from all surrogate spirals
        at that radius.
    max_radius : int
        Maximum radius to consider (MATLAB uses 180).

    Returns
    -------
    distributions : Dict[int, np.ndarray]
        Mapping of radius -> sorted array of compatibility ratios for computing percentiles.

    Notes
    -----
    - MATLAB code: spiral_detection_surfilt.m lines 1496-1532
    """
    distributions: Dict[int, np.ndarray] = {}

    for radius in range(1, max_radius + 1):
        ratios = surrogate_patterns_by_radius.get(radius, [])
        if ratios:
            distributions[radius] = np.array(sorted(ratios), dtype=float)
        else:
            # No surrogate data for this radius - use empty array
            distributions[radius] = np.array([], dtype=float)

    return distributions


def compute_percentile_threshold_by_radius(
    distributions: Dict[int, np.ndarray],
    percentile: float = 95.0,
) -> Dict[int, float]:
    """
    Compute percentile threshold for each radius from surrogate distributions.

    This replicates MATLAB lines 1555-1558 where the 95th percentile of surrogate
    compatibility ratios is computed for each radius.

    Parameters
    ----------
    distributions : Dict[int, np.ndarray]
        Mapping of radius -> sorted compatibility ratio array.
    percentile : float
        Percentile to compute (95.0 for p<0.05 threshold).

    Returns
    -------
    thresholds : Dict[int, float]
        Mapping of radius -> percentile threshold value.

    Notes
    -----
    - MATLAB code: spiral_detection_surfilt.m lines 1555-1558
    - If no surrogate data exists for a radius, threshold is set to 0.0
    """
    thresholds: Dict[int, float] = {}

    for radius, distribution in distributions.items():
        if len(distribution) > 0:
            # MATLAB: temp1_sur_sort_95perc = temp1_sur_sort(round(0.95 * size(temp1_sur_sort(:),1)))
            idx = int(np.round(percentile / 100.0 * len(distribution)))
            idx = min(idx, len(distribution) - 1)  # Ensure within bounds
            idx = max(idx, 0)
            thresholds[radius] = float(distribution[idx])
        else:
            # No surrogate data - permissive threshold
            thresholds[radius] = 0.0

    return thresholds


@dataclass
class SurrogateCompatibilityAccumulator:
    """
    Incrementally accumulate surrogate compatibility statistics.

    Allows the caller to feed patterns as they are processed (e.g., while
    iterating surrogates) and compute percentile thresholds at the end.
    """

    max_radius: int = 180
    _entries: List[Tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    _pattern_counter: int = 0

    def ingest_patterns(
        self,
        patterns,
        expansion_radii: Dict[int, Dict[int, float]],
        phase_field_vx: np.ndarray,
        phase_field_vy: np.ndarray,
        phase_alignment_mask: Optional[np.ndarray],
        *,
        show_progress: bool = False,
    ) -> None:
        """Add compatibility counts for the provided patterns."""
        if phase_alignment_mask is None:
            phase_alignment_mask = np.ones_like(phase_field_vx, dtype=bool)

        frame_bar = None
        if show_progress:
            total_frames = int(sum(len(getattr(p, "absolute_times", [])) for p in patterns))
            if total_frames > 0:
                frame_bar = tqdm(
                    total=total_frames,
                    desc="Surrogate compatibility",
                    unit="frame",
                    leave=False,
                )

        for pattern in patterns:
            pattern_radii = expansion_radii.get(pattern.pattern_id, {})

            for idx, abs_time in enumerate(pattern.absolute_times):
                abs_time_int = int(abs_time)
                radius = pattern_radii.get(abs_time_int, 0.0)
                if radius <= 0:
                    continue

                center = pattern.weighted_centroids[idx]
                if not np.all(np.isfinite(center)):
                    continue

                _ratio, compatible_counts, total_counts = compute_compatibility_ratio_for_frame(
                    center_x=center[0],
                    center_y=center[1],
                    vx_frame=phase_field_vx[:, :, abs_time_int],
                    vy_frame=phase_field_vy[:, :, abs_time_int],
                    phase_alignment_mask=phase_alignment_mask[:, :, abs_time_int],
                    expansion_radius=radius,
                )

                if total_counts.sum() == 0:
                    continue

                self._entries.append(
                    (self._pattern_counter, compatible_counts, total_counts)
                )
                self._pattern_counter += 1
                if frame_bar is not None:
                    frame_bar.update(1)

        if frame_bar is not None:
            frame_bar.close()

    def finalize(self, percentile: float = 95.0) -> Dict[int, float]:
        """Compute percentile thresholds from accumulated entries."""
        if not self._entries:
            return {}

        by_radius = accumulate_surrogate_compatibility_by_radius(
            self._entries,
            max_radius=self.max_radius,
        )
        distributions = build_surrogate_compatibility_distributions(
            by_radius,
            max_radius=self.max_radius,
        )
        return compute_percentile_threshold_by_radius(
            distributions,
            percentile=percentile,
        )


def filter_patterns_by_compatibility(
    patterns: List,  # List[SpiralPattern]
    compatibility_ratios: Dict[int, Dict[int, float]],  # pattern_id -> abs_time -> ratio
    expansion_radii: Dict[int, Dict[int, float]],  # pattern_id -> abs_time -> radius
    surrogate_thresholds: Dict[int, float],  # radius -> threshold
    *,
    show_progress: bool = False,
) -> Dict[int, Dict[int, bool]]:
    """
    Filter spiral patterns by comparing their compatibility ratios to surrogate thresholds.

    This replicates MATLAB lines 1535-1587 where each spiral frame is compared to
    the 95th percentile threshold for its specific radius.

    Parameters
    ----------
    patterns : List[SpiralPattern]
        Detected spiral patterns.
    compatibility_ratios : Dict[int, Dict[int, float]]
        Nested dict: pattern_id -> abs_time -> compatibility_ratio.
    expansion_radii : Dict[int, Dict[int, float]]
        Nested dict: pattern_id -> abs_time -> expansion_radius.
    surrogate_thresholds : Dict[int, float]
        Mapping of radius -> 95th percentile threshold from surrogates.
    show_progress : bool
        If True, show tqdm progress bar.

    Returns
    -------
    passes_filter : Dict[int, Dict[int, bool]]
        Nested dict: pattern_id -> abs_time -> whether frame passes filter.

    Notes
    -----
    - MATLAB code: spiral_detection_surfilt.m lines 1541-1587
    - A frame passes if: compatibility_ratio > threshold[radius]
    """
    passes_filter: Dict[int, Dict[int, bool]] = {}

    iterator = patterns
    if show_progress and len(patterns) > 10:
        iterator = tqdm(patterns, desc="Filtering by compatibility", unit="pattern")

    for pattern in iterator:
        pattern_id = pattern.pattern_id
        passes_filter[pattern_id] = {}

        pattern_ratios = compatibility_ratios.get(pattern_id, {})
        pattern_radii = expansion_radii.get(pattern_id, {})

        for abs_time in pattern.absolute_times:
            abs_time_int = int(abs_time)
            ratio = pattern_ratios.get(abs_time_int)
            radius = pattern_radii.get(abs_time_int)

            if ratio is None or radius is None:
                # Missing data - mark as failed
                passes_filter[pattern_id][abs_time_int] = False
                continue

            # Get threshold for this radius (MATLAB lines 1555-1558)
            radius_int = int(np.round(radius))
            threshold = surrogate_thresholds.get(radius_int, 0.0)

            # MATLAB line 1561: if temp1_compatibility_ratio > temp1_sur_sort_95perc
            passes_filter[pattern_id][abs_time_int] = (ratio > threshold)

    return passes_filter


def accumulate_surrogate_compatibility_by_radius(
    surrogate_compatible_counts: List[Tuple[int, np.ndarray, np.ndarray]],  # [(pattern_id, compatible_counts, total_counts), ...]
    max_radius: int = 180,
) -> Dict[int, List[float]]:
    """
    Accumulate compatibility ratios from all surrogate spirals, organized by radius.

    This prepares data for building radius-specific distributions.

    Parameters
    ----------
    surrogate_compatible_counts : List[Tuple[int, np.ndarray, np.ndarray]]
        List of (pattern_id, compatible_voxel_count_by_radius, total_voxel_count_by_radius)
        tuples from all surrogate spirals.
    max_radius : int
        Maximum radius to track.

    Returns
    -------
    by_radius : Dict[int, List[float]]
        Mapping of radius -> list of compatibility ratios from all surrogate spirals.
    """
    by_radius: Dict[int, List[float]] = {r: [] for r in range(1, max_radius + 1)}

    for _, compatible_counts, total_counts in surrogate_compatible_counts:
        for radius in range(1, min(max_radius + 1, len(compatible_counts))):
            # Cumulative sum up to this radius (MATLAB lines 1500-1530)
            total_up_to_radius = np.sum(total_counts[:radius])
            compatible_up_to_radius = np.sum(compatible_counts[:radius])

            if total_up_to_radius > 0:
                ratio = compatible_up_to_radius / total_up_to_radius
                by_radius[radius].append(ratio)

    return by_radius


def compute_surrogate_compatibility_thresholds(
    surrogate_detection_func,
    n_surrogates: int,
    percentile: float = 95.0,
    max_radius: int = 180,
    *,
    show_progress: bool = False,
) -> Dict[int, float]:
    """
    Compute compatibility ratio thresholds from surrogate spiral detections.

    This is the main entry point for computing surrogate-based filtering thresholds.
    It takes a function that generates surrogate spirals and returns radius-specific
    95th percentile thresholds.

    Parameters
    ----------
    surrogate_detection_func : callable
        A function that yields (expansion_radii_map, phase_field) tuples for each
        surrogate realization. The function should detect spirals and return the
        expansion radii and phase field data needed for compatibility calculation.
    n_surrogates : int
        Number of surrogate realizations to process.
    percentile : float
        Percentile to use for thresholding (default 95.0).
    max_radius : int
        Maximum radius to track (default 180).
    show_progress : bool
        If True, show tqdm progress bar.

    Returns
    -------
    Dict[int, float]
        Mapping of radius (pixels) to percentile threshold value.

    Examples
    --------
    >>> def surrogate_detector():
    ...     for i in range(100):
    ...         # Generate surrogate, detect spirals, yield results
    ...         yield expansion_radii, normalized_vx, normalized_vy, phase_mask
    >>> thresholds = compute_surrogate_compatibility_thresholds(
    ...     surrogate_detector(), n_surrogates=100, percentile=95.0
    ... )
    """
    accumulator = SurrogateCompatibilityAccumulator(max_radius=max_radius)

    iterator = surrogate_detection_func
    if show_progress:
        iterator = tqdm(
            iterator,
            total=n_surrogates,
            desc="Computing surrogate compatibility",
            unit="surrogate",
        )

    for surrogate_data in iterator:
        if len(surrogate_data) != 5:
            raise ValueError(
                f"Expected 5 elements from surrogate_detection_func, got {len(surrogate_data)}"
            )

        patterns, expansion_radii_map, vx, vy, phase_alignment_mask = surrogate_data
        accumulator.ingest_patterns(
            patterns=patterns,
            expansion_radii=expansion_radii_map,
            phase_field_vx=vx,
            phase_field_vy=vy,
            phase_alignment_mask=phase_alignment_mask,
        )

    return accumulator.finalize(percentile=percentile)


def apply_compatibility_ratios_to_patterns(
    patterns: List,  # List[SpiralPattern]
    phase_field_vx: np.ndarray,
    phase_field_vy: np.ndarray,
    phase_alignment_mask: Optional[np.ndarray],
    *,
    show_progress: bool = False,
) -> List:
    """
    Calculate and store compatibility ratios for a list of spiral patterns.

    Parameters
    ----------
    patterns : List[SpiralPattern]
        Detected spiral patterns with expansion_radii filled.
    phase_field_vx, phase_field_vy : np.ndarray
        Normalized phase gradient vector fields, shape (rows, cols, frames).
    phase_alignment_mask : Optional[np.ndarray]
        Phase coherence mask, shape (rows, cols, frames).
    show_progress : bool
        If True, show tqdm progress bar.

    Returns
    -------
    List[SpiralPattern]
        Patterns with compatibility_ratios filled.
    """
    updated_patterns = []

    pattern_iter = patterns
    if show_progress and len(patterns) > 10:
        pattern_iter = tqdm(pattern_iter, desc="Computing compatibility ratios", unit="pattern")

    for pattern in pattern_iter:
        if pattern.expansion_radii is None:
            updated_patterns.append(pattern)
            continue

        compatibility_ratios = np.zeros(pattern.duration, dtype=float)
        compatibility_passes = np.zeros(pattern.duration, dtype=bool)

        for idx, abs_time in enumerate(pattern.absolute_times):
            center = pattern.weighted_centroids[idx]
            radius = pattern.expansion_radii[idx]

            if radius > 0 and np.all(np.isfinite(center)):
                try:
                    ratio, _, _ = compute_compatibility_ratio_for_frame(
                        center_x=center[0],
                        center_y=center[1],
                        vx_frame=phase_field_vx[:, :, abs_time],
                        vy_frame=phase_field_vy[:, :, abs_time],
                        phase_alignment_mask=(
                            phase_alignment_mask[:, :, abs_time]
                            if phase_alignment_mask is not None
                            else np.ones_like(phase_field_vx[:, :, abs_time], dtype=bool)
                        ),
                        expansion_radius=radius,
                    )
                    compatibility_ratios[idx] = ratio
                    compatibility_passes[idx] = ratio > 0.0
                except Exception:
                    compatibility_ratios[idx] = 0.0
                    compatibility_passes[idx] = False
            else:
                compatibility_ratios[idx] = 0.0
                compatibility_passes[idx] = False

        # Create a new pattern with the dataclass replace method would be cleaner,
        # but since we need to update the compatibility_ratios field, we'll create a new instance
        # Copy all fields and update compatibility_ratios
        from dataclasses import replace
        try:
            updated_pattern = replace(
                pattern,
                compatibility_ratios=compatibility_ratios,
                compatibility_passes=compatibility_passes,
            )
        except TypeError:
            # If replace doesn't work (older Python), manually create new instance
            updated_pattern = type(pattern)(
                **{
                    **pattern.__dict__,
                    "compatibility_ratios": compatibility_ratios,
                    "compatibility_passes": compatibility_passes,
                }
            )

        updated_patterns.append(updated_pattern)

    return updated_patterns
