"""
Spiral pattern detection using connected components analysis.

This module implements spatiotemporal pattern detection for spiral waves in
phase field data, translating the MATLAB pattDetection_v4.m logic to Python.

Key features:
- Connected components detection in 3D (x, y, time) using scipy.ndimage
- Temporal and spatial thresholding (minimum duration and size)
- Geometric and amplitude-weighted centroid computation
- Comprehensive pattern metadata extraction (duration, size, power, etc.)
- Progress tracking with tqdm for large datasets
- Device-agnostic design (NumPy baseline, extensible to GPU)

MATLAB Parity:
- Connected components: bwconncomp → scipy.ndimage.label
- Bounding boxes: regionprops → scipy.ndimage.find_objects
- Weighted centroids: regionprops(...,'WeightedCentroid') → scipy.ndimage.center_of_mass

References:
    matcode/pattDetection_v4.m: Original MATLAB implementation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import warnings

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from matphase.detect.thresholds import apply_curl_threshold


@dataclass
class SpiralPattern:
    """
    Metadata for a single detected spiral pattern.

    Attributes:
        pattern_id: Unique identifier for this pattern
        duration: Temporal duration in frames
        start_time: First frame index (0-based)
        end_time: Last frame index (0-based)
        absolute_times: Array of frame indices where pattern appears
        total_size: Total voxel count across all timepoints
        centroids: Geometric centroids at each timepoint [(x, y), ...]
        weighted_centroids: Amplitude-weighted centroids at each timepoint [(x, y), ...]
        instantaneous_sizes: Spatial extent (voxel count) at each timepoint
        instantaneous_powers: Total power (sum of absolute values) at each timepoint
        instantaneous_peak_amps: Peak amplitude at each timepoint
        instantaneous_widths: Average bounding box dimension at each timepoint
        bounding_box: 3D bounding box (x_min, x_max, y_min, y_max, t_min, t_max)
        rotation_direction: Rotation descriptor ('cw', 'ccw', 'unspecified')
        curl_sign: Sign of curl associated with this pattern (+1=ccw, -1=cw, None=unknown)
        voxel_indices: Flat indices of voxels belonging to this pattern (for reconstruction)
        expansion_radii: Expansion radius used at each timepoint (shape: duration,)
        compatibility_ratios: Compatibility ratio at each timepoint (shape: duration,), None if not computed
    """
    pattern_id: int
    duration: int
    start_time: int
    end_time: int
    absolute_times: np.ndarray
    total_size: int
    centroids: np.ndarray  # Shape: (duration, 2)
    weighted_centroids: np.ndarray  # Shape: (duration, 2)
    instantaneous_sizes: np.ndarray  # Shape: (duration,)
    instantaneous_powers: np.ndarray  # Shape: (duration,)
    instantaneous_peak_amps: np.ndarray  # Shape: (duration,)
    instantaneous_widths: np.ndarray  # Shape: (duration,)
    bounding_box: Tuple[int, int, int, int, int, int]
    rotation_direction: str
    curl_sign: Optional[int]
    voxel_indices: np.ndarray = field(repr=False)  # Don't print in repr
    expansion_radii: Optional[np.ndarray] = None  # Shape: (duration,)
    compatibility_ratios: Optional[np.ndarray] = None  # Shape: (duration,)
    compatibility_passes: Optional[np.ndarray] = None  # Shape: (duration,), bool mask


@dataclass
class SpiralDetectionResult:
    """
    Complete result of spiral detection on a 3D spatiotemporal volume.

    Attributes:
        patterns: List of detected SpiralPattern objects
        num_patterns: Number of detected patterns
        labeled_volume: 3D array with pattern labels (0 = background, 1+ = pattern ID)
        input_shape: Shape of input volume (height, width, time)
        detection_params: Dictionary of detection parameters used
        statistics: Summary statistics (total patterns, avg duration, etc.)
        rotation_direction: Rotation direction for this detection pass (or "mixed")
        curl_sign: Curl sign associated with detection (None if mixed/unspecified)
    """
    patterns: List[SpiralPattern]
    num_patterns: int
    labeled_volume: np.ndarray
    input_shape: Tuple[int, int, int]
    detection_params: Dict
    statistics: Dict
    rotation_direction: Optional[str]
    curl_sign: Optional[int]


def detect_spirals(
    signal_binary: np.ndarray,
    signal_amplitude: Optional[np.ndarray] = None,
    min_duration: int = 1,
    min_size: int = 3,
    connectivity: int = 6,
    use_weighted_centroids: bool = True,
    show_progress: bool = False,
    rotation_direction: Optional[str] = None,
    curl_sign: Optional[int] = None,
    pattern_id_offset: int = 0,
) -> SpiralDetectionResult:
    """
    Detect spatiotemporal spiral patterns using connected components analysis.

    This is the main entry point for pattern detection, translating MATLAB's
    pattDetection_v4.m to Python. It identifies contiguous regions in a 3D
    binary volume (x, y, time), filters by temporal/spatial thresholds, and
    extracts comprehensive metadata for each pattern.

    Parameters:
        signal_binary: Binary 3D array (height, width, time) marking spiral regions
        signal_amplitude: Optional amplitude data for weighted centroids and power
                         If None, uses signal_binary for all calculations
        min_duration: Minimum pattern duration in frames (MATLAB: params.minPattTime)
        min_size: Minimum spatial extent in pixels (MATLAB: params.minPattSize)
        connectivity: Connectivity for 3D labeling (6=face, 18=edge, 26=vertex)
        use_weighted_centroids: If True, compute amplitude-weighted centroids
        show_progress: If True, show tqdm progress bars
        rotation_direction: Optional label ('cw' for clockwise, 'ccw' for counterclockwise)
            describing the curl orientation represented by signal_binary. If None, defaults
            to 'unspecified'.
        curl_sign: Optional curl sign (+1=ccw, -1=cw). If provided, must match
            rotation_direction. When None, inferred from rotation_direction.
        pattern_id_offset: Integer offset applied to assigned pattern IDs. Useful when
            concatenating results from multiple orientation passes.

    Returns:
        SpiralDetectionResult containing all detected patterns and metadata

    Notes:
        - MATLAB uses 1-based indexing; Python uses 0-based (frame 0.5 → frame 0)
        - Pattern ordering is deterministic (sorted by first appearance time)
        - NaN values in signal_amplitude are handled gracefully (ignored in sums)

    Examples:
        >>> # Detect spirals in curl field (negative curl = clockwise spirals)
        >>> curl_binary = curl < -threshold
        >>> result = detect_spirals(curl_binary, curl, min_duration=2, min_size=5)
        >>> print(f"Detected {result.num_patterns} spirals")
        >>> for pattern in result.patterns:
        ...     print(f"Pattern {pattern.pattern_id}: duration={pattern.duration}")
    """
    # Input validation
    if signal_binary.ndim != 3:
        raise ValueError(f"signal_binary must be 3D, got shape {signal_binary.shape}")

    if signal_amplitude is None:
        signal_amplitude = signal_binary.astype(float)
    else:
        if signal_amplitude.shape != signal_binary.shape:
            raise ValueError(
                f"Shape mismatch: signal_binary {signal_binary.shape} "
                f"vs signal_amplitude {signal_amplitude.shape}"
            )
        # Replace NaN/Inf amplitudes with zeros to keep centroid math finite
        signal_amplitude = np.nan_to_num(
            signal_amplitude,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    rotation_label, normalized_curl_sign = _normalize_rotation_inputs(
        rotation_direction, curl_sign
    )

    if pattern_id_offset < 0:
        raise ValueError("pattern_id_offset must be >= 0")

    # Get connectivity structure for scipy.ndimage.label
    # MATLAB bwconncomp default is 6-connectivity (face neighbors only)
    structure = _get_connectivity_structure(connectivity)

    # Step 1: Connected components labeling (MATLAB: bwconncomp)
    labeled_volume, num_components = ndimage.label(signal_binary, structure=structure)

    if num_components == 0:
        detection_params = {
            "min_duration": min_duration,
            "min_size": min_size,
            "connectivity": connectivity,
            "use_weighted_centroids": use_weighted_centroids,
            "pattern_id_offset": pattern_id_offset,
            "rotation_direction": rotation_label,
            "curl_sign": normalized_curl_sign,
        }
        return _create_detection_result(
            [],
            signal_binary.shape,
            detection_params,
            rotation_label,
            normalized_curl_sign,
        )

    # Step 2: Extract bounding boxes and filter by thresholds
    # MATLAB: regionprops(CC, 'BoundingBox', 'Area')
    slices = ndimage.find_objects(labeled_volume)
    height, width, n_times = signal_binary.shape

    patterns = []
    valid_labels = []

    # Create weighted signal for centroid computation (MATLAB: sigPlot = sigBinary .* sigIn)
    signal_weighted = signal_binary.astype(float) * signal_amplitude

    # Progress bar setup
    iterator = range(1, num_components + 1)
    if show_progress and num_components > 10:
        iterator = tqdm(iterator, desc="Filtering patterns", unit="pattern")

    for label_id in iterator:
        if slices[label_id - 1] is None:  # Empty component (shouldn't happen)
            continue

        bbox = slices[label_id - 1]

        # Extract bounding box dimensions (MATLAB: boundPatt)
        # bbox format: (slice_y, slice_x, slice_t)
        y_slice, x_slice, t_slice = bbox

        # Get start/end indices (0-based Python indexing)
        x_min, x_max = x_slice.start, x_slice.stop
        y_min, y_max = y_slice.start, y_slice.stop
        t_min, t_max = t_slice.start, t_slice.stop

        # Duration and spatial extent
        duration = t_max - t_min
        spatial_height = y_max - y_min
        spatial_width = x_max - x_min

        # MATLAB thresholds (lines 43-48 in pattDetection_v4.m)
        # Temporal threshold
        if duration < min_duration:
            continue

        # Spatial threshold (both dimensions must meet minimum)
        if spatial_height < min_size or spatial_width < min_size:
            continue

        # Pattern passed filters - mark for detailed extraction
        valid_labels.append(label_id)

    # Step 3: Extract detailed metadata for valid patterns
    # MATLAB: Loop lines 33-111 in pattDetection_v4.m

    # Progress bar for detailed extraction
    iterator = valid_labels
    if show_progress and len(valid_labels) > 10:
        iterator = tqdm(iterator, desc="Extracting pattern metadata", unit="pattern")

    for pattern_idx, label_id in enumerate(iterator, start=1):
        bbox = slices[label_id - 1]
        y_slice, x_slice, t_slice = bbox

        # Get pattern mask and indices
        pattern_mask = labeled_volume == label_id
        pattern_indices = np.where(pattern_mask)
        total_size = len(pattern_indices[0])

        # Time range (MATLAB: pattTimeStart/End, note MATLAB uses 1-based + 0.5 offset)
        t_min, t_max = t_slice.start, t_slice.stop
        duration = t_max - t_min
        absolute_times = np.arange(t_min, t_max, dtype=int)

        # Initialize per-timepoint arrays
        centroids = np.zeros((duration, 2))
        weighted_centroids = np.zeros((duration, 2))
        instantaneous_sizes = np.zeros(duration)
        instantaneous_powers = np.zeros(duration)
        instantaneous_peak_amps = np.zeros(duration)
        instantaneous_widths = np.zeros(duration)

        # Loop through each timepoint within pattern duration
        for time_idx, abs_time in enumerate(absolute_times):
            # Extract 2D slice at this timepoint (MATLAB: instantBinary)
            instant_binary = pattern_mask[:, :, abs_time]
            instant_pattern = signal_weighted[:, :, abs_time] * instant_binary

            # Instantaneous size (MATLAB: instantScale)
            inst_size = np.sum(instant_binary)
            instantaneous_sizes[time_idx] = inst_size

            if inst_size == 0:  # Empty slice (shouldn't happen for valid patterns)
                warnings.warn(
                    f"Pattern {pattern_idx} has empty slice at time {abs_time}"
                )
                continue

            # Instantaneous power (MATLAB: instantTotalPower)
            # Sum of absolute amplitudes
            instantaneous_powers[time_idx] = np.nansum(np.abs(instant_pattern))

            # Peak amplitude (MATLAB: instantPeakAmp)
            instantaneous_peak_amps[time_idx] = np.nanmax(np.abs(instant_pattern))

            # Geometric centroid (MATLAB: S.Centroid)
            # scipy uses (y, x) order, we want (x, y)
            y_cent, x_cent = ndimage.center_of_mass(instant_binary)
            centroids[time_idx] = [x_cent, y_cent]

            # Weighted centroid (MATLAB: S.WeightedCentroid)
            if use_weighted_centroids and instantaneous_powers[time_idx] > 0:
                # Use absolute values for weighting
                weights = np.abs(instant_pattern)
                y_wcent, x_wcent = ndimage.center_of_mass(weights)
                weighted_centroids[time_idx] = [x_wcent, y_wcent]
            else:
                weighted_centroids[time_idx] = centroids[time_idx]

            # Instantaneous width (MATLAB: width, average of bounding box dimensions)
            # Find bounding box of instant pattern
            rows, cols = np.where(instant_binary)
            if len(rows) > 0:
                bbox_height = rows.max() - rows.min() + 1
                bbox_width = cols.max() - cols.min() + 1
                instantaneous_widths[time_idx] = (bbox_height + bbox_width) / 2.0
            else:
                instantaneous_widths[time_idx] = 0.0

        # Create SpiralPattern object
        global_pattern_id = pattern_id_offset + pattern_idx

        pattern = SpiralPattern(
            pattern_id=global_pattern_id,
            duration=duration,
            start_time=t_min,
            end_time=t_max - 1,  # Inclusive end time
            absolute_times=absolute_times,
            total_size=total_size,
            centroids=centroids,
            weighted_centroids=weighted_centroids,
            instantaneous_sizes=instantaneous_sizes,
            instantaneous_powers=instantaneous_powers,
            instantaneous_peak_amps=instantaneous_peak_amps,
            instantaneous_widths=instantaneous_widths,
            bounding_box=(
                x_slice.start, x_slice.stop - 1,
                y_slice.start, y_slice.stop - 1,
                t_slice.start, t_slice.stop - 1
            ),
            rotation_direction=rotation_label,
            curl_sign=normalized_curl_sign,
            voxel_indices=np.ravel_multi_index(pattern_indices, signal_binary.shape)
        )
        patterns.append(pattern)

    detection_params = {
        "min_duration": min_duration,
        "min_size": min_size,
        "connectivity": connectivity,
        "use_weighted_centroids": use_weighted_centroids,
        "pattern_id_offset": pattern_id_offset,
        "rotation_direction": rotation_label,
        "curl_sign": normalized_curl_sign,
    }

    return _create_detection_result(
        patterns,
        signal_binary.shape,
        detection_params,
        rotation_label,
        normalized_curl_sign,
    )


def detect_spirals_directional(
    curl_field: np.ndarray,
    signal_amplitude: Optional[np.ndarray] = None,
    curl_threshold: float = 1.0,
    rotation_mode: Literal["both", "ccw", "cw"] = "both",
    min_duration: int = 1,
    min_size: int = 3,
    connectivity: int = 6,
    use_weighted_centroids: bool = True,
    show_progress: bool = False,
) -> SpiralDetectionResult:
    """
    Convenience wrapper that thresholds curl by rotation direction and concatenates detections.

    Parameters:
        curl_field: 3D curl array (height, width, time)
        signal_amplitude: Optional amplitude cube used for weighted centroids/power. Defaults
            to curl_field if None.
        curl_threshold: Threshold applied to |curl| before connected-component labeling.
        rotation_mode: 'ccw', 'cw', or 'both'. When 'both', directions are processed
            separately to avoid merging opposite rotations, and IDs remain unique.
        Other parameters mirror detect_spirals.
    """
    if curl_field.ndim != 3:
        raise ValueError(f"curl_field must be 3D, got shape {curl_field.shape}")

    amplitude = signal_amplitude if signal_amplitude is not None else curl_field
    directions = _expand_rotation_mode(rotation_mode)
    combined_patterns: List[SpiralPattern] = []
    current_offset = 0

    for direction in directions:
        threshold_result = apply_curl_threshold(
            curl_field,
            threshold=curl_threshold,
            absolute=False,
            return_binary=True,
            rotation_mode=direction,
        )
        binary_mask = threshold_result.binary_mask
        if not np.any(binary_mask):
            continue

        direction_result = detect_spirals(
            binary_mask,
            amplitude,
            min_duration=min_duration,
            min_size=min_size,
            connectivity=connectivity,
            use_weighted_centroids=use_weighted_centroids,
            show_progress=show_progress,
            rotation_direction=direction,
            curl_sign=1 if direction == "ccw" else -1,
            pattern_id_offset=current_offset,
        )
        combined_patterns.extend(direction_result.patterns)
        current_offset += direction_result.num_patterns

    detected_dirs = {p.rotation_direction for p in combined_patterns}
    if len(detected_dirs) == 1:
        rotation_label = detected_dirs.pop()
    elif len(detected_dirs) == 0:
        rotation_label = "unspecified"
    else:
        rotation_label = "mixed"

    summary_curl = None
    if rotation_label == "ccw":
        summary_curl = 1
    elif rotation_label == "cw":
        summary_curl = -1

    detection_params = {
        "min_duration": min_duration,
        "min_size": min_size,
        "connectivity": connectivity,
        "use_weighted_centroids": use_weighted_centroids,
        "pattern_id_offset": 0,
        "rotation_mode": list(directions),
        "curl_threshold": curl_threshold,
    }

    return _create_detection_result(
        combined_patterns,
        curl_field.shape,
        detection_params,
        rotation_label,
        summary_curl,
    )


def detect_spirals_from_masks(
    masks: Dict[str, np.ndarray],
    *,
    signal_amplitude: Optional[np.ndarray] = None,
    min_duration: int = 1,
    min_size: int = 3,
    connectivity: int = 6,
    use_weighted_centroids: bool = True,
) -> SpiralDetectionResult:
    """
    Detect spirals directly from pre-computed binary masks.

    Parameters
    ----------
    masks : Dict[str, np.ndarray]
        Mapping of rotation label ("ccw"/"cw") to binary volumes.
    signal_amplitude : np.ndarray, optional
        Amplitude cube used for weighted centroids/power (defaults to mask).
    min_duration, min_size, connectivity, use_weighted_centroids : see detect_spirals().

    Returns
    -------
    SpiralDetectionResult
        Combined detection result across provided masks.
    """
    combined_patterns: List[SpiralPattern] = []
    pattern_offset = 0
    detected_dirs: List[str] = []
    input_shape = None

    for direction in ("ccw", "cw"):
        mask = masks.get(direction)
        if mask is None:
            continue
        if mask.ndim != 3:
            raise ValueError(
                f"Binary mask for {direction} must be 3D, got shape {mask.shape}"
            )
        if input_shape is None:
            input_shape = mask.shape
        elif input_shape != mask.shape:
            raise ValueError(
                f"All masks must share the same shape: {input_shape} != {mask.shape}"
            )

        if not np.any(mask):
            continue

        result = detect_spirals(
            mask.astype(bool),
            signal_amplitude=signal_amplitude,
            min_duration=min_duration,
            min_size=min_size,
            connectivity=connectivity,
            use_weighted_centroids=use_weighted_centroids,
            show_progress=False,
            rotation_direction=direction,
            curl_sign=1 if direction == "ccw" else -1,
            pattern_id_offset=pattern_offset,
        )
        if result.num_patterns > 0:
            detected_dirs.append(direction)
            combined_patterns.extend(result.patterns)
            pattern_offset += result.num_patterns

    if input_shape is None:
        raise ValueError("At least one binary mask must be provided to detect spirals.")

    if len(set(detected_dirs)) == 1:
        rotation_label = detected_dirs[0] if detected_dirs else "unspecified"
    elif detected_dirs:
        rotation_label = "mixed"
    else:
        rotation_label = "unspecified"

    summary_curl = None
    if rotation_label == "ccw":
        summary_curl = 1
    elif rotation_label == "cw":
        summary_curl = -1

    detection_params = {
        "min_duration": min_duration,
        "min_size": min_size,
        "connectivity": connectivity,
        "use_weighted_centroids": use_weighted_centroids,
        "pattern_id_offset": 0,
        "rotation_mode": detected_dirs or [],
        "curl_threshold": None,
    }

    return _create_detection_result(
        combined_patterns,
        input_shape,
        detection_params,
        rotation_label,
        summary_curl,
    )


def _create_detection_result(
    patterns: List[SpiralPattern],
    input_shape: Tuple[int, int, int],
    detection_params: Dict,
    rotation_direction: Optional[str],
    curl_sign: Optional[int],
    extra_statistics: Optional[Dict[str, float]] = None,
) -> SpiralDetectionResult:
    """Compose SpiralDetectionResult with relabeled volume and summary stats."""
    final_labeled_volume = np.zeros(input_shape, dtype=int)
    for pattern in patterns:
        mask = np.unravel_index(pattern.voxel_indices, input_shape)
        final_labeled_volume[mask] = pattern.pattern_id

    num_patterns = len(patterns)
    if num_patterns > 0:
        durations = np.array([p.duration for p in patterns], dtype=float)
        sizes = np.array([p.total_size for p in patterns], dtype=float)
        statistics = {
            "total_patterns": num_patterns,
            "avg_duration": float(np.mean(durations)),
            "avg_size": float(np.mean(sizes)),
            "total_voxels": int(np.sum(sizes)),
            "max_duration": int(np.max(durations)),
            "max_size": int(np.max(sizes)),
        }
    else:
        statistics = {
            "total_patterns": 0,
            "avg_duration": 0.0,
            "avg_size": 0.0,
            "total_voxels": 0,
            "max_duration": 0,
            "max_size": 0,
        }

    if extra_statistics:
        statistics.update(extra_statistics)

    return SpiralDetectionResult(
        patterns=patterns,
        num_patterns=num_patterns,
        labeled_volume=final_labeled_volume,
        input_shape=input_shape,
        detection_params=detection_params,
        statistics=statistics,
        rotation_direction=rotation_direction,
        curl_sign=curl_sign,
    )


def _normalize_rotation_inputs(
    rotation_direction: Optional[str],
    curl_sign: Optional[int],
) -> Tuple[str, Optional[int]]:
    """Validate and reconcile rotation labels and curl signs."""
    normalized_direction = (
        rotation_direction.lower()
        if isinstance(rotation_direction, str)
        else None
    )

    if normalized_direction is not None and normalized_direction not in {
        "cw",
        "ccw",
        "unspecified",
        "mixed",
    }:
        raise ValueError(
            "rotation_direction must be 'cw', 'ccw', 'unspecified', or 'mixed'"
        )

    if curl_sign is not None and curl_sign not in (-1, 1):
        raise ValueError("curl_sign must be -1, 1, or None")

    if normalized_direction in {"cw", "ccw"} and curl_sign is None:
        curl_sign = 1 if normalized_direction == "ccw" else -1

    if curl_sign in (-1, 1) and normalized_direction is None:
        normalized_direction = "ccw" if curl_sign > 0 else "cw"

    if normalized_direction in {"cw", "ccw"} and curl_sign in (-1, 1):
        expected = 1 if normalized_direction == "ccw" else -1
        if curl_sign != expected:
            raise ValueError(
                "rotation_direction and curl_sign refer to conflicting orientations"
            )

    if normalized_direction is None:
        normalized_direction = "unspecified"

    if normalized_direction in {"unspecified", "mixed"}:
        curl_sign = None

    return normalized_direction, curl_sign


def _expand_rotation_mode(rotation_mode: Literal["both", "ccw", "cw"]) -> Tuple[str, ...]:
    """Normalize rotation mode into an ordered tuple of direction labels."""
    if rotation_mode == "both":
        return ("ccw", "cw")
    if rotation_mode in {"ccw", "cw"}:
        return (rotation_mode,)
    raise ValueError("rotation_mode must be 'both', 'ccw', or 'cw'")


def _get_connectivity_structure(connectivity: int) -> Optional[np.ndarray]:
    """
    Generate connectivity structure for scipy.ndimage.label.

    Parameters:
        connectivity: 6 (face), 18 (edge+face), or 26 (vertex+edge+face)

    Returns:
        Structure array for scipy.ndimage.label, or None for default connectivity

    Notes:
        - MATLAB bwconncomp default is 6-connectivity (faces only)
        - scipy.ndimage.label default is full connectivity (26 in 3D)
        - We explicitly define structures for reproducibility
    """
    if connectivity == 6:
        # Face connectivity only (3×3×3 with 6 neighbors)
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, 1] = True  # Center
        structure[1, 1, 0] = True  # Front
        structure[1, 1, 2] = True  # Back
        structure[1, 0, 1] = True  # Top
        structure[1, 2, 1] = True  # Bottom
        structure[0, 1, 1] = True  # Left
        structure[2, 1, 1] = True  # Right
        return structure
    elif connectivity == 18:
        # Edge + face connectivity (3×3×3 with 18 neighbors)
        structure = ndimage.generate_binary_structure(3, 1)  # Rank 1 = edges
        return structure
    elif connectivity == 26:
        # Full connectivity (3×3×3 with 26 neighbors)
        structure = ndimage.generate_binary_structure(3, 2)  # Rank 2 = vertices
        return structure
    else:
        raise ValueError(
            f"connectivity must be 6, 18, or 26, got {connectivity}"
        )


def filter_patterns_by_curl_strength(
    detection_result: SpiralDetectionResult,
    curl_field: np.ndarray,
    min_avg_curl: float,
    use_absolute: bool = True,
) -> SpiralDetectionResult:
    """
    Filter detected patterns by average curl strength.

    This post-processing step removes weak patterns that don't meet a minimum
    curl threshold. Useful for focusing on strong spirals.

    Parameters:
        detection_result: Result from detect_spirals()
        curl_field: Original curl field (same shape as input to detect_spirals)
        min_avg_curl: Minimum average |curl| value for pattern to be retained
        use_absolute: If True, use absolute curl values; else use signed values

    Returns:
        New SpiralDetectionResult with filtered patterns
    """
    filtered_patterns = []

    for pattern in detection_result.patterns:
        # Extract curl values for this pattern's voxels
        mask = np.unravel_index(pattern.voxel_indices, curl_field.shape)
        pattern_curl = curl_field[mask]

        # Compute average curl strength
        if use_absolute:
            avg_curl = np.nanmean(np.abs(pattern_curl))
        else:
            avg_curl = np.nanmean(pattern_curl)

        # Keep pattern if it exceeds threshold
        if avg_curl >= min_avg_curl:
            filtered_patterns.append(pattern)

    num_patterns = len(filtered_patterns)
    extra_statistics = {
        "original_num_patterns": detection_result.num_patterns,
        "filtered_by_curl": detection_result.num_patterns - num_patterns,
    }

    detection_params = {
        **detection_result.detection_params,
        "min_avg_curl": min_avg_curl,
        "use_absolute_curl": use_absolute,
    }

    return _create_detection_result(
        filtered_patterns,
        detection_result.input_shape,
        detection_params,
        detection_result.rotation_direction,
        detection_result.curl_sign,
        extra_statistics=extra_statistics,
    )


def get_pattern_trajectories(
    patterns: List[SpiralPattern],
    use_weighted: bool = True
) -> List[np.ndarray]:
    """
    Extract centroid trajectories for each pattern.

    Parameters:
        patterns: List of SpiralPattern objects
        use_weighted: If True, use weighted centroids; else geometric centroids

    Returns:
        List of trajectories, each shape (duration, 2) with (x, y) coordinates
    """
    trajectories = []
    for pattern in patterns:
        if use_weighted:
            trajectories.append(pattern.weighted_centroids)
        else:
            trajectories.append(pattern.centroids)
    return trajectories


def get_pattern_statistics_summary(result: SpiralDetectionResult) -> Dict:
    """
    Generate comprehensive statistics summary for detected patterns.

    Parameters:
        result: SpiralDetectionResult from detect_spirals()

    Returns:
        Dictionary with detailed statistics
    """
    if result.num_patterns == 0:
        return {
            "num_patterns": 0,
            "duration": {"mean": 0, "std": 0, "min": 0, "max": 0},
            "size": {"mean": 0, "std": 0, "min": 0, "max": 0},
            "power": {"mean": 0, "std": 0, "min": 0, "max": 0},
        }

    durations = [p.duration for p in result.patterns]
    sizes = [p.total_size for p in result.patterns]
    powers = [np.mean(p.instantaneous_powers) for p in result.patterns]

    return {
        "num_patterns": result.num_patterns,
        "duration": {
            "mean": float(np.mean(durations)),
            "std": float(np.std(durations)),
            "min": int(np.min(durations)),
            "max": int(np.max(durations)),
        },
        "size": {
            "mean": float(np.mean(sizes)),
            "std": float(np.std(sizes)),
            "min": int(np.min(sizes)),
            "max": int(np.max(sizes)),
        },
        "power": {
            "mean": float(np.mean(powers)),
            "std": float(np.std(powers)),
            "min": float(np.min(powers)),
            "max": float(np.max(powers)),
        },
    }
