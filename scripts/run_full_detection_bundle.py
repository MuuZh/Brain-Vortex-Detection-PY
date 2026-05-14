#!/usr/bin/env python
"""
Run the MatPhase preprocessing+detection pipeline and save SpiralAnalysisDataset bundles.

Example:
    conda run -n base python scripts/run_full_detection_bundle.py \
        --config configs/defaults.yaml \
        --hemisphere both \
        --curl-threshold 0.85 \
        --bundle-root output/bundles_real
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matphase.detect.spirals import SpiralPattern
import argparse
from pathlib import Path
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm

from matphase.analysis.storage import (
    save_subject_bundle_from_detection,
    _sanitize_bundle_name,
)
from matphase.config import load_config
from matphase.config.schema import MatPhaseConfig
from matphase.detect.expansion import (
    apply_expanded_masks_to_detection,
    compute_phase_alignment_mask,
    expand_spiral_patterns,
)
from matphase.detect.phase_field import compute_phase_field
from matphase.detect.spirals import (
    SpiralDetectionResult,
    detect_spirals_directional,
)
from matphase.detect.surrogates import generate_surrogate_fft
from matphase.detect.thresholds import (
    apply_combined_threshold,
    compute_expansion_field,
    compute_phase_coherence,
)
from matphase.io import load_cifti, load_surface
from matphase.io.parcellation import load_parcellation, parcellation_to_mask
from matphase.preprocess import interpolate_to_grid_batch
from matphase.preprocess.spatial import spatial_bandpass_filter
from matphase.preprocess.temporal import temporal_bandpass_filter
from matphase.utils import get_logger, setup_logging

logger = get_logger(__name__)


def _resolve_sampling_rate(config: MatPhaseConfig, cifti_rate: Optional[float]) -> float:
    """Return sampling rate (Hz) from CIFTI metadata or config fallback."""
    if cifti_rate and cifti_rate > 0:
        logger.info(
            "Using sampling rate %.6f Hz from CIFTI metadata.", float(cifti_rate))
        return float(cifti_rate)
    if config.preprocessing.temporal_sampling_rate:
        logger.info(
            "Using sampling rate %.6f Hz from config.preprocessing.temporal_sampling_rate.",
            float(config.preprocessing.temporal_sampling_rate),
        )
        return float(config.preprocessing.temporal_sampling_rate)
    raise ValueError(
        "Unable to determine sampling rate. "
        "Provide preprocessing.temporal_sampling_rate or use data with metadata."
    )


def _hilbert_phase_cube(data: np.ndarray, show_progress: bool) -> np.ndarray:
    """Compute Hilbert phase for each voxel time-series using vectorized FFT."""
    rows, cols, frames = data.shape
    flattened = data.reshape(rows * cols, frames)
    nan_mask = np.isnan(flattened)

    # Replace NaNs with zeros for Hilbert computation
    filled = np.where(nan_mask, 0.0, flattened)

    if show_progress:
        valid_count = np.sum(~np.all(nan_mask, axis=1))
        logger.info(
            "Running Hilbert transform for %d voxel traces.", int(valid_count))

    # Compute analytic signal for all traces at once
    analytic = hilbert(filled, axis=-1)
    phase = np.angle(analytic).astype(np.float32)

    # Restore NaNs where original data was NaN
    phase = np.where(nan_mask, np.nan, phase)
    return phase.reshape(rows, cols, frames)


def _generate_surrogate_stream(
    data: np.ndarray,
    *,
    n_surrogates: int,
    random_seed: Optional[int],
    phase_mode: str,
    show_progress: bool,
) -> Iterable[np.ndarray]:
    """Yield surrogate realizations without storing the entire batch in memory."""
    if n_surrogates < 1:
        raise ValueError("n_surrogates_threshold must be >= 1")

    iterator = range(n_surrogates)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc="Generating surrogates",
            unit="surrogate",
            total=n_surrogates,
        )

    for idx in iterator:
        seed = (random_seed + idx) if random_seed is not None else None
        result = generate_surrogate_fft(
            data,
            random_seed=seed,
            phase_mode=phase_mode,
            enforce_symmetry=True,
        )
        yield result.surrogate


def _estimate_thresholds_from_surrogates(
    spatial_bandpass: np.ndarray,
    config: MatPhaseConfig,
    *,
    show_progress: bool,
) -> Dict[str, float]:
    """Return thresholds computed from surrogate distributions."""
    curl_max_values: List[float] = []
    expansion_max_values: List[float] = []
    coherence_max_values: List[float] = []

    spacing = float(config.preprocessing.downsample_rate)
    percentile = config.detection.surrogate_percentile
    compute_extra = (
        config.detection.expansion_threshold is not None
        or config.detection.phase_coherence_threshold is not None
    )

    for surrogate_data in _generate_surrogate_stream(
        spatial_bandpass,
        n_surrogates=config.detection.n_surrogates_threshold,
        random_seed=config.preprocessing.surrogate_random_seed,
        phase_mode=config.preprocessing.surrogate_phase_mode,
        show_progress=show_progress,
    ):
        surrogate_phase = _hilbert_phase_cube(
            surrogate_data, show_progress=False)
        surrogate_field = compute_phase_field(
            surrogate_phase,
            spacing=spacing,
            show_progress=False,
        )

        curl = surrogate_field.curl
        finite_curl = np.isfinite(curl)
        if np.any(finite_curl):
            curl_max_values.append(float(np.nanmax(np.abs(curl[finite_curl]))))

        if compute_extra:
            grad_x = surrogate_field.gradient_x
            grad_y = surrogate_field.gradient_y

            expansion = compute_expansion_field(grad_x, grad_y)
            finite_exp = np.isfinite(expansion)
            if np.any(finite_exp):
                expansion_max_values.append(
                    float(np.nanmax(np.abs(expansion[finite_exp]))))

            coherence = compute_phase_coherence(grad_x, grad_y)
            finite_coh = np.isfinite(coherence)
            if np.any(finite_coh):
                coherence_max_values.append(
                    float(np.nanmax(coherence[finite_coh])))

    if not curl_max_values:
        raise RuntimeError(
            "Surrogate generation produced no finite curl values for thresholding")

    thresholds: Dict[str, float] = {
        "curl": float(np.percentile(curl_max_values, percentile)),
    }
    if compute_extra and expansion_max_values:
        thresholds["expansion"] = float(
            np.percentile(expansion_max_values, percentile))
    if compute_extra and coherence_max_values:
        thresholds["phase_coherence"] = float(
            np.percentile(coherence_max_values, percentile))
    return thresholds


def _build_pattern_time_value_map(
    patterns: Sequence["SpiralPattern"],
    attribute: str,
) -> Dict[int, Dict[int, float]]:
    """
    Build pattern_id -> abs_time -> value map for a SpiralPattern attribute.

    Parameters
    ----------
    patterns : Sequence[SpiralPattern]
        Pattern collection.
    attribute : {"compatibility_ratios", "expansion_radii"}
        Attribute name to pull per-frame values from.
    """
    mapping: Dict[int, Dict[int, float]] = {}
    for pattern in patterns:
        values = getattr(pattern, attribute, None)
        if values is None or len(values) != len(pattern.absolute_times):
            continue
        per_time: Dict[int, float] = {}
        for idx, abs_time in enumerate(pattern.absolute_times):
            value = float(values[idx])
            if not np.isfinite(value):
                continue
            if attribute == "expansion_radii" and value <= 0:
                continue
            per_time[int(abs_time)] = value
        if per_time:
            mapping[pattern.pattern_id] = per_time
    return mapping


def _filter_pattern_masks_by_pass_map(
    pattern_masks: Dict[int, Dict[int, np.ndarray]],
    passes_filter: Dict[int, Dict[int, bool]],
) -> tuple[Dict[int, Dict[int, np.ndarray]], int, int]:
    """
    Keep only pattern slices that pass the compatibility filter.

    Returns
    -------
    filtered_masks : Dict[int, Dict[int, np.ndarray]]
        Updated pattern masks with failed frames removed.
    kept_count : int
        Number of slices retained.
    removed_count : int
        Number of slices removed.
    """
    filtered: Dict[int, Dict[int, np.ndarray]] = {}
    kept = 0
    removed = 0
    for pattern_id, slices in pattern_masks.items():
        pass_map = passes_filter.get(pattern_id, {})
        for abs_time, mask in slices.items():
            if pass_map.get(int(abs_time), False):
                filtered.setdefault(pattern_id, {})[abs_time] = mask
                kept += 1
            else:
                removed += 1
    return filtered, kept, removed


def _apply_pass_map_to_patterns(
    patterns: Sequence["SpiralPattern"],
    passes_filter: Dict[int, Dict[int, bool]],
) -> List[SpiralPattern]:
    """
    Return new SpiralPattern list with compatibility_passes set from pass map.
    """
    updated: List[SpiralPattern] = []
    for pattern in patterns:
        pass_array = np.zeros(pattern.duration, dtype=bool)
        pattern_map = passes_filter.get(pattern.pattern_id, {})
        for idx, abs_time in enumerate(pattern.absolute_times):
            pass_array[idx] = bool(pattern_map.get(int(abs_time), False))
        try:
            updated_pattern = replace(pattern, compatibility_passes=pass_array)
        except TypeError:
            updated_pattern = type(pattern)(
                **{**pattern.__dict__, "compatibility_passes": pass_array}
            )
        updated.append(updated_pattern)
    return updated


def _compute_surrogate_compatibility_distributions(
    raw_bandpassed: np.ndarray,
    spatial_bandpass: np.ndarray,
    config: MatPhaseConfig,
    phase_field_spacing: float,
    *,
    show_progress: bool,
) -> Dict[int, float]:
    """
    Compute compatibility ratio distributions from surrogate data for filtering.

    This is a wrapper that creates a generator for surrogate spiral detection
    and passes it to the modular compatibility.compute_surrogate_compatibility_thresholds().
    """
    from matphase.detect.compatibility import SurrogateCompatibilityAccumulator

    logger.info(
        "Computing surrogate compatibility distributions for filtering...")

    # Use fixed curl threshold (matching MATLAB behavior)
    curl_threshold = config.detection.curl_threshold
    percentile = config.detection.surrogate_percentile
    # MATLAB bins compatibility ratios over radii 1..180 regardless of expansion cap
    max_radius = 180
    accumulator = SurrogateCompatibilityAccumulator(max_radius=max_radius)
    total_surrogates = config.detection.n_surrogates_threshold

    surrogate_iterator = _generate_surrogate_stream(
        raw_bandpassed,
        n_surrogates=total_surrogates,
        random_seed=config.preprocessing.surrogate_random_seed,
        phase_mode=config.preprocessing.surrogate_phase_mode,
        show_progress=show_progress and total_surrogates > 1,
    )

    use_phase_diff_mask = config.detection.phase_difference_threshold is not None

    for idx, surrogate_data in enumerate(surrogate_iterator, start=1):
        if show_progress:
            logger.info("Surrogate %d/%d: Hilbert + detection",
                        idx, total_surrogates)

        # MATLAB parity: build surrogate compatibility mask from raw vs. smooth phases
        surrogate_spatial = spatial_bandpass_filter(
            surrogate_data,
            sigma_scales=config.preprocessing.sigma_scales,
            downsample_rate=config.preprocessing.downsample_rate,
            mode=config.preprocessing.spatial_filter_mode,
            show_progress=False,
        )
        surrogate_bandpass = surrogate_spatial.bandpass[:, :, :, 0]
        surrogate_valid_mask = np.isfinite(surrogate_bandpass)

        surrogate_phase_smooth = _hilbert_phase_cube(
            surrogate_bandpass,
            show_progress=show_progress,
        )
        surrogate_phase_alignment_mask = None
        if use_phase_diff_mask:
            surrogate_phase_raw = _hilbert_phase_cube(
                surrogate_data,
                show_progress=show_progress,
            )
            surrogate_phase_alignment_mask = compute_phase_alignment_mask(
                raw_phase=surrogate_phase_raw,
                smooth_phase=surrogate_phase_smooth,
                threshold=config.detection.phase_difference_threshold,
            )

        surrogate_phase = surrogate_phase_smooth
        surrogate_field = compute_phase_field(
            surrogate_phase,
            spacing=phase_field_spacing,
            show_progress=show_progress,
        )
        surrogate_result = detect_spirals_directional(
            surrogate_field.curl,
            signal_amplitude=surrogate_field.curl,
            curl_threshold=curl_threshold,
            rotation_mode=config.detection.rotation_mode,
            min_duration=config.detection.min_pattern_duration,
            min_size=config.detection.min_pattern_size,
            connectivity=config.detection.connectivity,
            use_weighted_centroids=config.detection.use_weighted_centroids,
            show_progress=show_progress,
        )

        expansion_radii_map: Dict[int, Dict[int, float]] = {}
        if config.detection.enable_spiral_expansion and surrogate_result.num_patterns > 0:
            _, _, expansion_radii_map = expand_spiral_patterns(
                surrogate_result,
                surrogate_field.normalized_x,
                surrogate_field.normalized_y,
                valid_mask=surrogate_valid_mask,
                phase_alignment_mask=None,
                angle_center=config.detection.angle_window_center,
                angle_half_width=config.detection.angle_window_half_width,
                expansion_threshold=(
                    config.detection.expansion_threshold
                    if config.detection.expansion_threshold is not None
                    else 1.0
                ),
                radius_min=config.detection.expansion_radius_min,
                radius_max=config.detection.expansion_radius_max,
                radius_step=config.detection.expansion_radius_step,
                center_patch_radius=config.detection.center_patch_radius,
                show_progress=show_progress,
            )

        accumulator.ingest_patterns(
            patterns=surrogate_result.patterns,
            expansion_radii=expansion_radii_map,
            phase_field_vx=surrogate_field.normalized_x,
            phase_field_vy=surrogate_field.normalized_y,
            phase_alignment_mask=surrogate_phase_alignment_mask,
            show_progress=show_progress,
        )

        if show_progress:
            logger.info(
                "Surrogate %d/%d processed (%d patterns)",
                idx,
                total_surrogates,
                surrogate_result.num_patterns,
            )

    thresholds_by_radius = accumulator.finalize(percentile=percentile)

    if thresholds_by_radius:
        non_empty_radii = sum(
            1 for t in thresholds_by_radius.values() if t > 0)
        logger.info(
            "Built compatibility thresholds for %d radii (%.1f%% percentile)",
            non_empty_radii,
            percentile,
        )
    else:
        logger.warning(
            "No surrogate spirals detected; compatibility filtering will not be applied.")

    return thresholds_by_radius


def _compute_significant_metrics(
    base_result: SpiralDetectionResult,
    pattern_masks: Optional[Dict[int, Dict[int, np.ndarray]]],
    passes_filter: Optional[Dict[int, Dict[int, bool]]] = None,
) -> Dict[str, object]:
    """
    Compute MATLAB-style significant spiral metrics for metadata logging.
    """
    def _frame_is_valid(pattern_id: int, abs_time: int) -> bool:
        if not passes_filter:
            return True
        pattern_map = passes_filter.get(pattern_id)
        if pattern_map is None:
            return False
        return pattern_map.get(int(abs_time), False)

    total_frame_count = int(
        np.sum([len(pattern.absolute_times)
               for pattern in base_result.patterns], dtype=np.int64)
    )
    frame_count_kept = 0
    frame_count_extend = 0
    duration_extend: List[int] = []
    radius_values: List[float] = []

    if pattern_masks:
        for pattern in base_result.patterns:
            slices = pattern_masks.get(pattern.pattern_id)
            if not slices:
                continue
            kept_times = sorted(
                int(t)
                for t in slices.keys()
                if _frame_is_valid(pattern.pattern_id, int(t))
            )
            if not kept_times:
                continue
            frame_count_kept += len(kept_times)
            duration = int(kept_times[-1] - kept_times[0] + 1)
            duration_extend.append(duration)
            frame_count_extend += duration
            for abs_time, mask in slices.items():
                if not _frame_is_valid(pattern.pattern_id, int(abs_time)):
                    continue
                if mask is None:
                    continue
                voxels = int(np.count_nonzero(mask))
                if voxels > 0:
                    radius_values.append(float(np.sqrt(voxels / np.pi)))
    else:
        for pattern in base_result.patterns:
            kept_times = [
                int(abs_time)
                for abs_time in pattern.absolute_times
                if _frame_is_valid(pattern.pattern_id, int(abs_time))
            ]
            if not kept_times:
                continue
            frame_count_kept += len(kept_times)
            duration = int(kept_times[-1] - kept_times[0] + 1)
            duration_extend.append(duration)
            for idx, abs_time in enumerate(pattern.absolute_times):
                if not _frame_is_valid(pattern.pattern_id, int(abs_time)):
                    continue
                size = float(pattern.instantaneous_sizes[idx])
                if size > 0:
                    radius_values.append(float(np.sqrt(size / np.pi)))
        frame_count_extend = frame_count_kept

    duration_avg = float(np.mean(duration_extend)) if duration_extend else 0.0
    radius_avg = float(np.mean(radius_values)) if radius_values else 0.0
    kept_ratio = (
        float(frame_count_kept) / float(total_frame_count)
        if total_frame_count > 0
        else 0.0
    )
    kept_ratio_extend = (
        float(frame_count_extend) / float(total_frame_count)
        if total_frame_count > 0
        else 0.0
    )

    return {
        "significant_spiral_duration_extend_avg": duration_avg,
        "significant_spiral_frame_kept_ratio": kept_ratio,
        "significant_spiral_frame_kept_ratio_extend": kept_ratio_extend,
        "significant_spiral_frame_count": frame_count_kept,
        "significant_spiral_frame_count_extend": frame_count_extend,
        "total_spiral_frame_count": total_frame_count,
        "significant_spiral_radius_avg": radius_avg,
    }


def _resolve_detection_thresholds(
    spatial_bandpass: np.ndarray,
    config: MatPhaseConfig,
    *,
    show_progress: bool,
) -> Dict[str, Optional[float]]:
    """
    Resolve final thresholds for spiral detection.

    NOTE: Following MATLAB behavior, curl_threshold is ALWAYS taken from config,
    not from surrogate data. Surrogate data is used ONLY for post-detection
    compatibility ratio filtering, not for curl thresholding.
    """
    # MATLAB-compatible behavior: curl threshold is always from config (default 1.0)
    thresholds = {
        "curl": config.detection.curl_threshold,
        "expansion": config.detection.expansion_threshold,
        "phase_coherence": config.detection.phase_coherence_threshold,
    }

    if thresholds["curl"] is None:
        raise ValueError(
            "Curl threshold is undefined. Set detection.curl_threshold in config."
        )
    return thresholds


def _hemisphere_ranges(config: MatPhaseConfig, hemisphere: str) -> Dict[str, tuple[float, float]]:
    """Return grid coordinate ranges for the requested hemisphere."""
    if hemisphere == "left":
        return {
            "x_range": (
                config.preprocessing.left_x_coord_min,
                config.preprocessing.left_x_coord_max,
            ),
            "y_range": (
                config.preprocessing.left_y_coord_min,
                config.preprocessing.left_y_coord_max,
            ),
        }
    return {
        "x_range": (
            config.preprocessing.right_x_coord_min,
            config.preprocessing.right_x_coord_max,
        ),
        "y_range": (
            config.preprocessing.right_y_coord_min,
            config.preprocessing.right_y_coord_max,
        ),
    }


def run_pipeline_for_hemisphere(
    hemisphere: str,
    config: MatPhaseConfig,
    sampling_rate_hz: float,
    show_progress: bool,
    phase_field_spacing: float,
    phase_cube_output_path: Optional[Path] = None,
    faces: Optional[np.ndarray] = None,
) -> Tuple[SpiralDetectionResult, Optional[Dict[int, float]]]:
    """
    Run preprocessing + detection for one hemisphere.

    Returns:
        Tuple of (detection_result, surrogate_compatibility_thresholds).
        surrogate_compatibility_thresholds is None if use_surrogate_thresholds=False.
    """
    structure = "CORTEX_LEFT" if hemisphere == "left" else "CORTEX_RIGHT"
    data_root = Path(config.paths.data_dir)
    cifti_path = data_root / config.paths.cifti_file
    surface_path = data_root / (
        config.paths.surface_left if hemisphere == "left" else config.paths.surface_right
    )
    parcellation_path = data_root / (
        config.paths.parcellation_left if hemisphere == "left" else config.paths.parcellation_right
    )

    logger.info("Processing %s hemisphere", hemisphere.upper())
    logger.info("  CIFTI: %s", cifti_path)
    logger.info("  Surface: %s", surface_path)

    cifti_ts = load_cifti(str(cifti_path))
    surface = load_surface(str(surface_path))

    parcellation_mask = None
    if parcellation_path.exists():
        logger.info("  Parcellation: %s", parcellation_path)
        parcellation = load_parcellation(str(parcellation_path))
        parcellation_mask = parcellation_to_mask(parcellation)

    coords = surface.vertices[:, :2]
    surface_data = cifti_ts.get_full_surface_data(structure)
    hemi_ranges = _hemisphere_ranges(config, hemisphere)

    logger.info("Interpolating surface data to grid...")
    grid = interpolate_to_grid_batch(
        signal=surface_data,
        positions=coords,
        faces=surface.faces if config.preprocessing.interpolation_method == "tri_linear" else None,
        x_range=hemi_ranges["x_range"],
        y_range=hemi_ranges["y_range"],
        downsample_rate=config.preprocessing.downsample_rate,
        method=config.preprocessing.interpolation_method,
        coordinate_system=config.preprocessing.interpolation_coordinate_system,
        parcellation_mask=parcellation_mask,
        n_jobs=1,
        show_progress=show_progress,
    )
    logger.info("Grid shape: %s", grid.shape)

    tr_seconds = 1.0 / \
        sampling_rate_hz if sampling_rate_hz > 0 else float("inf")
    logger.info(
        "Temporal sampling rate: %.6f Hz (TR=%.6f s)",
        sampling_rate_hz,
        tr_seconds,
    )
    logger.info("Running temporal bandpass filter + Hilbert...")
    temporal = temporal_bandpass_filter(
        grid,
        sampling_rate=sampling_rate_hz,
        freq_low=config.preprocessing.filter_low_freq,
        freq_high=config.preprocessing.filter_high_freq,
        filter_order=config.preprocessing.filter_order,
        phase_method=config.preprocessing.phase_extraction_method,
        filter_range=config.preprocessing.gp_filter_range,
        phase_correction_threshold=config.preprocessing.gp_phase_correction_threshold,
        neg_freq_extension=config.preprocessing.gp_neg_freq_extension,
        return_inst_freq=config.preprocessing.gp_return_inst_freq,
        return_neg_freq_mask=config.preprocessing.gp_return_neg_freq_mask,
        demean=config.preprocessing.temporal_demean,
        show_progress=show_progress and config.preprocessing.show_temporal_progress,
    )

    logger.info("Running spatial DoG filter...")
    spatial = spatial_bandpass_filter(
        temporal.bandpassed,
        sigma_scales=config.preprocessing.sigma_scales,
        downsample_rate=config.preprocessing.downsample_rate,
        mode=config.preprocessing.spatial_filter_mode,
        show_progress=show_progress and config.preprocessing.show_spatial_progress,
    )
    spatial_bandpass = spatial.bandpass[:, :, :, 0]
    valid_mask = np.isfinite(spatial_bandpass)

    # For MATLAB parity, build the phase-difference mask using pre-spatial (unsmoothed)
    # temporal-bandpassed data as the "raw" phase, and post-spatial DoG phase as the "smooth".
    use_phase_alignment = (
        config.detection.phase_difference_threshold is not None
        and config.detection.use_surrogate_thresholds
    )
    raw_phase_cube = None
    if use_phase_alignment:
        logger.info(
            "Extracting raw (pre-spatial) phase cube for compatibility mask...")
        raw_phase_cube = _hilbert_phase_cube(
            temporal.bandpassed,
            show_progress=show_progress,
        )

    logger.info("Extracting phase from spatially filtered data...")
    phase_cube = _hilbert_phase_cube(
        spatial_bandpass, show_progress=show_progress)
    if config.output.save_phase_cube and phase_cube_output_path is not None:
        phase_cube_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(phase_cube_output_path, phase_cube)
        logger.info("Saved phase cube snapshot -> %s", phase_cube_output_path)
    phase_alignment_mask = None
    if use_phase_alignment and raw_phase_cube is not None:
        logger.info(
            "Computing phase compatibility mask (threshold=%.3f rad).",
            config.detection.phase_difference_threshold,
        )
        phase_alignment_mask = compute_phase_alignment_mask(
            raw_phase=raw_phase_cube,
            smooth_phase=phase_cube,
            threshold=config.detection.phase_difference_threshold,
        )
        keep_fraction = float(np.mean(phase_alignment_mask.astype(float)))
        logger.info(
            "Phase compatibility retains %.2f%% of voxels.",
            keep_fraction * 100.0,
        )

    logger.info("Computing phase field (gradients + curl)...")
    phase_field = compute_phase_field(
        phase_cube,
        spacing=phase_field_spacing,
        show_progress=show_progress,
    )
    curl_field = phase_field.curl

    finite_mask = np.isfinite(curl_field)
    if np.any(finite_mask):
        curl_vals = curl_field[finite_mask]
        abs_vals = np.abs(curl_vals)
        logger.info(
            "Curl stats: min=%+.3f max=%+.3f abs95=%.3f abs_max=%.3f",
            float(curl_vals.min()),
            float(curl_vals.max()),
            float(np.percentile(abs_vals, 95)),
            float(abs_vals.max()),
        )
    else:
        logger.warning(
            "Curl field contains no finite values; detection will yield zero patterns.")
        curl_vals = np.array([])

    thresholds = _resolve_detection_thresholds(
        spatial_bandpass,
        config,
        show_progress=show_progress,
    )
    curl_threshold = thresholds["curl"]
    expansion_threshold = thresholds.get("expansion")
    phase_coherence_threshold = thresholds.get("phase_coherence")
    logger.info(
        "Thresholds -> curl: %.3f%s%s",
        curl_threshold,
        (
            f", expansion: {expansion_threshold:.3f}"
            if expansion_threshold is not None
            else ""
        ),
        (
            f", phase_coherence: {phase_coherence_threshold:.3f}"
            if phase_coherence_threshold is not None
            else ""
        ),
    )

    # Compute surrogate-based compatibility thresholds if enabled
    surrogate_compatibility_thresholds: Optional[Dict[int, float]] = None
    if config.detection.use_surrogate_thresholds:
        logger.info("Computing surrogate-based compatibility thresholds...")
        surrogate_compatibility_thresholds = _compute_surrogate_compatibility_distributions(
            raw_bandpassed=temporal.bandpassed,
            spatial_bandpass=spatial_bandpass,
            config=config,
            phase_field_spacing=phase_field_spacing,
            show_progress=show_progress,
        )

    expansion_threshold_filter = (
        None
        if config.detection.enable_spiral_expansion
        else expansion_threshold
    )

    filtered_curl, threshold_details = apply_combined_threshold(
        curl_field,
        curl_threshold=curl_threshold,
        phase_gradient_x=phase_field.gradient_x,
        phase_gradient_y=phase_field.gradient_y,
        expansion_threshold=expansion_threshold_filter,
        phase_coherence_threshold=phase_coherence_threshold,
        fill_value=0.0,
    )

    for name, result in threshold_details.items():
        logger.info(
            "%s threshold %.3f pass_fraction=%.4f",
            name.capitalize(),
            result.threshold_value,
            result.pass_fraction,
        )
        if name == "curl" and result.pass_fraction == 0.0:
            logger.warning(
                "No voxels exceed the %s threshold %.3f (consider adjusting thresholds or using surrogates).",
                name,
                result.threshold_value,
            )

    logger.info(
        "Detecting spirals (threshold=%s, min_duration=%s, min_size=%s, rotation=%s)...",
        curl_threshold,
        config.detection.min_pattern_duration,
        config.detection.min_pattern_size,
        config.detection.rotation_mode,
    )
    detection_result = detect_spirals_directional(
        filtered_curl,
        signal_amplitude=curl_field,
        curl_threshold=curl_threshold,
        rotation_mode=config.detection.rotation_mode,
        min_duration=config.detection.min_pattern_duration,
        min_size=config.detection.min_pattern_size,
        connectivity=config.detection.connectivity,
        use_weighted_centroids=config.detection.use_weighted_centroids,
        show_progress=show_progress,
    )
    logger.info(
        "Detected %d spiral patterns in %s hemisphere",
        detection_result.num_patterns,
        hemisphere.upper(),
    )

    pattern_masks_for_metrics: Optional[Dict[int,
                                             Dict[int, np.ndarray]]] = None
    compatibility_pass_map: Optional[Dict[int, Dict[int, bool]]] = None
    final_result = detection_result
    if (
        config.detection.enable_spiral_expansion
        and detection_result.num_patterns > 0
    ):
        logger.info(
            "Expanding %d spiral seeds via MATLAB radius/angle rules.",
            detection_result.num_patterns,
        )
        expanded_masks, pattern_mask_map, expansion_radii_map = expand_spiral_patterns(
            detection_result,
            phase_field.normalized_x,
            phase_field.normalized_y,
            valid_mask=valid_mask,
            phase_alignment_mask=None,
            angle_center=config.detection.angle_window_center,
            angle_half_width=config.detection.angle_window_half_width,
            expansion_threshold=(
                config.detection.expansion_threshold
                if config.detection.expansion_threshold is not None
                else 1.0
            ),
            radius_min=config.detection.expansion_radius_min,
            radius_max=config.detection.expansion_radius_max,
            radius_step=config.detection.expansion_radius_step,
            center_patch_radius=config.detection.center_patch_radius,
            show_progress=show_progress,
        )
        for direction, mask in expanded_masks.items():
            if mask.size == 0:
                continue
            voxels = int(np.count_nonzero(mask))
            logger.info("Expanded mask %s covers %d voxels",
                        direction.upper(), voxels)
        final_result = apply_expanded_masks_to_detection(
            detection_result,
            pattern_mask_map,
            amplitude_field=curl_field,
            expansion_radii=expansion_radii_map,
        )
        if pattern_mask_map:
            pattern_masks_for_metrics = pattern_mask_map
        logger.info(
            "Expanded spiral patterns -> %d total patterns.",
            final_result.num_patterns,
        )
        if final_result.num_patterns > 0:
            sizes = [p.instantaneous_sizes.mean()
                     for p in final_result.patterns]
            logger.info(
                "Expanded patterns size stats -> mean=%.2f std=%.2f min=%.2f max=%.2f",
                float(np.mean(sizes)),
                float(np.std(sizes)),
                float(np.min(sizes)),
                float(np.max(sizes)),
            )
    elif config.detection.enable_spiral_expansion:
        logger.info("No spiral seeds detected; skipping expansion stage.")

    # Apply compatibility ratio calculation if use_surrogate_thresholds is enabled
    if config.detection.use_surrogate_thresholds and final_result.num_patterns > 0:
        from matphase.detect.compatibility import (
            apply_compatibility_ratios_to_patterns,
            filter_patterns_by_compatibility,
        )

        logger.info("Calculating compatibility ratios for real spirals...")

        # Calculate compatibility ratios for all patterns
        updated_patterns = apply_compatibility_ratios_to_patterns(
            patterns=final_result.patterns,
            phase_field_vx=phase_field.normalized_x,
            phase_field_vy=phase_field.normalized_y,
            phase_alignment_mask=phase_alignment_mask,
            show_progress=show_progress,
        )

        # Rebuild detection result with updated patterns
        labeled_volume = np.zeros(final_result.input_shape, dtype=int)
        for pattern in updated_patterns:
            mask = np.unravel_index(
                pattern.voxel_indices, final_result.input_shape)
            labeled_volume[mask] = pattern.pattern_id

        final_result = SpiralDetectionResult(
            patterns=updated_patterns,
            num_patterns=len(updated_patterns),
            labeled_volume=labeled_volume,
            input_shape=final_result.input_shape,
            detection_params=final_result.detection_params,
            statistics=final_result.statistics,
            rotation_direction=final_result.rotation_direction,
            curl_sign=final_result.curl_sign,
        )

        logger.info("Compatibility ratios calculated for %d patterns",
                    len(updated_patterns))

        # Derive compatibility + radius maps for filtering
        compatibility_ratio_map = _build_pattern_time_value_map(
            updated_patterns,
            "compatibility_ratios",
        )
        expansion_radius_map = _build_pattern_time_value_map(
            updated_patterns,
            "expansion_radii",
        )

        if (
            surrogate_compatibility_thresholds
            and compatibility_ratio_map
            and expansion_radius_map
        ):
            logger.info(
                "Applying surrogate percentile filter to real spirals...")
            compatibility_pass_map = filter_patterns_by_compatibility(
                updated_patterns,
                compatibility_ratios=compatibility_ratio_map,
                expansion_radii=expansion_radius_map,
                surrogate_thresholds=surrogate_compatibility_thresholds,
                show_progress=show_progress,
            )
            if compatibility_pass_map:
                kept_frames = sum(
                    sum(1 for keep in times.values() if keep)
                    for times in compatibility_pass_map.values()
                )
                total_frames = sum(len(times)
                                   for times in compatibility_pass_map.values())
                kept_pct = (kept_frames / total_frames *
                            100.0) if total_frames > 0 else 0.0
                logger.info(
                    "Compatibility filter kept %d/%d frames (%.2f%%).",
                    kept_frames,
                    total_frames,
                    kept_pct,
                )
                if pattern_masks_for_metrics:
                    (
                        pattern_masks_for_metrics,
                        kept_slices,
                        removed_slices,
                    ) = _filter_pattern_masks_by_pass_map(
                        pattern_masks_for_metrics,
                        compatibility_pass_map,
                    )
                    logger.info(
                        "Filtered expanded masks: kept %d slices, removed %d.",
                        kept_slices,
                        removed_slices,
                    )
                updated_patterns = _apply_pass_map_to_patterns(
                    updated_patterns,
                    compatibility_pass_map,
                )
                final_result = replace(
                    final_result,
                    patterns=updated_patterns,
                    num_patterns=len(updated_patterns),
                )

    significance_metrics = _compute_significant_metrics(
        detection_result,
        pattern_masks_for_metrics,
        compatibility_pass_map,
    )
    final_result.statistics.update(significance_metrics)

    return final_result, surrogate_compatibility_thresholds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MatPhase preprocessing+detection and save SpiralAnalysisDataset bundles.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/defaults.yaml"),
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--cifti-file",
        type=Path,
        help="Override paths.cifti_file with an explicit CIFTI path (bypasses data_dir).",
    )
    parser.add_argument(
        "--hemisphere",
        choices=["left", "right", "both"],
        default="both",
        help="Hemisphere selection for processing.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Directory for saved bundles (default: <output_dir>/bundles).",
    )
    parser.add_argument(
        "--bundle-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to bundle folder names.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override paths.data_dir from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override paths.output_dir from config.",
    )
    parser.add_argument(
        "--curl-threshold",
        type=float,
        default=None,
        help="Override detection.curl_threshold for this run.",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=None,
        help="Override detection.min_pattern_duration.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Override detection.min_pattern_size.",
    )
    parser.add_argument(
        "--rotation-mode",
        choices=["both", "ccw", "cw"],
        default=None,
        help="Override detection.rotation_mode.",
    )
    parser.add_argument(
        "--expansion-threshold",
        type=float,
        default=None,
        help="Override detection.expansion_threshold (set to 1.0 for MATLAB parity).",
    )
    parser.add_argument(
        "--phase-coherence-threshold",
        type=float,
        default=None,
        help="Override detection.phase_coherence_threshold (use pi/6 for MATLAB parity).",
    )
    parser.add_argument(
        "--phase-field-spacing",
        type=float,
        default=None,
        help="Override phase-field grid spacing (default 1.0 to mirror MATLAB step size).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Override temporal sampling rate in Hz (mutually exclusive with --tr).",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Override repetition time in seconds (mutually exclusive with --sampling-rate).",
    )
    parser.add_argument(
        "--use-surrogate-thresholds",
        action="store_true",
        help="Derive thresholds from surrogate realizations instead of fixed values.",
    )
    parser.add_argument(
        "--surrogate-percentile",
        type=float,
        default=None,
        help="Override detection.surrogate_percentile when using surrogate thresholds.",
    )
    parser.add_argument(
        "--n-surrogates-threshold",
        type=int,
        default=None,
        help="Override detection.n_surrogates_threshold for surrogate estimation.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Enable tqdm progress indicators.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path; defaults to stderr unless provided.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config, apply_env=True)

    if args.data_dir:
        config.paths.data_dir = args.data_dir
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.cifti_file:
        config.paths.cifti_file = args.cifti_file
    if args.sampling_rate is not None and args.tr is not None:
        raise ValueError("Use either --sampling-rate or --tr, not both.")
    if args.sampling_rate is not None:
        if args.sampling_rate <= 0:
            raise ValueError("--sampling-rate must be positive.")
        config.preprocessing.temporal_sampling_rate = args.sampling_rate
    if args.tr is not None:
        if args.tr <= 0:
            raise ValueError("--tr must be positive.")
        config.preprocessing.temporal_sampling_rate = 1.0 / args.tr
    if args.curl_threshold is not None:
        config.detection.curl_threshold = args.curl_threshold
    if args.min_duration is not None:
        config.detection.min_pattern_duration = args.min_duration
    if args.min_size is not None:
        config.detection.min_pattern_size = args.min_size
    if args.rotation_mode is not None:
        config.detection.rotation_mode = args.rotation_mode
    if args.expansion_threshold is not None:
        config.detection.expansion_threshold = args.expansion_threshold
    if args.phase_coherence_threshold is not None:
        config.detection.phase_coherence_threshold = args.phase_coherence_threshold
    if args.use_surrogate_thresholds:
        config.detection.use_surrogate_thresholds = True
    if args.surrogate_percentile is not None:
        config.detection.surrogate_percentile = args.surrogate_percentile
    if args.n_surrogates_threshold is not None:
        config.detection.n_surrogates_threshold = args.n_surrogates_threshold
    phase_field_spacing = args.phase_field_spacing if args.phase_field_spacing is not None else 1.0

    data_root = Path(config.paths.data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    cifti_path_candidate = Path(config.paths.cifti_file)
    cifti_path = (
        cifti_path_candidate
        if cifti_path_candidate.is_absolute()
        else data_root / cifti_path_candidate
    )
    if not cifti_path.exists():
        raise FileNotFoundError(f"CIFTI file not found: {cifti_path}")

    bundle_root = args.bundle_root or (config.paths.output_dir / "bundles")

    hemispheres: List[str]
    if args.hemisphere == "both":
        hemispheres = ["left", "right"]
    else:
        hemispheres = [args.hemisphere]

    bundle_suffixes: Dict[str, str] = {}
    bundle_dirs: Dict[str, Path] = {}
    for hemi in hemispheres:
        suffix = args.bundle_suffix or hemi
        bundle_suffixes[hemi] = suffix
        bundle_name = _sanitize_bundle_name(cifti_path, suffix=suffix)
        bundle_dirs[hemi] = bundle_root / bundle_name

    log_file = args.log_file if args.log_file is not None else config.output.log_file
    if log_file is None:
        # Default to per-bundle log file for first hemisphere
        first_dir = next(iter(bundle_dirs.values()))
        log_file = first_dir / "run.log"
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level=config.output.log_level, log_file=log_path)
    logger.info("MatPhase full pipeline starting...")
    logger.info(
        "Phase-field spacing set to %.3f (units match MATLAB index grid).",
        phase_field_spacing,
    )
    if config.detection.apply_phase_difference_mask:
        logger.warning(
            "config.detection.apply_phase_difference_mask is DEPRECATED and ignored for expansion; "
            "phase-difference masking is applied only for compatibility filtering (MATLAB parity)."
        )

    bundle_root.mkdir(parents=True, exist_ok=True)

    cifti_ts = load_cifti(str(cifti_path))
    override_sampling = args.sampling_rate is not None or args.tr is not None
    metadata_sampling_rate = getattr(cifti_ts.metadata, "sampling_rate", None)
    if override_sampling:
        logger.info("CLI sampling-rate/TR override active; ignoring CIFTI metadata value (%.6f Hz).",
                    metadata_sampling_rate or float("nan"))
        metadata_sampling_rate = None
    sampling_rate_hz = _resolve_sampling_rate(
        config,
        metadata_sampling_rate,
    )
    tr_seconds_effective = 1.0 / \
        sampling_rate_hz if sampling_rate_hz > 0 else float("inf")
    logger.info(
        "Effective temporal sampling: %.6f Hz (TR=%.6f s)",
        sampling_rate_hz,
        tr_seconds_effective,
    )

    results: Dict[str, SpiralDetectionResult] = {}
    surrogate_thresholds_by_hemi: Dict[str, Optional[Dict[int, float]]] = {}
    for hemi in hemispheres:
        phase_cube_path = (
            bundle_dirs[hemi] / config.output.phase_cube_filename
            if config.output.save_phase_cube
            else None
        )
        detection_result, surrogate_thresholds = run_pipeline_for_hemisphere(
            hemisphere=hemi,
            config=config,
            sampling_rate_hz=sampling_rate_hz,
            show_progress=args.show_progress,
            phase_field_spacing=phase_field_spacing,
            phase_cube_output_path=phase_cube_path,
            faces=None,
        )
        results[hemi] = detection_result
        surrogate_thresholds_by_hemi[hemi] = surrogate_thresholds

    logger.info("Saving SpiralAnalysisDataset bundles to %s", bundle_root)
    saved_dirs: List[Path] = []
    for hemi, detection_result in results.items():
        suffix = bundle_suffixes[hemi]

        # Prepare extra metadata including surrogate thresholds if available
        extra_metadata = {
            "hemisphere": hemi,
            "rotation_mode": config.detection.rotation_mode,
            "curl_threshold": config.detection.curl_threshold,
            "min_duration": config.detection.min_pattern_duration,
            "min_size": config.detection.min_pattern_size,
        }

        # Add surrogate compatibility thresholds to metadata if computed
        surrogate_thresholds = surrogate_thresholds_by_hemi.get(hemi)
        if surrogate_thresholds is not None and len(surrogate_thresholds) > 0:
            extra_metadata["surrogate_compatibility_thresholds"] = surrogate_thresholds
            extra_metadata["surrogate_percentile"] = config.detection.surrogate_percentile
            extra_metadata["n_surrogates_threshold"] = config.detection.n_surrogates_threshold

        bundle_dir = save_subject_bundle_from_detection(
            detection_result=detection_result,
            cifti_file=cifti_path,
            output_root=bundle_root,
            extra_metadata=extra_metadata,
            processing_config={
                "paths": config.paths.model_dump(),
                "preprocessing": config.preprocessing.model_dump(),
                "detection": config.detection.model_dump(),
            },
            suffix=suffix,
            overwrite=True,
            show_progress=args.show_progress,
        )
        logger.info(
            "Saved %s hemisphere bundle -> %s "
            "(metadata.json, patterns.parquet, frame_index.parquet, coords.feather)",
            hemi.upper(),
            bundle_dir,
        )
        if config.output.save_phase_cube:
            cube_path = bundle_dir / config.output.phase_cube_filename
            if cube_path.exists():
                logger.info("Phase cube preserved -> %s", cube_path)
            else:
                logger.warning(
                    "Phase cube expected at %s but was not found.", cube_path)
        saved_dirs.append(bundle_dir)

    logger.info("Done! Bundles created: %s", ", ".join(str(p)
                for p in saved_dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
