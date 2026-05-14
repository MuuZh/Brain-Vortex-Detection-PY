#!/usr/bin/env python
"""
Run the MatPhase preprocessing pipeline up to analytic cube extraction.

This script saves a complex analytic cube (Hilbert transform) per subject
and hemisphere, then stops without running spiral detection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import hilbert

from matphase.analysis.storage import _sanitize_bundle_name
from matphase.config import load_config
from matphase.config.schema import MatPhaseConfig
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
        logger.info("Using sampling rate %.6f Hz from CIFTI metadata.", float(cifti_rate))
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


def _hemisphere_ranges(config: MatPhaseConfig, hemisphere: str) -> Dict[str, tuple[float, float]]:
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


def _compute_analytic_cube(data: np.ndarray, show_progress: bool) -> np.ndarray:
    """Compute complex analytic signal via Hilbert transform (y, x, t)."""
    rows, cols, frames = data.shape
    flattened = data.reshape(rows * cols, frames)
    nan_mask = np.isnan(flattened)

    filled = np.where(nan_mask, 0.0, flattened)
    if show_progress:
        valid_count = int(np.sum(~np.all(nan_mask, axis=1)))
        logger.info("Running Hilbert transform for %d voxel traces.", valid_count)

    analytic = hilbert(filled, axis=-1).astype(np.complex64)
    analytic = np.where(nan_mask, np.nan + 1j * np.nan, analytic)
    return analytic.reshape(rows, cols, frames)


def _run_for_hemisphere(
    hemisphere: str,
    config: MatPhaseConfig,
    sampling_rate_hz: float,
    show_progress: bool,
    analytic_output_path: Path,
) -> Path:
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

    tr_seconds = 1.0 / sampling_rate_hz if sampling_rate_hz > 0 else float("inf")
    logger.info(
        "Temporal sampling rate: %.6f Hz (TR=%.6f s)",
        sampling_rate_hz,
        tr_seconds,
    )

    logger.info("Running temporal bandpass filter...")
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

    if config.preprocessing.phase_extraction_method == "generalized_phase":
        logger.warning(
            "Generalized Phase is enabled for temporal extraction. "
            "Analytic cube is still computed via Hilbert transform on spatial bandpass."
        )

    logger.info("Computing analytic cube (Hilbert)...")
    analytic_cube = _compute_analytic_cube(spatial_bandpass, show_progress=show_progress)

    analytic_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(analytic_output_path, analytic_cube)
    logger.info("Saved analytic cube -> %s", analytic_output_path)
    return analytic_output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing to save analytic cube (no detection)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/defaults.yaml"),
        help="Path to base configuration YAML.",
    )
    parser.add_argument(
        "--hemisphere",
        choices=["left", "right", "both"],
        default="left",
        help="Hemisphere selection.",
    )
    parser.add_argument(
        "--cifti-file",
        type=str,
        default=None,
        help="Override paths.cifti_file from config.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Output root for analytic bundles (default: output_dir/bundles).",
    )
    parser.add_argument(
        "--flat-output-dir",
        type=Path,
        default=None,
        help="Save all analytic cubes in one folder (filenames prefixed by bundle name).",
    )
    parser.add_argument(
        "--bundle-suffix",
        type=str,
        default=None,
        help="Suffix appended to bundle directory name (default: hemisphere).",
    )
    parser.add_argument(
        "--analytic-filename",
        type=str,
        default="analytic_cube.npy",
        help="Filename for saved analytic cube.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override paths.output_dir for this run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override paths.data_dir for this run.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Override sampling rate in Hz.",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Override TR in seconds (sampling rate = 1/TR).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (default: bundle/run.log).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Enable progress bars.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)

    if args.output_dir is not None:
        config.paths.output_dir = args.output_dir
    if args.data_dir is not None:
        config.paths.data_dir = args.data_dir
    if args.cifti_file is not None:
        config.paths.cifti_file = args.cifti_file

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
    flat_output_dir = args.flat_output_dir

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

    log_file = args.log_file
    if log_file is None:
        first_dir = next(iter(bundle_dirs.values()))
        if flat_output_dir is not None:
            log_file = flat_output_dir / "log" / f"{first_dir.name}__run.log"
        else:
            log_file = first_dir / "run.log"
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level=config.output.log_level, log_file=log_path)

    cifti_ts = load_cifti(str(cifti_path))
    override_sampling = args.sampling_rate is not None or args.tr is not None
    metadata_sampling_rate = getattr(cifti_ts.metadata, "sampling_rate", None)
    if override_sampling:
        logger.info(
            "CLI sampling-rate/TR override active; ignoring CIFTI metadata value (%.6f Hz).",
            metadata_sampling_rate or float("nan"),
        )
        metadata_sampling_rate = None
    sampling_rate_hz = args.sampling_rate
    if sampling_rate_hz is None and args.tr is not None:
        sampling_rate_hz = 1.0 / args.tr
    if sampling_rate_hz is None:
        sampling_rate_hz = _resolve_sampling_rate(config, metadata_sampling_rate)

    saved_paths: List[Path] = []
    for hemi in hemispheres:
        bundle_dir = bundle_dirs[hemi]
        if flat_output_dir is not None:
            flat_output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = bundle_dir.name
            base_name = Path(args.analytic_filename).name
            output_path = flat_output_dir / f"{safe_name}__{base_name}"
        else:
            output_path = bundle_dir / args.analytic_filename
        _run_for_hemisphere(
            hemisphere=hemi,
            config=config,
            sampling_rate_hz=sampling_rate_hz,
            show_progress=args.show_progress,
            analytic_output_path=output_path,
        )
        saved_paths.append(output_path)

    logger.info("Done. Analytic cubes saved: %s", ", ".join(str(p) for p in saved_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
