"""
Export utilities for detection results.

Functions for exporting detection data to CSV, JSON, and batch plot generation.
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from tqdm import tqdm

from matphase.detect.spirals import SpiralDetectionResult, SpiralPattern


def export_detection_csv(
    result: SpiralDetectionResult,
    output_path: Union[str, Path],
    include_statistics: bool = True,
) -> None:
    """
    Export detection results to CSV format.

    Creates two CSV files:
    1. {output_path}_patterns.csv: Per-pattern summary
    2. {output_path}_timepoints.csv: Per-timepoint details (optional)

    Parameters:
        result: SpiralDetectionResult to export
        output_path: Output file path (without extension)
        include_statistics: Whether to export per-timepoint statistics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export pattern-level summary
    patterns_file = output_path.parent / f"{output_path.stem}_patterns.csv"

    with open(patterns_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'pattern_id', 'rotation_direction', 'curl_sign',
            'duration', 'start_time',
            'mean_centroid_x', 'mean_centroid_y',
            'weighted_centroid_x', 'weighted_centroid_y',
            'mean_size', 'std_size', 'max_size',
            'mean_power', 'std_power', 'max_power',
            'mean_width', 'std_width',
            'bbox_y_min', 'bbox_y_max', 'bbox_x_min', 'bbox_x_max'
        ])

        # Data rows
        for pattern in result.patterns:
            # Compute statistics
            centroids_arr = np.array(pattern.centroids)
            valid_centroids = centroids_arr[~np.isnan(centroids_arr).any(axis=1)]

            mean_cx = np.nanmean(valid_centroids[:, 1]) if len(valid_centroids) > 0 else np.nan
            mean_cy = np.nanmean(valid_centroids[:, 0]) if len(valid_centroids) > 0 else np.nan

            # Weighted centroids are arrays, compute mean
            weighted_cents = pattern.weighted_centroids
            if weighted_cents is not None and len(weighted_cents) > 0:
                weighted_cx = np.nanmean(weighted_cents[:, 1])
                weighted_cy = np.nanmean(weighted_cents[:, 0])
            else:
                weighted_cx, weighted_cy = np.nan, np.nan

            mean_size = np.nanmean(pattern.instantaneous_sizes)
            std_size = np.nanstd(pattern.instantaneous_sizes)
            max_size = np.nanmax(pattern.instantaneous_sizes)

            mean_power = np.nanmean(pattern.instantaneous_powers)
            std_power = np.nanstd(pattern.instantaneous_powers)
            max_power = np.nanmax(pattern.instantaneous_powers)

            mean_width = np.nanmean(pattern.instantaneous_widths)
            std_width = np.nanstd(pattern.instantaneous_widths)

            bbox = pattern.bounding_box if pattern.bounding_box else (np.nan, np.nan, np.nan, np.nan)

            writer.writerow([
                pattern.pattern_id, pattern.rotation_direction, pattern.curl_sign,
                pattern.duration, pattern.start_time,
                mean_cx, mean_cy,
                weighted_cx, weighted_cy,
                mean_size, std_size, max_size,
                mean_power, std_power, max_power,
                mean_width, std_width,
                bbox[0], bbox[1], bbox[2], bbox[3]
            ])

    # Export per-timepoint statistics if requested
    if include_statistics:
        timepoints_file = output_path.parent / f"{output_path.stem}_timepoints.csv"

        with open(timepoints_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'pattern_id', 'timepoint', 'frame_index',
                'centroid_x', 'centroid_y',
                'size', 'power', 'peak_amplitude', 'width'
            ])

            # Data rows
            for pattern in result.patterns:
                for frame_idx in range(pattern.duration):
                    timepoint = pattern.start_time + frame_idx

                    cy, cx = pattern.centroids[frame_idx] if frame_idx < len(pattern.centroids) else (np.nan, np.nan)
                    size = pattern.instantaneous_sizes[frame_idx] if frame_idx < len(pattern.instantaneous_sizes) else np.nan
                    power = pattern.instantaneous_powers[frame_idx] if frame_idx < len(pattern.instantaneous_powers) else np.nan
                    peak_amp = pattern.instantaneous_peak_amps[frame_idx] if frame_idx < len(pattern.instantaneous_peak_amps) else np.nan
                    width = pattern.instantaneous_widths[frame_idx] if frame_idx < len(pattern.instantaneous_widths) else np.nan

                    writer.writerow([
                        pattern.pattern_id, timepoint, frame_idx,
                        cx, cy,
                        size, power, peak_amp, width
                    ])


def export_detection_json(
    result: SpiralDetectionResult,
    output_path: Union[str, Path],
    include_voxel_indices: bool = False,
) -> None:
    """
    Export detection results to JSON format.

    Parameters:
        result: SpiralDetectionResult to export
        output_path: Output file path
        include_voxel_indices: Whether to include voxel indices (can be large)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build exportable dictionary
    export_data = {
        'summary': result.statistics,
        'patterns': []
    }

    for pattern in result.patterns:
        pattern_data = {
            'pattern_id': int(pattern.pattern_id),
            'rotation_direction': pattern.rotation_direction,
            'curl_sign': pattern.curl_sign,
            'duration': int(pattern.duration),
            'start_time': int(pattern.start_time),
            'end_time': int(pattern.end_time),
            'centroids': _convert_to_list(pattern.centroids),
            'weighted_centroids': _convert_to_list(pattern.weighted_centroids),
            'instantaneous_sizes': _convert_to_list(pattern.instantaneous_sizes),
            'instantaneous_powers': _convert_to_list(pattern.instantaneous_powers),
            'instantaneous_peak_amplitudes': _convert_to_list(pattern.instantaneous_peak_amps),
            'instantaneous_widths': _convert_to_list(pattern.instantaneous_widths),
            'bounding_box': _convert_to_list(pattern.bounding_box) if pattern.bounding_box else None,
            'total_size': int(pattern.total_size),
        }

        if include_voxel_indices and pattern.voxel_indices is not None:
            # Convert voxel indices to list (warning: can be very large)
            pattern_data['voxel_indices'] = [
                _convert_to_list(indices) for indices in pattern.voxel_indices
            ]

        export_data['patterns'].append(pattern_data)

    # Write JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, cls=_NumpyEncoder)


def export_batch_plots(
    plot_function,
    output_dir: Union[str, Path],
    prefix: str,
    n_plots: int,
    show_progress: bool = True,
    **plot_kwargs,
) -> List[Path]:
    """
    Generic batch plot exporter with progress bar.

    Parameters:
        plot_function: Function that takes timepoint index and returns matplotlib figure
        output_dir: Output directory
        prefix: Filename prefix
        n_plots: Number of plots to generate
        show_progress: Show tqdm progress bar
        **plot_kwargs: Additional arguments passed to plot_function

    Returns:
        List of saved file paths
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    # Use tqdm for progress tracking (only show if >10 plots)
    iterator = range(n_plots)
    if show_progress and n_plots > 10:
        iterator = tqdm(iterator, desc=f"Generating {prefix} plots")

    for i in iterator:
        output_path = output_dir / f"{prefix}_{i:04d}.png"

        try:
            fig = plot_function(timepoint=i, **plot_kwargs)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_paths.append(output_path)
        except Exception as e:
            print(f"Warning: Failed to generate plot {i}: {e}")
            continue

    return saved_paths


def export_summary_report(
    result: SpiralDetectionResult,
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export human-readable summary report.

    Parameters:
        result: SpiralDetectionResult to summarize
        output_path: Output text file path
        metadata: Optional metadata to include in report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SPIRAL DETECTION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Metadata
        if metadata:
            f.write("Metadata:\n")
            f.write("-" * 40 + "\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write("Detection Context:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Rotation mode: {result.rotation_direction or 'unspecified'}\n")
        if result.curl_sign is not None:
            f.write(f"  Curl sign: {result.curl_sign}\n")
        f.write("\n")

        # Summary statistics
        f.write("Summary Statistics:\n")
        f.write("-" * 40 + "\n")
        for key, value in result.statistics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")

        # Pattern details
        f.write(f"Detected Patterns: {len(result.patterns)}\n")
        f.write("=" * 80 + "\n\n")

        for pattern in result.patterns:
            f.write(f"Pattern {pattern.pattern_id}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Duration: {pattern.duration} frames\n")
            f.write(f"  Start time: {pattern.start_time}\n")
            f.write(
                f"  Rotation: {pattern.rotation_direction} "
                f"(curl={pattern.curl_sign if pattern.curl_sign is not None else 'n/a'})\n"
            )

            # Statistics
            mean_size = np.nanmean(pattern.instantaneous_sizes)
            mean_power = np.nanmean(pattern.instantaneous_powers)
            mean_width = np.nanmean(pattern.instantaneous_widths)

            f.write(f"  Mean size: {mean_size:.2f} voxels\n")
            f.write(f"  Mean power: {mean_power:.4f}\n")
            f.write(f"  Mean width: {mean_width:.2f} pixels\n")

            # Centroid trajectory
            centroids = np.array(pattern.centroids)
            valid_centroids = centroids[~np.isnan(centroids).any(axis=1)]
            if len(valid_centroids) > 0:
                mean_cx = np.nanmean(valid_centroids[:, 1])
                mean_cy = np.nanmean(valid_centroids[:, 0])
                f.write(f"  Mean centroid: ({mean_cx:.2f}, {mean_cy:.2f})\n")

            # Bounding box
            if pattern.bounding_box:
                bb = pattern.bounding_box
                f.write(f"  Bounding box: Y[{bb[0]}:{bb[1]}], X[{bb[2]}:{bb[3]}]\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


# Helper functions

def _convert_to_list(obj):
    """Convert numpy arrays and tuples to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        return [_convert_to_list(item) if isinstance(item, (np.ndarray, tuple, list)) else item
                for item in obj]
    else:
        return obj


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
