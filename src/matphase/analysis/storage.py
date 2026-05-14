"""Storage helpers for Phase 5 analysis bundles.

This module serializes spiral detection outputs into per-subject bundles
consisting of three core artifacts:

- patterns.parquet: Pattern-level summary table
- frame_index.parquet: Per-pattern/per-frame indexing with coordinate offsets
- coords.feather: Sparse coordinate pool (y, x) referenced by frame_index rows

Each bundle also includes a metadata.json that captures the originating CIFTI
file path plus processing parameters. This design keeps per-subject detection
outputs independent of study/group labeling; higher-level merge utilities can
inspect metadata.json later to inject cohort information.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from matphase.detect.spirals import SpiralDetectionResult, SpiralPattern


DEFAULT_METADATA_FILE = "metadata.json"
DEFAULT_PATTERNS_FILE = "patterns.parquet"
DEFAULT_FRAME_INDEX_FILE = "frame_index.parquet"
DEFAULT_COORDS_FILE = "coords.feather"
DEFAULT_SCHEMA_VERSION = "1.0"


def _sanitize_bundle_name(cifti_file: Union[str, Path], suffix: Optional[str] = None) -> str:
    """Derive a filesystem-safe folder name from the CIFTI file."""
    base = Path(cifti_file).name
    if not base:
        base = "cifti"
    # Replace multi-extension dots with underscores so run.dtseries.nii -> run_dtseries_nii
    base = base.replace(".", "_")
    # Keep alphanumeric, dash, underscore. Collapse other chars to dash.
    base = re.sub(r"[^A-Za-z0-9_\-]+", "-", base)
    base = re.sub(r"-{2,}", "-", base).strip("-_")
    if not base:
        base = "cifti"
    if suffix:
        safe_suffix = re.sub(r"[^A-Za-z0-9_\-]+", "-", suffix).strip("-_")
        if safe_suffix:
            base = f"{base}_{safe_suffix}"
    return base


def _json_ready(value: Any) -> Any:
    """Convert numpy/scalar objects into JSON serializable values."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _ensure_dir(path: Path, *, overwrite: bool = False) -> None:
    if path.exists():
        if not overwrite and any(path.iterdir()):
            raise FileExistsError(f"Destination {path} already exists and is not empty. Set overwrite=True to replace.")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _flatten_bounding_box(bbox: Optional[Tuple[int, int, int, int, int, int]]) -> Dict[str, Optional[int]]:
    if not bbox:
        return {k: None for k in ("bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1", "bbox_t0", "bbox_t1")}
    return {
        "bbox_x0": bbox[0],
        "bbox_x1": bbox[1],
        "bbox_y0": bbox[2],
        "bbox_y1": bbox[3],
        "bbox_t0": bbox[4],
        "bbox_t1": bbox[5],
    }


@dataclass
class SpiralAnalysisDataset:
    """In-memory representation of a subject-level storage bundle."""

    metadata: Dict[str, Any]
    patterns: pd.DataFrame
    frame_index: pd.DataFrame
    coords: np.ndarray

    @classmethod
    def from_detection_result(
        cls,
        detection_result: SpiralDetectionResult,
        cifti_file: Union[str, Path],
        *,
        extra_metadata: Optional[Dict[str, Any]] = None,
        processing_config: Optional[Dict[str, Any]] = None,
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        coords_dtype: np.dtype = np.int16,
        show_progress: bool = False,
    ) -> "SpiralAnalysisDataset":
        """Create dataset from a SpiralDetectionResult."""
        input_shape = detection_result.input_shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {input_shape}")
        height, width, _ = input_shape
        max_dim = max(height, width)
        if np.iinfo(coords_dtype).max < max_dim:
            raise ValueError(
                f"coords_dtype {coords_dtype} cannot hold coordinate {max_dim}. "
                "Use a larger dtype (e.g., np.int32)."
            )

        patterns_records: List[Dict[str, Any]] = []
        frame_rows: List[Dict[str, Any]] = []
        coords_chunks: List[np.ndarray] = []
        coord_offset = 0

        iterator: Iterable[SpiralPattern] = detection_result.patterns
        if show_progress and len(detection_result.patterns) > 10:
            iterator = tqdm(iterator, desc="Serializing patterns", unit="pattern")

        for pattern in iterator:
            bbox = _flatten_bounding_box(pattern.bounding_box)

            # Calculate pattern-level aggregates for compatibility filtering
            all_frames_filtered = False
            fraction_frames_passing = None
            mean_compatibility_ratio = None
            compat_passes = getattr(pattern, "compatibility_passes", None)

            if compat_passes is not None:
                passing_count = int(np.sum(compat_passes))
                fraction_frames_passing = float(passing_count) / float(pattern.duration)
                all_frames_filtered = passing_count == 0
                if pattern.compatibility_ratios is not None:
                    valid_mask = np.isfinite(pattern.compatibility_ratios) & compat_passes
                    valid_ratios = pattern.compatibility_ratios[valid_mask]
                    if len(valid_ratios) > 0:
                        mean_compatibility_ratio = float(np.mean(valid_ratios))
            elif pattern.compatibility_ratios is not None:
                valid_ratios = pattern.compatibility_ratios[np.isfinite(pattern.compatibility_ratios)]
                if len(valid_ratios) > 0:
                    mean_compatibility_ratio = float(np.mean(valid_ratios))
                    passing_count = np.sum(valid_ratios > 0)
                    fraction_frames_passing = float(passing_count) / float(pattern.duration)
                    all_frames_filtered = (passing_count == 0)
                else:
                    all_frames_filtered = True
                    fraction_frames_passing = 0.0

            patterns_records.append(
                {
                    "pattern_id": pattern.pattern_id,
                    "rotation_direction": pattern.rotation_direction,
                    "curl_sign": pattern.curl_sign,
                    "start_frame": pattern.start_time,
                    "end_frame": pattern.end_time,
                    "duration": pattern.duration,
                    "total_size": pattern.total_size,
                    "mean_size": float(np.mean(pattern.instantaneous_sizes)),
                    "mean_power": float(np.mean(pattern.instantaneous_powers)),
                    "mean_peak_amp": float(np.mean(pattern.instantaneous_peak_amps)),
                    "all_frames_filtered": all_frames_filtered,
                    "fraction_frames_passing": fraction_frames_passing,
                    "mean_compatibility_ratio": mean_compatibility_ratio,
                    **bbox,
                }
            )

            # Map voxel indices back to (y, x, t)
            y_coords, x_coords, t_coords = np.unravel_index(pattern.voxel_indices, input_shape)
            sort_idx = np.argsort(t_coords, kind="mergesort")
            t_sorted = t_coords[sort_idx]
            y_sorted = y_coords[sort_idx].astype(coords_dtype, copy=False)
            x_sorted = x_coords[sort_idx].astype(coords_dtype, copy=False)

            for frame_idx, abs_time in enumerate(pattern.absolute_times):
                start = np.searchsorted(t_sorted, abs_time, side="left")
                end = np.searchsorted(t_sorted, abs_time, side="right")
                if start == end:
                    # Pattern metadata guarantees voxels, but skip if none for robustness
                    continue

                frame_chunk = np.column_stack((y_sorted[start:end], x_sorted[start:end]))
                coords_chunks.append(frame_chunk)
                chunk_len = frame_chunk.shape[0]
                coord_start = coord_offset
                coord_end = coord_offset + chunk_len
                coord_offset = coord_end

                # Get expansion_radius if available
                expansion_radius = None
                if pattern.expansion_radii is not None and frame_idx < len(pattern.expansion_radii):
                    expansion_radius = float(pattern.expansion_radii[frame_idx])

                # Get compatibility_ratio if available
                compatibility_ratio = None
                if pattern.compatibility_ratios is not None and frame_idx < len(pattern.compatibility_ratios):
                    compatibility_ratio = float(pattern.compatibility_ratios[frame_idx])

                compatibility_pass = None
                if compat_passes is not None and frame_idx < len(compat_passes):
                    compatibility_pass = bool(compat_passes[frame_idx])

                frame_rows.append(
                    {
                        "pattern_id": pattern.pattern_id,
                        "frame_idx": int(frame_idx),
                        "abs_time": int(abs_time),
                        "coord_start": int(coord_start),
                        "coord_end": int(coord_end),
                        "voxel_count": int(chunk_len),
                        "centroid_x": float(pattern.centroids[frame_idx, 0]),
                        "centroid_y": float(pattern.centroids[frame_idx, 1]),
                        "weighted_centroid_x": float(pattern.weighted_centroids[frame_idx, 0]),
                        "weighted_centroid_y": float(pattern.weighted_centroids[frame_idx, 1]),
                        "instantaneous_size": float(pattern.instantaneous_sizes[frame_idx]),
                        "instantaneous_power": float(pattern.instantaneous_powers[frame_idx]),
                        "instantaneous_peak_amp": float(pattern.instantaneous_peak_amps[frame_idx]),
                        "instantaneous_width": float(pattern.instantaneous_widths[frame_idx]),
                        "expansion_radius": expansion_radius,
                        "compatibility_ratio": compatibility_ratio,
                        "compatibility_pass": compatibility_pass,
                    }
                )

        coords_array = (
            np.concatenate(coords_chunks, axis=0).astype(coords_dtype, copy=False)
            if coords_chunks
            else np.empty((0, 2), dtype=coords_dtype)
        )

        metadata: Dict[str, Any] = {
            "schema_version": schema_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cifti_file": str(cifti_file),
            "grid_height": height,
            "grid_width": width,
            "frame_count": input_shape[2],
            "detection_params": _json_ready(detection_result.detection_params),
            "detection_statistics": _json_ready(detection_result.statistics),
        }
        if extra_metadata:
            metadata["extra_metadata"] = _json_ready(extra_metadata)
        if processing_config:
            metadata["processing_config"] = _json_ready(processing_config)

        patterns_df = pd.DataFrame(patterns_records)
        frame_index_df = pd.DataFrame(frame_rows)
        int_columns = [
            "pattern_id",
            "frame_idx",
            "abs_time",
            "coord_start",
            "coord_end",
            "voxel_count",
        ]
        for column in int_columns:
            if column in frame_index_df.columns and not frame_index_df.empty:
                frame_index_df[column] = frame_index_df[column].astype(np.int64)
        return cls(metadata=metadata, patterns=patterns_df, frame_index=frame_index_df, coords=coords_array)

    def save(
        self,
        output_dir: Union[str, Path],
        *,
        overwrite: bool = False,
        metadata_filename: str = DEFAULT_METADATA_FILE,
        patterns_filename: str = DEFAULT_PATTERNS_FILE,
        frame_index_filename: str = DEFAULT_FRAME_INDEX_FILE,
        coords_filename: str = DEFAULT_COORDS_FILE,
    ) -> Path:
        """Persist dataset to disk."""
        output_path = Path(output_dir)
        _ensure_dir(output_path, overwrite=overwrite)

        metadata_path = output_path / metadata_filename
        patterns_path = output_path / patterns_filename
        frame_index_path = output_path / frame_index_filename
        coords_path = output_path / coords_filename

        metadata_path.write_text(json.dumps(_json_ready(self.metadata), indent=2))
        self.patterns.to_parquet(patterns_path, index=False)
        self.frame_index.to_parquet(frame_index_path, index=False)

        coords_df = pd.DataFrame(self.coords, columns=["y", "x"])
        coords_df.to_feather(coords_path)
        return output_path


def load_subject_bundle(
    bundle_dir: Union[str, Path],
    *,
    metadata_filename: str = DEFAULT_METADATA_FILE,
    patterns_filename: str = DEFAULT_PATTERNS_FILE,
    frame_index_filename: str = DEFAULT_FRAME_INDEX_FILE,
    coords_filename: str = DEFAULT_COORDS_FILE,
) -> SpiralAnalysisDataset:
    """Load a previously saved bundle into memory."""
    bundle_path = Path(bundle_dir)
    metadata_path = bundle_path / metadata_filename
    patterns_path = bundle_path / patterns_filename
    frame_index_path = bundle_path / frame_index_filename
    coords_path = bundle_path / coords_filename

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file at {metadata_path}")
    metadata = json.loads(metadata_path.read_text())

    patterns_df = pd.read_parquet(patterns_path)
    frame_index_df = pd.read_parquet(frame_index_path)
    coords_df = pd.read_feather(coords_path)
    coords_array = coords_df[["y", "x"]].to_numpy()

    return SpiralAnalysisDataset(
        metadata=metadata,
        patterns=patterns_df,
        frame_index=frame_index_df,
        coords=coords_array,
    )


def save_subject_bundle_from_detection(
    detection_result: SpiralDetectionResult,
    cifti_file: Union[str, Path],
    output_root: Union[str, Path],
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
    processing_config: Optional[Dict[str, Any]] = None,
    suffix: Optional[str] = None,
    overwrite: bool = False,
    show_progress: bool = False,
) -> Path:
    """Convenience helper that builds + saves a bundle derived from a detection result."""
    dataset = SpiralAnalysisDataset.from_detection_result(
        detection_result,
        cifti_file,
        extra_metadata=extra_metadata,
        processing_config=processing_config,
        show_progress=show_progress,
    )
    bundle_name = _sanitize_bundle_name(cifti_file, suffix=suffix)
    output_dir = Path(output_root) / bundle_name
    dataset.save(output_dir, overwrite=overwrite)
    return output_dir


def write_storage_preview(dataset: SpiralAnalysisDataset, output_dir: Union[str, Path]) -> List[Path]:
    """Save small text previews of the dataset schema for documentation/validation."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern_preview = out_dir / "patterns_head.txt"
    frame_preview = out_dir / "frame_index_head.txt"

    pattern_preview.write_text(dataset.patterns.head().to_markdown(index=False))
    frame_preview.write_text(dataset.frame_index.head().to_markdown(index=False))
    return [pattern_preview, frame_preview]
