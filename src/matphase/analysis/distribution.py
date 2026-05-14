"""Spiral distribution metrics and visualization helpers (Phase 5 Session 2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from matphase.analysis._utils import DatasetInput, normalize_dataset_sequence
from matphase.analysis.storage import SpiralAnalysisDataset


RotationFilter = Optional[Literal["cw", "ccw"]]


@dataclass
class MetricStats:
    """Summary statistics for a scalar metric."""

    mean: float
    std: float
    n: int

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "MetricStats":
        values = np.asarray(samples, dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            return cls(mean=float("nan"), std=float("nan"), n=0)
        return cls(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            n=int(values.size),
        )


@dataclass
class TemplateStats:
    """Spatial summary of the time-averaged spiral template."""

    mean: np.ndarray
    std: np.ndarray
    zscore: np.ndarray


@dataclass
class SpiralDistributionMetrics:
    """Aggregated metrics for a cohort/rotation subset."""

    rotation_label: str
    template: TemplateStats
    count_stats: MetricStats
    duration_stats: MetricStats
    radius_stats: MetricStats
    transverse_speed_stats: MetricStats
    samples: Dict[str, np.ndarray]


def compute_spiral_distribution_metrics(
    datasets: DatasetInput,
    *,
    rotation: Optional[str] = None,
    use_weighted_centroids: bool = True,
    show_progress: bool = False,
) -> SpiralDistributionMetrics:
    """
    Aggregate spiral distribution metrics across one or more subject bundles.

    Parameters
    ----------
    datasets:
        Either a single `SpiralAnalysisDataset` / `SpiralDetectionResult` or an
        iterable of them. Detection results are converted on the fly to storage
        datasets using an in-memory placeholder path.
    rotation:
        Optional rotation filter. Accepts "cw", "ccw", "clockwise",
        "counterclockwise", "anticlockwise", "mixed", "all", or None (default).
        When set to a specific rotation, subjects that lack matching spirals
        contribute NaNs so they do not skew averages.
    use_weighted_centroids:
        Whether to use weighted centroids when computing transverse speed.
    show_progress:
        Display a tqdm progress bar while iterating over subjects.
    """

    dataset_list = normalize_dataset_sequence(datasets)
    rotation_filter = _normalize_rotation_filter(rotation)

    grid_height = int(dataset_list[0].metadata["grid_height"])
    grid_width = int(dataset_list[0].metadata["grid_width"])

    subject_templates: List[np.ndarray] = []
    subject_count_means: List[float] = []
    duration_samples: List[np.ndarray] = []
    radius_samples: List[np.ndarray] = []
    speed_samples: List[np.ndarray] = []

    iterator: Iterable[SpiralAnalysisDataset] = dataset_list
    if show_progress and len(dataset_list) > 1:
        iterator = tqdm(dataset_list, desc="Spiral distribution metrics", unit="subject")

    for dataset in iterator:
        _validate_grid(dataset, grid_height, grid_width)
        subject_data = _compute_subject_samples(
            dataset,
            rotation_filter=rotation_filter,
            use_weighted_centroids=use_weighted_centroids,
        )
        subject_templates.append(subject_data["template"])
        subject_count_means.append(subject_data["count_mean"])
        duration_samples.append(subject_data["durations"])
        radius_samples.append(subject_data["radii"])
        speed_samples.append(subject_data["speeds"])

    template_stack = np.stack(subject_templates, axis=0).astype(np.float32, copy=False)
    template_mean = np.nanmean(template_stack, axis=0)
    template_std = np.nanstd(template_stack, axis=0)
    template_zscore = _safe_zscore(template_mean, template_std)
    template_stats = TemplateStats(mean=template_mean, std=template_std, zscore=template_zscore)

    count_stats = MetricStats.from_samples(subject_count_means)
    duration_values = _concat_samples(duration_samples)
    radius_values = _concat_samples(radius_samples)
    speed_values = _concat_samples(speed_samples)

    duration_stats = MetricStats.from_samples(duration_values)
    radius_stats = MetricStats.from_samples(radius_values)
    speed_stats = MetricStats.from_samples(speed_values)

    samples = {
        "subject_count_means": np.asarray(subject_count_means, dtype=float),
        "duration": duration_values,
        "radius": radius_values,
        "transverse_speed": speed_values,
    }

    rotation_label = "all" if rotation_filter is None else rotation_filter
    return SpiralDistributionMetrics(
        rotation_label=rotation_label,
        template=template_stats,
        count_stats=count_stats,
        duration_stats=duration_stats,
        radius_stats=radius_stats,
        transverse_speed_stats=speed_stats,
        samples=samples,
    )


def save_distribution_artifacts(
    metrics: SpiralDistributionMetrics,
    output_dir: Union[str, Path],
    *,
    prefix: str = "distribution",
    dpi: int = 150,
) -> List[Path]:
    """
    Save diagnostic heatmaps/histograms for the given metrics.

    Returns
    -------
    List[Path]
        Paths to the generated PNG files.
    """

    import matplotlib.pyplot as plt  # Local import to avoid hard dependency during pure analysis

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots: List[Path] = []

    # Template z-score heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    data = np.nan_to_num(metrics.template.zscore, nan=0.0)
    heatmap = ax.imshow(data, cmap="RdBu_r")
    ax.set_title(f"Spiral Template Z-Score ({metrics.rotation_label})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(heatmap, ax=ax, shrink=0.7, label="Z-Score")
    template_path = output_path / f"{prefix}_template_zscore.png"
    fig.savefig(template_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    plots.append(template_path)

    # Histograms for scalar metrics
    hist_fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    hist_fig.suptitle(f"Spiral Metric Distributions ({metrics.rotation_label})")
    histogram_specs = [
        ("Duration (frames)", metrics.samples.get("duration")),
        ("Radius (voxels^0.5)", metrics.samples.get("radius")),
        ("Transverse speed (px/frame)", metrics.samples.get("transverse_speed")),
    ]
    for axis, (label, values) in zip(axes, histogram_specs):
        axis.set_title(label)
        axis.set_xlabel(label)
        axis.set_ylabel("Count")
        if values is None:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        clean = np.asarray(values, dtype=float)
        clean = clean[~np.isnan(clean)]
        if clean.size == 0:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            axis.hist(clean, bins=min(20, max(5, int(np.sqrt(clean.size)))), color="#1f77b4", alpha=0.8)
    hist_path = output_path / f"{prefix}_metric_histograms.png"
    hist_fig.savefig(hist_path, dpi=dpi, bbox_inches="tight")
    plt.close(hist_fig)
    plots.append(hist_path)

    return plots


def _normalize_rotation_filter(rotation: Optional[str]) -> RotationFilter:
    if rotation is None:
        return None
    value = rotation.strip().lower()
    if value in {"cw", "clockwise"}:
        return "cw"
    if value in {"ccw", "counterclockwise", "anticlockwise"}:
        return "ccw"
    if value in {"mixed", "all", "any"}:
        return None
    raise ValueError(f"Unsupported rotation label: {rotation}")


def _infer_rotation_labels(patterns_df: pd.DataFrame) -> np.ndarray:
    if patterns_df.empty:
        return np.empty(0, dtype=object)
    labels = np.array(["unspecified"] * len(patterns_df), dtype=object)
    if "rotation_direction" in patterns_df:
        rotation_series = patterns_df["rotation_direction"].fillna("").astype(str).str.lower()
        ccw_mask = rotation_series.isin({"ccw", "counterclockwise", "anticlockwise"})
        cw_mask = rotation_series.isin({"cw", "clockwise"})
        labels[ccw_mask.to_numpy()] = "ccw"
        labels[cw_mask.to_numpy()] = "cw"
    if "curl_sign" in patterns_df:
        curl_values = patterns_df["curl_sign"].to_numpy(dtype=float)
        unspecified = labels == "unspecified"
        cw_mask = np.isfinite(curl_values) & (curl_values < 0) & unspecified
        ccw_mask = np.isfinite(curl_values) & (curl_values > 0) & unspecified
        labels[cw_mask] = "cw"
        labels[ccw_mask] = "ccw"
    return labels


def _compute_subject_samples(
    dataset: SpiralAnalysisDataset,
    *,
    rotation_filter: RotationFilter,
    use_weighted_centroids: bool,
) -> Dict[str, np.ndarray]:
    patterns_df = dataset.patterns.copy()
    patterns_df["_rotation_norm"] = _infer_rotation_labels(patterns_df)
    if rotation_filter is not None:
        patterns_df = patterns_df[patterns_df["_rotation_norm"] == rotation_filter]
    pattern_ids = patterns_df["pattern_id"].to_numpy(dtype=np.int64) if not patterns_df.empty else np.array([], dtype=np.int64)
    frame_subset = (
        dataset.frame_index[dataset.frame_index["pattern_id"].isin(pattern_ids)]
        if pattern_ids.size > 0
        else dataset.frame_index.iloc[0:0]
    )

    height = int(dataset.metadata["grid_height"])
    width = int(dataset.metadata["grid_width"])
    frame_count = int(dataset.metadata["frame_count"])

    allow_empty_zero = rotation_filter is None
    template = _compute_subject_template(
        frame_subset,
        dataset.coords,
        height,
        width,
        frame_count,
        allow_empty_zero=allow_empty_zero,
    )

    count_mean = _compute_subject_count_mean(frame_subset, frame_count, allow_empty_zero=allow_empty_zero)
    durations = patterns_df["duration"].to_numpy(dtype=float) if not patterns_df.empty else np.empty(0)
    durations = durations[~np.isnan(durations)]
    radii = _compute_radius_samples(frame_subset)
    speeds = _compute_transverse_speed_samples(frame_subset, use_weighted_centroids=use_weighted_centroids)

    return {
        "template": template,
        "count_mean": count_mean,
        "durations": durations,
        "radii": radii,
        "speeds": speeds,
    }


def _compute_subject_template(
    frame_subset: pd.DataFrame,
    coords: np.ndarray,
    height: int,
    width: int,
    frame_count: int,
    *,
    allow_empty_zero: bool,
) -> np.ndarray:
    if frame_subset.empty:
        fill = 0.0 if allow_empty_zero else np.nan
        return np.full((height, width), fill, dtype=np.float32)

    coord_ranges = []
    for start, end in zip(
        frame_subset["coord_start"].to_numpy(dtype=np.int64),
        frame_subset["coord_end"].to_numpy(dtype=np.int64),
    ):
        if end > start:
            coord_ranges.append(np.arange(start, end, dtype=np.int64))
    if not coord_ranges:
        fill = 0.0 if allow_empty_zero else np.nan
        return np.full((height, width), fill, dtype=np.float32)

    coord_indices = np.concatenate(coord_ranges)
    coord_subset = coords[coord_indices]
    template = np.zeros((height, width), dtype=np.float32)
    np.add.at(template, (coord_subset[:, 0], coord_subset[:, 1]), 1.0)

    if frame_count > 0:
        template /= float(frame_count)
    return template


def _compute_subject_count_mean(
    frame_subset: pd.DataFrame,
    frame_count: int,
    *,
    allow_empty_zero: bool,
) -> float:
    if frame_count <= 0:
        return float("nan")
    if frame_subset.empty:
        return 0.0 if allow_empty_zero else float("nan")

    counts = np.zeros(frame_count, dtype=float)
    grouped = frame_subset.groupby("abs_time", as_index=True)["pattern_id"].count()
    times = grouped.index.to_numpy(dtype=int)
    valid_mask = (times >= 0) & (times < frame_count)
    counts[times[valid_mask]] = grouped.to_numpy(dtype=float)[valid_mask]
    return float(np.nanmean(counts))


def _compute_radius_samples(frame_subset: pd.DataFrame) -> np.ndarray:
    if frame_subset.empty or "instantaneous_size" not in frame_subset:
        return np.empty(0, dtype=float)
    sizes = frame_subset["instantaneous_size"].to_numpy(dtype=float)
    sizes = sizes[~np.isnan(sizes)]
    if sizes.size == 0:
        return np.empty(0, dtype=float)
    radii = np.sqrt(np.maximum(sizes, 0.0) / np.pi)
    return radii


def _compute_transverse_speed_samples(
    frame_subset: pd.DataFrame,
    *,
    use_weighted_centroids: bool,
) -> np.ndarray:
    if frame_subset.empty:
        return np.empty(0, dtype=float)

    columns = (
        ["weighted_centroid_x", "weighted_centroid_y"]
        if use_weighted_centroids and {"weighted_centroid_x", "weighted_centroid_y"}.issubset(frame_subset.columns)
        else ["centroid_x", "centroid_y"]
    )

    coords_df = frame_subset[["pattern_id", "frame_idx", *columns]].sort_values(["pattern_id", "frame_idx"])
    coords_array = coords_df[columns].to_numpy(dtype=float)
    if coords_array.shape[0] < 2:
        return np.empty(0, dtype=float)

    pattern_ids = coords_df["pattern_id"].to_numpy(dtype=np.int64)
    diffs = np.diff(coords_array, axis=0)
    same_pattern = pattern_ids[1:] == pattern_ids[:-1]
    diffs = diffs[same_pattern]
    if diffs.size == 0:
        return np.empty(0, dtype=float)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    distances = distances[~np.isnan(distances)]
    return distances


def _concat_samples(arrays: Sequence[np.ndarray]) -> np.ndarray:
    valid_arrays = [np.asarray(arr, dtype=float) for arr in arrays if arr.size > 0]
    if not valid_arrays:
        return np.empty(0, dtype=float)
    return np.concatenate(valid_arrays)


def _safe_zscore(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = np.full_like(mean, np.nan, dtype=np.float32)
    mask = std > 0
    z[mask] = mean[mask] / std[mask]
    return z


def _validate_grid(dataset: SpiralAnalysisDataset, height: int, width: int) -> None:
    d_height = int(dataset.metadata["grid_height"])
    d_width = int(dataset.metadata["grid_width"])
    if (d_height, d_width) != (height, width):
        raise ValueError(
            f"Inconsistent grid size: expected {(height, width)}, got {(d_height, d_width)} for {dataset.metadata.get('cifti_file')}"
        )
