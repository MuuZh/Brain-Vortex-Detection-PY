"""Temporal trend analysis for spiral activity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm

from matphase.analysis._utils import DatasetInput, normalize_dataset_sequence
from matphase.analysis.storage import SpiralAnalysisDataset


RotationFilter = Optional[str]
MetricSeries = Dict[str, np.ndarray]


@dataclass
class TrendStats:
    """Linear trend statistics for a metric."""

    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float


@dataclass
class TemporalTrendResult:
    """Aggregated time-series metrics and trend stats."""

    rotation_label: str
    time_axis: np.ndarray
    mean_series: MetricSeries
    trend_stats: Dict[str, TrendStats]
    subject_series: List[MetricSeries]


def compute_temporal_trends(
    datasets: DatasetInput,
    *,
    rotation: Optional[str] = None,
    time_bin: int = 1,
    smooth_window: int = 1,
    show_progress: bool = False,
) -> TemporalTrendResult:
    """
    Aggregate time-resolved spiral metrics and estimate linear trends.

    Parameters
    ----------
    datasets:
        One or more `SpiralAnalysisDataset` or `SpiralDetectionResult` objects.
    rotation:
        Optional rotation filter. Supports "cw", "ccw", "clockwise",
        "counterclockwise", "anticlockwise", "mixed", "all", or None.
    time_bin:
        Bin width in frames for downsampling (>=1). A value of 3 computes
        metrics over non-overlapping 3-frame windows.
    smooth_window:
        Rolling mean window (in bins) applied after binning (>=1).
    show_progress:
        Display a tqdm progress bar when iterating over subjects.
    """

    dataset_list = normalize_dataset_sequence(datasets)
    rotation_filter = _normalize_rotation_filter(rotation)

    iterator: Iterable[SpiralAnalysisDataset] = dataset_list
    if show_progress and len(dataset_list) > 1:
        iterator = tqdm(dataset_list, desc="Temporal trends", unit="subject")

    subject_series: List[MetricSeries] = []
    max_bins = 0
    for dataset in iterator:
        series = _compute_subject_series(
            dataset,
            rotation_filter=rotation_filter,
            time_bin=time_bin,
            smooth_window=smooth_window,
        )
        subject_series.append(series)
        max_bins = max(max_bins, len(next(iter(series.values()))) if series else 0)

    metrics = _collect_metric_keys(subject_series)
    mean_series: MetricSeries = {}
    time_axis = np.arange(max_bins) * max(time_bin, 1)

    for metric in metrics:
        stacked = _stack_series(subject_series, metric, max_bins)
        mean_series[metric] = np.nanmean(stacked, axis=0)

    trend_stats: Dict[str, TrendStats] = {}
    for metric, values in mean_series.items():
        trend_stats[metric] = _compute_trend(time_axis, values)

    rotation_label = "all" if rotation_filter is None else rotation_filter
    return TemporalTrendResult(
        rotation_label=rotation_label,
        time_axis=time_axis,
        mean_series=mean_series,
        trend_stats=trend_stats,
        subject_series=subject_series,
    )


def save_temporal_trend_csv(result: TemporalTrendResult, output_path: Union[str, Path]) -> Path:
    """
    Save aggregated time-series metrics to CSV.
    """

    output_path = Path(output_path)
    data = {"time": result.time_axis}
    data.update(result.mean_series)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return output_path


def save_temporal_trend_plot(
    result: TemporalTrendResult,
    output_path: Union[str, Path],
    *,
    metrics: Optional[Sequence[str]] = None,
    dpi: int = 150,
) -> Path:
    """
    Save line plots of temporal metrics.
    """

    if metrics is None:
        metrics = list(result.mean_series.keys())
    metrics = [m for m in metrics if m in result.mean_series]
    if not metrics:
        raise ValueError("No metrics available to plot.")

    output_path = Path(output_path)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.5 * len(metrics)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, metric in zip(axes, metrics):
        series = result.mean_series[metric]
        ax.plot(result.time_axis, series, label=f"{metric} (mean)")
        trend = result.trend_stats.get(metric)
        if trend and np.isfinite(trend.slope):
            fitted = trend.intercept + trend.slope * result.time_axis
            ax.plot(result.time_axis, fitted, "--", label=f"trend slope={trend.slope:.3f}")
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("time (frames)")
    fig.suptitle(f"Temporal trends (rotation={result.rotation_label})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _compute_subject_series(
    dataset: SpiralAnalysisDataset,
    *,
    rotation_filter: RotationFilter,
    time_bin: int,
    smooth_window: int,
) -> MetricSeries:
    frame_count = int(dataset.metadata.get("frame_count", 0))
    if frame_count <= 0:
        raise ValueError("Dataset metadata must include a positive frame_count.")

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

    counts = np.zeros(frame_count, dtype=float)
    size = np.full(frame_count, np.nan, dtype=float)
    width = np.full(frame_count, np.nan, dtype=float)
    power = np.full(frame_count, np.nan, dtype=float)
    duration = np.full(frame_count, np.nan, dtype=float)

    if not frame_subset.empty:
        grouped_counts = frame_subset.groupby("abs_time", as_index=True)["pattern_id"].count()
        times = grouped_counts.index.to_numpy(dtype=int)
        valid = (times >= 0) & (times < frame_count)
        counts[times[valid]] = grouped_counts.to_numpy(dtype=float)[valid]

        for column, target in [
            ("instantaneous_size", size),
            ("instantaneous_width", width),
            ("instantaneous_power", power),
        ]:
            if column in frame_subset:
                grouped = frame_subset.groupby("abs_time", as_index=True)[column].mean()
                t_idx = grouped.index.to_numpy(dtype=int)
                valid_mask = (t_idx >= 0) & (t_idx < frame_count)
                target[t_idx[valid_mask]] = grouped.to_numpy(dtype=float)[valid_mask]

        if "duration" in patterns_df.columns:
            duration_map = patterns_df.set_index("pattern_id")["duration"].to_dict()
            frame_subset = frame_subset.copy()
            frame_subset["pattern_duration"] = frame_subset["pattern_id"].map(duration_map)
            grouped_duration = frame_subset.groupby("abs_time", as_index=True)["pattern_duration"].mean()
            t_idx = grouped_duration.index.to_numpy(dtype=int)
            valid_mask = (t_idx >= 0) & (t_idx < frame_count)
            duration[t_idx[valid_mask]] = grouped_duration.to_numpy(dtype=float)[valid_mask]

    series = {
        "active_count": counts,
        "instantaneous_size": size,
        "instantaneous_width": width,
        "instantaneous_power": power,
        "pattern_duration": duration,
    }

    binned = {key: _apply_time_bin(values, time_bin) for key, values in series.items()}
    smoothed = {key: _apply_smoothing(values, smooth_window) for key, values in binned.items()}
    return smoothed


def _apply_time_bin(series: np.ndarray, time_bin: int) -> np.ndarray:
    if time_bin <= 1:
        return series
    series = np.asarray(series, dtype=float)
    n = len(series)
    bins: List[np.ndarray] = []
    for start in range(0, n, time_bin):
        chunk = series[start : start + time_bin]
        bins.append(np.nanmean(chunk))
    return np.asarray(bins, dtype=float)


def _apply_smoothing(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series
    ser = pd.Series(series, dtype=float)
    return ser.rolling(window, min_periods=1).mean().to_numpy()


def _stack_series(series_list: List[MetricSeries], metric: str, max_len: int) -> np.ndarray:
    stacked: List[np.ndarray] = []
    for series in series_list:
        values = series.get(metric, np.full(max_len, np.nan))
        if len(values) < max_len:
            pad = np.full(max_len - len(values), np.nan)
            values = np.concatenate([values, pad])
        stacked.append(values)
    return np.stack(stacked, axis=0) if stacked else np.empty((0, max_len), dtype=float)


def _collect_metric_keys(series_list: List[MetricSeries]) -> List[str]:
    keys: List[str] = []
    for series in series_list:
        for key in series.keys():
            if key not in keys:
                keys.append(key)
    return keys


def _compute_trend(time_axis: np.ndarray, values: np.ndarray) -> TrendStats:
    valid = np.isfinite(time_axis) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return TrendStats(
            slope=float("nan"),
            intercept=float("nan"),
            rvalue=float("nan"),
            pvalue=float("nan"),
            stderr=float("nan"),
        )
    result = linregress(time_axis[valid], values[valid])
    return TrendStats(
        slope=float(result.slope),
        intercept=float(result.intercept),
        rvalue=float(result.rvalue),
        pvalue=float(result.pvalue),
        stderr=float(result.stderr),
    )


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
