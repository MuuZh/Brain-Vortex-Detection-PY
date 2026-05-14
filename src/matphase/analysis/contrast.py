"""Task contrast utilities for Phase 5 Session 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from matphase.analysis.storage import SpiralAnalysisDataset


@dataclass(frozen=True)
class TaskEvent:
    """Simple representation of a task block."""

    condition: str
    onset: int
    duration: int

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("TaskEvent duration must be positive")
        if self.onset < 0:
            raise ValueError("TaskEvent onset must be >= 0")
        if not self.condition:
            raise ValueError("TaskEvent condition label is required")


@dataclass(frozen=True)
class ContrastSpec:
    """Definition for a numerator Vs denominator contrast."""

    name: str
    numerator: Sequence[str]
    denominator: Sequence[str]
    percent_change_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ContrastSpec requires a name")
        if not self.numerator or not self.denominator:
            raise ValueError("ContrastSpec requires numerator and denominator labels")


@dataclass
class ContrastResult:
    """Computed outputs for a contrast definition."""

    name: str
    numerator_labels: List[str]
    denominator_labels: List[str]
    difference_map: np.ndarray
    percent_change_map: np.ndarray
    condition_maps: Dict[str, np.ndarray]
    frame_counts: Dict[str, int]
    summary: Dict[str, float]


def build_condition_frame_masks(
    frame_count: int,
    events: Sequence[TaskEvent],
    *,
    onset_buffer: int = 0,
    offset_buffer: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Convert a set of events into boolean masks per condition.
    """

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    masks: Dict[str, np.ndarray] = {}
    for event in events:
        condition = event.condition.strip()
        mask = masks.setdefault(condition, np.zeros(frame_count, dtype=bool))
        start = max(0, event.onset - max(0, onset_buffer))
        stop = min(frame_count, event.onset + event.duration + max(0, offset_buffer))
        if stop > start:
            mask[start:stop] = True
    return masks


def compute_contrasts(
    dataset: SpiralAnalysisDataset,
    frame_masks: Mapping[str, np.ndarray],
    specs: Sequence[ContrastSpec],
    *,
    show_progress: bool = False,
) -> List[ContrastResult]:
    """
    Compute one or more contrasts for a subject-level dataset.
    """

    if not specs:
        raise ValueError("At least one ContrastSpec is required")
    if not frame_masks:
        raise ValueError("At least one condition mask is required")

    condition_maps: Dict[str, np.ndarray] = {}
    frame_counts: Dict[str, int] = {}
    for label, mask in frame_masks.items():
        template, frame_total = _condition_template(dataset, mask, label, show_progress=show_progress)
        condition_maps[label] = template
        frame_counts[label] = frame_total

    results: List[ContrastResult] = []
    for spec in specs:
        numerator_map = _mix_condition_maps(condition_maps, frame_counts, spec.numerator)
        denominator_map = _mix_condition_maps(condition_maps, frame_counts, spec.denominator)
        difference_map = numerator_map - denominator_map
        denominator_safe = np.where(
            np.abs(denominator_map) < spec.percent_change_epsilon,
            spec.percent_change_epsilon,
            np.abs(denominator_map),
        )
        percent_change_map = np.where(
            np.isfinite(denominator_safe),
            difference_map / denominator_safe,
            np.nan,
        )
        summary = {
            "mean_difference": float(np.nanmean(difference_map)),
            "mean_percent_change": float(np.nanmean(percent_change_map)),
            "numerator_frames": float(sum(frame_counts.get(lbl, 0) for lbl in spec.numerator)),
            "denominator_frames": float(sum(frame_counts.get(lbl, 0) for lbl in spec.denominator)),
        }
        relevant_labels = [*spec.numerator, *spec.denominator]
        condition_subset = {label: condition_maps[label] for label in relevant_labels if label in condition_maps}
        results.append(
            ContrastResult(
                name=spec.name,
                numerator_labels=list(spec.numerator),
                denominator_labels=list(spec.denominator),
                difference_map=difference_map,
                percent_change_map=percent_change_map,
                condition_maps=condition_subset,
                frame_counts=frame_counts.copy(),
                summary=summary,
            )
        )
    return results


def save_contrast_artifacts(
    results: Sequence[ContrastResult],
    output_dir: Union[str, Path],
    *,
    prefix: str = "contrast",
    dpi: int = 150,
) -> List[Path]:
    """
    Save PNG artifacts (difference + percent-change heatmaps) for each contrast.
    """

    import matplotlib.pyplot as plt  # Local import to avoid eager dependency

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    for index, result in enumerate(results, start=1):
        safe_name = result.name.lower().replace(" ", "_")

        fig, ax = plt.subplots(figsize=(6, 4))
        heat = ax.imshow(np.nan_to_num(result.difference_map, nan=0.0), cmap="RdBu_r")
        ax.set_title(f"{result.name}: Difference")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(heat, ax=ax, shrink=0.7, label="Δ pattern density")
        diff_path = output_path / f"{prefix}_{index}_{safe_name}_difference.png"
        fig.savefig(diff_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(diff_path)

        fig_pc, ax_pc = plt.subplots(figsize=(6, 4))
        heat_pc = ax_pc.imshow(
            np.nan_to_num(result.percent_change_map, nan=0.0),
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        ax_pc.set_title(f"{result.name}: Percent Change")
        ax_pc.set_xlabel("X")
        ax_pc.set_ylabel("Y")
        fig_pc.colorbar(heat_pc, ax=ax_pc, shrink=0.7, label="% change")
        pc_path = output_path / f"{prefix}_{index}_{safe_name}_percent.png"
        fig_pc.savefig(pc_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_pc)
        generated.append(pc_path)

    return generated


def _condition_template(
    dataset: SpiralAnalysisDataset,
    frame_mask: np.ndarray,
    label: str,
    *,
    show_progress: bool,
) -> Tuple[np.ndarray, int]:
    frame_mask = np.asarray(frame_mask, dtype=bool)
    frame_count = frame_mask.size
    if frame_count == 0:
        raise ValueError(f"Condition '{label}' mask is empty")

    metadata = dataset.metadata
    height = int(metadata.get("grid_height"))
    width = int(metadata.get("grid_width"))
    template = np.zeros((height, width), dtype=np.float32)

    selected_frames = int(frame_mask.sum())
    if selected_frames == 0:
        return np.full((height, width), np.nan, dtype=np.float32), 0

    frame_index = dataset.frame_index
    if frame_index.empty:
        return np.zeros((height, width), dtype=np.float32), selected_frames

    abs_times = frame_index["abs_time"].to_numpy(dtype=np.int64, copy=False)
    if np.any(abs_times >= frame_count) or np.any(abs_times < 0):
        raise ValueError(f"Frame indices exceed mask bounds for condition '{label}'")
    selection = frame_mask[abs_times]
    subset = frame_index.loc[selection]

    iterator: Iterable = subset.itertuples(index=False)
    if show_progress and subset.shape[0] > 200:
        iterator = tqdm(iterator, total=subset.shape[0], desc=f"Condition {label}", unit="frame")

    coords = dataset.coords
    for row in iterator:
        start = int(row.coord_start)
        end = int(row.coord_end)
        if end <= start:
            continue
        chunk = coords[start:end]
        if chunk.size == 0:
            continue
        template[chunk[:, 0], chunk[:, 1]] += 1.0

    template /= float(selected_frames)
    return template, selected_frames


def _mix_condition_maps(
    condition_maps: Mapping[str, np.ndarray],
    frame_counts: Mapping[str, int],
    labels: Sequence[str],
) -> np.ndarray:
    valid_labels = [label for label in labels if label in condition_maps]
    if not valid_labels:
        first_shape = next(iter(condition_maps.values())).shape
        return np.full(first_shape, np.nan, dtype=np.float32)

    weights = np.array([max(frame_counts.get(label, 0), 0) for label in valid_labels], dtype=float)
    total_weight = float(np.sum(weights))
    stacked = np.stack([condition_maps[label] for label in valid_labels], axis=0)
    if total_weight == 0:
        return np.nanmean(stacked, axis=0)
    weighted = (stacked * weights[:, None, None]).sum(axis=0)
    return weighted / total_weight
