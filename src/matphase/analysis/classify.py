"""Pattern classification helpers for Phase 5 Session 3."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from matphase.analysis._utils import DatasetInput, normalize_dataset_sequence
from matphase.analysis.storage import SpiralAnalysisDataset


ClassifierType = Literal["nearest_centroid", "svm", "logistic", "random_forest"]
CVStrategy = Literal["kfold", "stratified_kfold", "leave_one_out"]


@dataclass
class ClassificationResult:
    """Summary of a cross-validated classification run."""

    classifier_type: str
    cv_strategy: str
    accuracy: float
    fold_scores: List[float]
    confusion_matrix: np.ndarray
    labels: List[str]
    feature_columns: List[str]
    predictions: List[str]
    truths: List[str]


def build_pattern_feature_table(
    datasets: DatasetInput,
    *,
    wrap_angles: bool = True,
) -> pd.DataFrame:
    """
    Derive feature rows (one per pattern) from one or more subject bundles.
    """

    normalized = normalize_dataset_sequence(datasets)
    rows: List[Dict[str, Union[str, float, int]]] = []

    for dataset in normalized:
        subject_id = Path(dataset.metadata.get("cifti_file", "subject")).stem
        frame_groups = (
            {pid: frame.copy() for pid, frame in dataset.frame_index.groupby("pattern_id")}
            if not dataset.frame_index.empty
            else {}
        )

        for pattern in dataset.patterns.itertuples(index=False):
            pattern_id = int(pattern.pattern_id)
            duration = float(pattern.duration)
            mean_size = float(pattern.mean_size)
            mean_power = float(pattern.mean_power)
            speed = _mean_transverse_speed(frame_groups.get(pattern_id))
            angle = _trajectory_angle(frame_groups.get(pattern_id))
            rows.append(
                {
                    "subject_id": subject_id,
                    "pattern_id": pattern_id,
                    "rotation_label": _normalize_rotation_label(getattr(pattern, "rotation_direction", "")),
                    "duration": duration,
                    "mean_size": mean_size,
                    "mean_power": mean_power,
                    "mean_peak_amp": float(pattern.mean_peak_amp),
                    "total_size": float(pattern.total_size),
                    "radius_estimate": _radius_from_area(mean_size),
                    "mean_speed": speed,
                    "trajectory_angle": angle,
                }
            )

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "pattern_id",
                "rotation_label",
                "duration",
                "mean_size",
                "mean_power",
                "mean_peak_amp",
                "total_size",
                "radius_estimate",
                "mean_speed",
                "trajectory_angle",
            ]
        )

    if wrap_angles and "trajectory_angle" in feature_df.columns:
        feature_df["trajectory_angle"] = _wrap_angle_array(feature_df["trajectory_angle"].to_numpy(dtype=float))
    return feature_df


def classify_patterns(
    feature_table: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: Optional[Sequence[str]] = None,
    classifier_type: ClassifierType = "nearest_centroid",
    cv_strategy: CVStrategy = "stratified_kfold",
    folds: int = 5,
    random_state: Optional[int] = None,
    min_class_samples: int = 2,
    show_progress: bool = False,
) -> ClassificationResult:
    """
    Run cross-validated classification on the provided feature table.
    """

    if label_column not in feature_table.columns:
        raise ValueError(f"Label column '{label_column}' not found in feature table")

    clean_df = feature_table.dropna(subset=[label_column]).copy()
    if clean_df.empty:
        raise ValueError("No labeled samples available for classification")

    numeric_cols = [col for col in clean_df.columns if col != label_column and clean_df[col].dtype.kind in {"i", "u", "f"}]
    if feature_columns is None:
        feature_columns = numeric_cols
    else:
        missing = [col for col in feature_columns if col not in clean_df.columns]
        if missing:
            raise ValueError(f"Requested feature columns not found: {missing}")

    if not feature_columns:
        raise ValueError("At least one numeric feature column is required")

    labels = clean_df[label_column].astype(str).to_numpy()
    features = clean_df[feature_columns].to_numpy(dtype=float)

    _validate_class_balance(labels, min_class_samples)

    if classifier_type != "nearest_centroid":
        warnings.warn(
            f"Classifier type '{classifier_type}' is not implemented; falling back to 'nearest_centroid'.",
            RuntimeWarning,
            stacklevel=2,
        )

    splits = list(
        _iter_cv_splits(
            labels,
            strategy=cv_strategy,
            folds=folds,
            random_state=random_state,
        )
    )
    if not splits:
        raise ValueError("Cross-validation produced no splits (check sample count vs fold count)")

    classifier = _NearestCentroidClassifier()
    all_predictions: List[str] = []
    all_truth: List[str] = []
    fold_scores: List[float] = []

    iterator = splits
    if show_progress:
        iterator = tqdm(iterator, total=len(splits), desc="Pattern classification", unit="fold")

    for train_idx, test_idx in iterator:
        train_labels = labels[train_idx]
        if np.unique(train_labels).size < 2:
            raise ValueError("Each training fold must contain at least two classes")

        classifier.fit(features[train_idx], train_labels)
        preds = classifier.predict(features[test_idx])
        truths = labels[test_idx]
        all_predictions.extend(preds)
        all_truth.extend(truths)
        acc = float(np.mean(preds == truths)) if truths.size > 0 else float("nan")
        fold_scores.append(acc)

    confusion, label_order = _confusion_matrix(all_truth, all_predictions)
    accuracy = float(np.mean(fold_scores)) if fold_scores else float("nan")
    return ClassificationResult(
        classifier_type=classifier_type,
        cv_strategy=cv_strategy,
        accuracy=accuracy,
        fold_scores=fold_scores,
        confusion_matrix=confusion,
        labels=label_order,
        feature_columns=list(feature_columns),
        predictions=all_predictions,
        truths=all_truth,
    )


def save_classification_artifacts(
    result: ClassificationResult,
    output_dir: Union[str, Path],
    *,
    prefix: str = "classification",
    dpi: int = 150,
) -> List[Path]:
    """
    Persist confusion matrix and per-fold accuracy plots.
    """

    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts: List[Path] = []

    fig, ax = plt.subplots(figsize=(4, 4))
    heat = ax.imshow(result.confusion_matrix, cmap="Blues")
    ax.set_title("Classification Confusion Matrix")
    ax.set_xticks(range(len(result.labels)))
    ax.set_xticklabels(result.labels, rotation=45, ha="right")
    ax.set_yticks(range(len(result.labels)))
    ax.set_yticklabels(result.labels)
    fig.colorbar(heat, ax=ax, shrink=0.8, label="Count")
    conf_path = output_path / f"{prefix}_confusion_matrix.png"
    fig.savefig(conf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(conf_path)

    fig_fold, ax_fold = plt.subplots(figsize=(5, 3))
    ax_fold.bar(np.arange(len(result.fold_scores)), result.fold_scores, color="#1f77b4")
    ax_fold.set_xlabel("Fold")
    ax_fold.set_ylabel("Accuracy")
    ax_fold.set_ylim(0, 1)
    ax_fold.set_title("Cross-Validation Accuracy")
    fold_path = output_path / f"{prefix}_fold_accuracy.png"
    fig_fold.savefig(fold_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_fold)
    artifacts.append(fold_path)

    return artifacts


def _radius_from_area(area: float) -> float:
    return math.sqrt(max(area, 0.0) / math.pi) if np.isfinite(area) else float("nan")


def _wrap_angle_array(values: np.ndarray) -> np.ndarray:
    wrapped = (values + math.pi) % (2 * math.pi) - math.pi
    return wrapped


def _normalize_rotation_label(label: str) -> str:
    value = str(label).strip().lower()
    if value in {"cw", "clockwise"}:
        return "cw"
    if value in {"ccw", "counterclockwise", "anticlockwise"}:
        return "ccw"
    if not value:
        return "unspecified"
    return value


def _mean_transverse_speed(frame_df: Optional[pd.DataFrame]) -> float:
    if frame_df is None or frame_df.empty:
        return float("nan")
    ordered = frame_df.sort_values("frame_idx")
    coords = ordered[["weighted_centroid_x", "weighted_centroid_y"]].to_numpy(dtype=float)
    if coords.shape[0] < 2:
        return float("nan")
    diffs = np.diff(coords, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    if distances.size == 0:
        return float("nan")
    return float(np.nanmean(distances))


def _trajectory_angle(frame_df: Optional[pd.DataFrame]) -> float:
    if frame_df is None or frame_df.empty:
        return float("nan")
    ordered = frame_df.sort_values("frame_idx")
    start = ordered.iloc[0]
    end = ordered.iloc[-1]
    dx = float(end.weighted_centroid_x - start.weighted_centroid_x)
    dy = float(end.weighted_centroid_y - start.weighted_centroid_y)
    if dx == 0 and dy == 0:
        return 0.0
    return float(math.atan2(dy, dx))


def _validate_class_balance(labels: np.ndarray, min_samples: int) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    if np.any(counts < max(1, min_samples)):
        failing = ", ".join(f"{label} ({count})" for label, count in zip(unique, counts) if count < min_samples)
        raise ValueError(f"Insufficient samples for classes: {failing}")


def _iter_cv_splits(
    labels: np.ndarray,
    *,
    strategy: CVStrategy,
    folds: int,
    random_state: Optional[int],
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    n_samples = labels.shape[0]
    if strategy == "leave_one_out":
        for idx in range(n_samples):
            train_idx = np.delete(np.arange(n_samples), idx)
            test_idx = np.array([idx])
            yield train_idx, test_idx
        return

    if folds < 2:
        raise ValueError("folds must be >= 2 for k-fold strategies")
    if folds > n_samples:
        folds = n_samples

    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)

    if strategy == "kfold":
        rng.shuffle(indices)
        fold_sizes = np.full(folds, n_samples // folds, dtype=int)
        fold_sizes[: n_samples % folds] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate((indices[:start], indices[stop:]))
            yield train_idx, test_idx
            current = stop
        return

    if strategy == "stratified_kfold":
        per_label_indices: Dict[str, List[int]] = {}
        for idx, label in enumerate(labels):
            per_label_indices.setdefault(label, []).append(idx)

        fold_bins: List[List[int]] = [[] for _ in range(folds)]
        for label, label_indices in per_label_indices.items():
            shuffled = np.array(label_indices)
            rng.shuffle(shuffled)
            for fold_idx, sample_idx in enumerate(shuffled):
                fold_bins[fold_idx % folds].append(sample_idx)

        for fold_idx in range(folds):
            test_idx = np.array(fold_bins[fold_idx])
            train_idx = np.array(sorted(set(indices) - set(test_idx)))
            yield train_idx, test_idx
        return

    raise ValueError(f"Unsupported CV strategy: {strategy}")


class _NearestCentroidClassifier:
    """Simple nearest-centroid classifier."""

    def __init__(self) -> None:
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        classes, inverse = np.unique(labels, return_inverse=True)
        centroids = []
        for class_index, cls in enumerate(classes):
            mask = inverse == class_index
            centroids.append(features[mask].mean(axis=0))
        self.centroids_ = np.vstack(centroids)
        self.labels_ = classes

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("Classifier must be fitted before prediction")
        diffs = features[:, None, :] - self.centroids_[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        indices = np.argmin(distances, axis=1)
        return self.labels_[indices]


def _confusion_matrix(
    truths: Sequence[str],
    predictions: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    labels = sorted(set(truths) | set(predictions))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(truths, predictions):
        matrix[label_to_idx[truth], label_to_idx[pred]] += 1
    return matrix, labels
