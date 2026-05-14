"""
Analysis CLI entry point for MatPhase (Phase 5 Session 4).

Provides orchestration for loading subject bundles, computing distribution,
contrast, and classification metrics, and emitting Markdown/HTML reports.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from matphase.analysis import (
    AnalysisReport,
    ValidationRecord,
    build_condition_frame_masks,
    classify_patterns,
    compute_contrasts,
    compute_spiral_distribution_metrics,
    save_classification_artifacts,
    save_contrast_artifacts,
    save_distribution_artifacts,
    save_report_files,
    summarize_classification,
    summarize_contrasts,
    summarize_distribution_metrics,
    write_storage_preview,
)
from matphase.analysis.classify import build_pattern_feature_table
from matphase.analysis.contrast import ContrastSpec, TaskEvent
from matphase.analysis.storage import SpiralAnalysisDataset, load_subject_bundle
from matphase.config.schema import MatPhaseConfig
from matphase.utils import get_logger


logger = get_logger(__name__)


@dataclass
class SubjectSpec:
    """Subject configuration for the analysis CLI."""

    subject_id: Optional[str] = None
    bundle_path: Optional[Path] = None
    dataset: Optional[SpiralAnalysisDataset] = None
    label_file: Optional[Path] = None
    design_file: Optional[Path] = None
    extra_metadata: Dict[str, str] = field(default_factory=dict)

    def require_dataset(self) -> SpiralAnalysisDataset:
        """Return or load the dataset associated with this spec."""

        if self.dataset is not None:
            return self.dataset
        if not self.bundle_path:
            raise ValueError("SubjectSpec requires either dataset or bundle_path")
        return load_subject_bundle(self.bundle_path)


def register_analysis_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'analysis' subcommand with the main CLI parser."""

    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Run post-detection analysis pipelines and generate summary reports",
    )
    analysis_parser.add_argument(
        "--bundle",
        dest="bundles",
        action="append",
        type=Path,
        help="Path to a saved SpiralAnalysisDataset bundle (metadata + parquet files)",
    )
    analysis_parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV/TSV manifest describing bundles (columns: bundle_dir, subject_id, label_file, design_file, ...)",
    )
    analysis_parser.add_argument(
        "--report-dir",
        type=Path,
        help="Output directory for generated reports (default: config.analysis.report.output_root/<prefix>_<timestamp>)",
    )
    analysis_parser.add_argument(
        "--report-prefix",
        type=str,
        help="Folder prefix for report runs (default: config.analysis.report.prefix)",
    )
    analysis_parser.add_argument(
        "--report-format",
        dest="report_formats",
        action="append",
        choices=["markdown", "html"],
        help="Report formats to save (default: config.analysis.report.formats)",
    )
    analysis_parser.add_argument(
        "--rotation",
        choices=["all", "cw", "ccw"],
        default="all",
        help="Rotation filter for distribution metrics (all=combined)",
    )
    analysis_parser.add_argument(
        "--cohort-name",
        type=str,
        default="cohort",
        help="Label for the cohort/run recorded in reports",
    )
    analysis_parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a synthetic demo dataset (useful for smoke tests without bundles)",
    )
    analysis_parser.add_argument(
        "--max-subjects",
        type=int,
        help="Optional limit on number of subjects processed (debugging aid)",
    )
    analysis_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars for subject iteration",
    )
    analysis_parser.add_argument(
        "--metadata",
        dest="metadata_overrides",
        action="append",
        default=[],
        help="Additional metadata key=value entries to inject into every report",
    )
    analysis_parser.set_defaults(subcommand_handler=run_analysis_cli)


def run_analysis_cli(args: argparse.Namespace, config: MatPhaseConfig) -> int:
    """Entry point for the analysis CLI subcommand."""

    specs = _resolve_subject_specs(args, config)
    if not specs:
        logger.error("No subject bundles or demo data supplied for analysis")
        return 1

    if args.max_subjects is not None:
        specs = specs[: args.max_subjects]

    report_root = _resolve_report_root(args, config)
    formats = _resolve_report_formats(args, config)
    rotation = None if args.rotation == "all" else args.rotation
    metadata_overrides = dict(_parse_key_value_pairs(args.metadata_overrides or []))

    subject_iterator: Iterable[SubjectSpec] = specs
    if not args.no_progress and len(specs) > 1:
        subject_iterator = tqdm(
            specs,
            desc="Analysis subjects",
            unit="subject",
        )

    failures = 0
    for spec in subject_iterator:
        subject_start = datetime.now(timezone.utc)
        dataset = spec.require_dataset()
        subject_id = spec.subject_id or _infer_subject_id(spec, dataset)
        safe_subject = _safe_name(subject_id)
        subject_output = report_root / safe_subject
        subject_output.mkdir(parents=True, exist_ok=True)

        metadata = {
            "bundle_path": str(spec.bundle_path) if spec.bundle_path else "in-memory",
            "cifti_file": dataset.metadata.get("cifti_file", "unknown"),
            "grid": f"{dataset.metadata.get('grid_height', '?')}x{dataset.metadata.get('grid_width', '?')}",
            "frame_count": str(dataset.metadata.get("frame_count", "?")),
            **spec.extra_metadata,
            **metadata_overrides,
        }

        validations: List[ValidationRecord] = []
        artifact_paths: Dict[str, List[Path]] = {}

        # Distribution metrics
        metrics_summary: Dict[str, str]
        try:
            metrics = compute_spiral_distribution_metrics(
                dataset,
                rotation=rotation,
                show_progress=not args.no_progress,
            )
            metrics_summary = summarize_distribution_metrics(metrics)
            dist_paths = save_distribution_artifacts(
                metrics,
                subject_output / "distribution",
                prefix=f"{safe_subject}_distribution",
            )
            artifact_paths["distribution"] = dist_paths
            validations.append(
                ValidationRecord(
                    name="distribution_metrics",
                    status="pass",
                    details=f"rotation={metrics.rotation_label}",
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Distribution metrics failed for %s", subject_id)
            validations.append(
                ValidationRecord(
                    name="distribution_metrics",
                    status="fail",
                    details=str(exc),
                )
            )
            metrics_summary = {"error": "Distribution metrics failed"}
            failures += 1
            continue  # Cannot build report without distribution metrics

        # Contrast analysis (optional)
        contrast_summaries: List[Dict[str, str]] = []
        if spec.design_file or config.analysis.contrast.design_file:
            try:
                design_file = spec.design_file or config.analysis.contrast.design_file
                if design_file is None:
                    raise FileNotFoundError("Design file missing")
                events = _load_task_events(design_file)
                frame_count = int(dataset.metadata.get("frame_count", 0))
                masks = build_condition_frame_masks(
                    frame_count,
                    events,
                    onset_buffer=config.analysis.task_onset_buffer,
                    offset_buffer=config.analysis.task_offset_buffer,
                )
                specs_to_run = _derive_contrast_specs(
                    events,
                    baseline=config.analysis.contrast.baseline_condition,
                    epsilon=config.analysis.contrast.percent_change_epsilon,
                )
                if specs_to_run:
                    contrast_results = compute_contrasts(
                        dataset,
                        masks,
                        specs_to_run,
                        show_progress=not args.no_progress,
                    )
                    contrast_summaries = summarize_contrasts(contrast_results)
                    contrast_paths = save_contrast_artifacts(
                        contrast_results,
                        subject_output / "contrast",
                        prefix=f"{safe_subject}_contrast",
                    )
                    artifact_paths["contrast"] = contrast_paths
                    validations.append(
                        ValidationRecord(
                            name="contrast_analysis",
                            status="pass",
                            details=f"{len(contrast_results)} contrasts",
                        )
                    )
                else:
                    validations.append(
                        ValidationRecord(
                            name="contrast_analysis",
                            status="skipped",
                            details="No eligible contrasts (insufficient unique conditions).",
                        )
                    )
            except FileNotFoundError as exc:
                logger.warning("Skipping contrast analysis for %s: %s", subject_id, exc)
                validations.append(
                    ValidationRecord(
                        name="contrast_analysis",
                        status="skipped",
                        details=str(exc),
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Contrast analysis failed for %s", subject_id)
                validations.append(
                    ValidationRecord(
                        name="contrast_analysis",
                        status="fail",
                        details=str(exc),
                    )
                )
                failures += 1

        # Classification (optional)
        classification_summary = None
        label_file = spec.label_file or config.analysis.classification.label_file
        if label_file:
            try:
                feature_table = build_pattern_feature_table(
                    dataset,
                    wrap_angles=config.analysis.classification.wrap_angles,
                )
                labeled_table, label_column = _merge_labels(feature_table, label_file)
                classification_result = classify_patterns(
                    labeled_table,
                    label_column=label_column,
                    feature_columns=config.analysis.classification.feature_columns or None,
                    classifier_type=config.analysis.classifier_type,
                    cv_strategy=config.analysis.classification.cv_strategy,
                    folds=config.analysis.classification.cross_validation_folds,
                    random_state=config.analysis.classification.random_state,
                    min_class_samples=config.analysis.classification.min_class_samples,
                    show_progress=not args.no_progress,
                )
                classification_summary = summarize_classification(classification_result)
                class_paths = save_classification_artifacts(
                    classification_result,
                    subject_output / "classification",
                    prefix=f"{safe_subject}_classification",
                )
                artifact_paths["classification"] = class_paths
                validations.append(
                    ValidationRecord(
                        name="pattern_classification",
                        status="pass",
                        details=f"accuracy={classification_result.accuracy:.3f}",
                    )
                )
            except FileNotFoundError as exc:
                logger.warning("Classification labels missing for %s: %s", subject_id, exc)
                validations.append(
                    ValidationRecord(
                        name="pattern_classification",
                        status="skipped",
                        details=str(exc),
                    )
                )
            except ValueError as exc:
                logger.warning("Classification skipped for %s: %s", subject_id, exc)
                validations.append(
                    ValidationRecord(
                        name="pattern_classification",
                        status="skipped",
                        details=str(exc),
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Classification failed for %s", subject_id)
                validations.append(
                    ValidationRecord(
                        name="pattern_classification",
                        status="fail",
                        details=str(exc),
                    )
                )
                failures += 1

        # Storage preview (always attempt)
        try:
            preview_paths = write_storage_preview(dataset, subject_output / "storage_preview")
            artifact_paths["storage_preview"] = preview_paths
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Storage preview failed for %s: %s", subject_id, exc)

        report = AnalysisReport(
            subject_id=safe_subject,
            cohort_name=args.cohort_name,
            metadata=metadata,
            metrics_summary=metrics_summary,
            contrast_summaries=contrast_summaries,
            classification_summary=classification_summary,
            artifact_paths=artifact_paths,
            validations=validations,
            start_time=subject_start,
            end_time=datetime.now(timezone.utc),
        )

        saved_reports = save_report_files(
            report,
            subject_output,
            formats=formats,
        )
        artifact_paths["reports"] = list(saved_reports.values())
        logger.info("Report generated for %s -> %s", subject_id, subject_output)

    if failures > 0:
        logger.warning("Analysis completed with %d failure(s)", failures)
    else:
        logger.info("Analysis completed successfully for %d subject(s)", len(specs))
    return 1 if failures > 0 else 0


def _resolve_subject_specs(args: argparse.Namespace, config: MatPhaseConfig) -> List[SubjectSpec]:
    specs: List[SubjectSpec] = []

    # CLI bundles
    for bundle in args.bundles or []:
        specs.append(SubjectSpec(bundle_path=bundle))

    # Manifest entries
    manifest_path = args.manifest or config.analysis.cli.subjects_manifest
    if manifest_path:
        specs.extend(_parse_manifest(manifest_path))

    # Config default bundles (when no CLI override)
    if not specs:
        for bundle in config.analysis.cli.default_bundles or []:
            specs.append(SubjectSpec(bundle_path=bundle))

    # Demo dataset
    if args.demo:
        specs.append(SubjectSpec(subject_id="demo_subject", dataset=_build_demo_dataset()))

    return specs


def _parse_manifest(path: Path) -> List[SubjectSpec]:
    specs: List[SubjectSpec] = []
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            bundle_dir = row.get("bundle_dir") or row.get("bundle_path")
            if not bundle_dir:
                continue
            bundle_path = (path.parent / bundle_dir).resolve()
            extra_meta = {k: v for k, v in row.items() if k not in {"bundle_dir", "bundle_path", "subject_id", "label_file", "design_file"} and v}
            label_file = row.get("label_file")
            design_file = row.get("design_file")
            specs.append(
                SubjectSpec(
                    subject_id=row.get("subject_id"),
                    bundle_path=bundle_path,
                    label_file=(path.parent / label_file).resolve() if label_file else None,
                    design_file=(path.parent / design_file).resolve() if design_file else None,
                    extra_metadata=extra_meta,
                )
            )
    return specs


def _resolve_report_root(args: argparse.Namespace, config: MatPhaseConfig) -> Path:
    if args.report_dir:
        root = args.report_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = args.report_prefix or config.analysis.report.prefix
        root = Path(config.analysis.report.output_root) / f"{prefix}_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_report_formats(args: argparse.Namespace, config: MatPhaseConfig) -> Sequence[str]:
    if args.report_formats:
        return args.report_formats
    return config.analysis.report.formats or ["markdown"]


def _infer_subject_id(spec: SubjectSpec, dataset: SpiralAnalysisDataset) -> str:
    if dataset.metadata.get("subject_id"):
        return str(dataset.metadata["subject_id"])
    cifti = Path(dataset.metadata.get("cifti_file", "subject")).stem
    if cifti:
        return cifti
    if spec.bundle_path:
        return spec.bundle_path.name
    return "subject"


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    safe = safe.strip("_") or "subject"
    return safe


def _parse_key_value_pairs(pairs: Sequence[str]) -> Sequence[tuple[str, str]]:
    parsed = []
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            parsed.append((key, value))
    return parsed


def _load_task_events(path: Path) -> List[TaskEvent]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    events: List[TaskEvent] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            condition = row.get("condition") or row.get("label")
            onset = row.get("onset")
            duration = row.get("duration")
            if condition is None or onset is None or duration is None:
                continue
            events.append(
                TaskEvent(
                    condition=str(condition).strip(),
                    onset=int(float(onset)),
                    duration=int(float(duration)),
                )
            )
    if not events:
        raise ValueError(f"No events found in design file {path}")
    return events


def _derive_contrast_specs(
    events: Sequence[TaskEvent],
    *,
    baseline: str,
    epsilon: float,
) -> List[ContrastSpec]:
    conditions = sorted({event.condition for event in events})
    if baseline not in conditions:
        return []
    specs: List[ContrastSpec] = []
    for condition in conditions:
        if condition == baseline:
            continue
        specs.append(
            ContrastSpec(
                name=f"{condition}-vs-{baseline}",
                numerator=[condition],
                denominator=[baseline],
                percent_change_epsilon=epsilon,
            )
        )
    return specs


def _merge_labels(feature_table: pd.DataFrame, label_file: Path) -> tuple[pd.DataFrame, str]:
    delimiter = "\t" if label_file.suffix.lower() in {".tsv", ".tab"} else ","
    labels = pd.read_csv(label_file, delimiter=delimiter)
    if "label" not in labels.columns:
        raise ValueError(f"Label file {label_file} must contain a 'label' column")
    join_cols = ["pattern_id"]
    if "subject_id" in labels.columns and "subject_id" in feature_table.columns:
        join_cols.append("subject_id")
    merged = feature_table.merge(labels, on=join_cols, how="left", suffixes=("", "_label"))
    if merged["label"].isna().all():
        raise ValueError(f"No matching labels found in {label_file}")
    return merged, "label"


def _build_demo_dataset() -> SpiralAnalysisDataset:
    """Create a small in-memory dataset for smoke testing."""

    grid = (6, 6, 4)
    pattern_coords = {
        1: np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=int,
        ),
        2: np.array(
            [
                [3, 3, 2],
                [3, 4, 2],
                [4, 3, 3],
            ],
            dtype=int,
        ),
    }

    from matphase.detect.spirals import SpiralDetectionResult, SpiralPattern  # Local import to avoid heavy cost

    spiral_patterns: List[SpiralPattern] = []
    labeled_volume = np.zeros(grid, dtype=int)

    for pattern_id, coords in pattern_coords.items():
        y, x, t = coords[:, 0], coords[:, 1], coords[:, 2]
        unique_times = np.unique(t)
        rotation = "ccw" if pattern_id == 1 else "cw"
        curl_sign = 1 if rotation == "ccw" else -1

        centroids = []
        inst_sizes = []
        inst_powers = []
        inst_peak = []
        inst_widths = []

        for frame in unique_times:
            mask = t == frame
            frame_y = y[mask]
            frame_x = x[mask]
            count = frame_y.size
            inst_sizes.append(float(count))
            centroids.append([float(np.mean(frame_x)), float(np.mean(frame_y))])
            inst_powers.append(float(count * (pattern_id + 0.5)))
            inst_peak.append(float(pattern_id))
            width = (
                float(np.max(frame_y) - np.min(frame_y) + 1)
                + float(np.max(frame_x) - np.min(frame_x) + 1)
            ) / 2.0
            inst_widths.append(width)

        voxel_indices = np.ravel_multi_index((y, x, t), grid)
        labeled_volume.flat[voxel_indices] = pattern_id

        spiral_patterns.append(
            SpiralPattern(
                pattern_id=pattern_id,
                duration=len(unique_times),
                start_time=int(unique_times[0]),
                end_time=int(unique_times[-1]),
                absolute_times=unique_times.astype(int),
                total_size=int(coords.shape[0]),
                centroids=np.asarray(centroids, dtype=float),
                weighted_centroids=np.asarray(centroids, dtype=float),
                instantaneous_sizes=np.asarray(inst_sizes, dtype=float),
                instantaneous_powers=np.asarray(inst_powers, dtype=float),
                instantaneous_peak_amps=np.asarray(inst_peak, dtype=float),
                instantaneous_widths=np.asarray(inst_widths, dtype=float),
                bounding_box=(
                    int(np.min(x)),
                    int(np.max(x)),
                    int(np.min(y)),
                    int(np.max(y)),
                    int(np.min(t)),
                    int(np.max(t)),
                ),
                rotation_direction=rotation,
                curl_sign=curl_sign,
                voxel_indices=voxel_indices,
            )
        )

    detection = SpiralDetectionResult(
        patterns=spiral_patterns,
        num_patterns=len(spiral_patterns),
        labeled_volume=labeled_volume,
        input_shape=grid,
        detection_params={"min_duration": 1, "min_size": 1},
        statistics={"total_patterns": len(spiral_patterns)},
        rotation_direction="mixed",
        curl_sign=None,
    )
    return SpiralAnalysisDataset.from_detection_result(detection, "demo_subject.dtseries.nii")
