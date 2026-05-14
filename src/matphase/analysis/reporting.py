"""
Reporting utilities for Phase 5 analysis pipelines.

Provides helpers to summarize metrics and persist Markdown/HTML reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence

from matphase.analysis.classify import ClassificationResult
from matphase.analysis.contrast import ContrastResult
from matphase.analysis.distribution import MetricStats, SpiralDistributionMetrics


ValidationStatus = Literal["pass", "fail", "skipped"]


@dataclass
class ValidationRecord:
    """Represents the outcome of a validation or artifact generation step."""

    name: str
    status: ValidationStatus
    details: str = ""


@dataclass
class AnalysisReport:
    """In-memory representation of a subject- or cohort-level report."""

    subject_id: str
    cohort_name: str
    metadata: Dict[str, str]
    metrics_summary: Dict[str, str]
    contrast_summaries: List[Dict[str, str]] = field(default_factory=list)
    classification_summary: Optional[Dict[str, str]] = None
    artifact_paths: Dict[str, List[Path]] = field(default_factory=dict)
    validations: List[ValidationRecord] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def summarize_distribution_metrics(metrics: SpiralDistributionMetrics) -> Dict[str, str]:
    """Convert spiral distribution metrics into printable strings."""

    return {
        "rotation_label": metrics.rotation_label,
        "mean_patterns_per_subject": _format_metric(metrics.count_stats),
        "mean_duration": _format_metric(metrics.duration_stats, unit="frames"),
        "mean_radius": _format_metric(metrics.radius_stats, unit="px"),
        "mean_transverse_speed": _format_metric(metrics.transverse_speed_stats, unit="px/frame"),
    }


def summarize_contrasts(results: Sequence[ContrastResult]) -> List[Dict[str, str]]:
    """Summarize contrast outputs for reporting."""

    summaries: List[Dict[str, str]] = []
    for result in results:
        summaries.append(
            {
                "name": result.name,
                "numerator": ", ".join(result.numerator_labels),
                "denominator": ", ".join(result.denominator_labels),
                "mean_difference": f"{float(result.summary.get('mean_difference', float('nan'))):.4f}",
                "mean_percent_change": f"{float(result.summary.get('mean_percent_change', float('nan'))):.4f}",
                "numerator_frames": str(int(result.summary.get("numerator_frames", 0))),
                "denominator_frames": str(int(result.summary.get("denominator_frames", 0))),
            }
        )
    return summaries


def summarize_classification(result: ClassificationResult) -> Dict[str, str]:
    """Summarize classification performance for reporting."""

    return {
        "classifier": result.classifier_type,
        "cv_strategy": result.cv_strategy,
        "accuracy": f"{result.accuracy:.4f}",
        "fold_scores": ", ".join(f"{score:.3f}" for score in result.fold_scores),
        "labels": ", ".join(result.labels),
    }


def render_markdown_report(
    report: AnalysisReport,
    *,
    include_artifact_table: bool = True,
) -> str:
    """Render an AnalysisReport into Markdown text."""

    header = [
        f"# MatPhase Analysis Report — {report.subject_id}",
        "",
        f"- Cohort: **{report.cohort_name}**",
        f"- Analysis window: {report.start_time.isoformat()} → {report.end_time.isoformat()}",
        f"- Total duration: {_format_duration(report.start_time, report.end_time)}",
        "",
    ]

    metadata_section = ["## Metadata", "", "| Key | Value |", "| --- | --- |"]
    for key, value in sorted(report.metadata.items()):
        metadata_section.append(f"| {key} | {value} |")
    metadata_section.append("")

    metrics_section = ["## Key Metrics", ""]
    for key, value in report.metrics_summary.items():
        metrics_section.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    metrics_section.append("")

    contrast_section = []
    if report.contrast_summaries:
        contrast_section.extend(["## Contrast Results", "", "| Contrast | Numerator | Denominator | Δ mean | % change | Frames (num/den) |", "| --- | --- | --- | --- | --- | --- |"])
        for summary in report.contrast_summaries:
            contrast_section.append(
                "| {name} | {numerator} | {denominator} | {mean_difference} | {mean_percent_change} | {numerator_frames}/{denominator_frames} |".format(
                    **summary
                )
            )
        contrast_section.append("")

    classification_section = []
    if report.classification_summary:
        classification_section.extend(
            [
                "## Classification Summary",
                "",
                "| Metric | Value |",
                "| --- | --- |",
            ]
        )
        for key, value in report.classification_summary.items():
            classification_section.append(f"| {key} | {value} |")
        classification_section.append("")

    artifact_section = []
    if include_artifact_table and report.artifact_paths:
        artifact_section.extend(["## Artifacts", "", "| Category | Path |", "| --- | --- |"])
        for category, paths in report.artifact_paths.items():
            for path in paths:
                artifact_section.append(f"| {category} | `{path.as_posix()}` |")
        artifact_section.append("")

    validation_section = ["## Validation Log", "", "| Step | Status | Details |", "| --- | --- | --- |"]
    for record in report.validations:
        validation_section.append(
            f"| {record.name} | {record.status.upper()} | {record.details or '-'} |"
        )
    validation_section.append("")

    sections = [
        *header,
        *metadata_section,
        *metrics_section,
        *contrast_section,
        *classification_section,
        *artifact_section,
        *validation_section,
    ]
    return "\n".join(sections)


def render_html_report(markdown_text: str, *, title: str = "MatPhase Analysis Report") -> str:
    """
    Create a minimal HTML document containing the Markdown text.

    This avoids adding a Markdown dependency while still providing an HTML
    container that is easy to preview in a browser.
    """

    escaped = markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<meta charset='utf-8'><title>{title}</title>",
            "<style>body { font-family: Consolas, 'Courier New', monospace; white-space: pre-wrap; }</style>",
            "</head>",
            "<body>",
            escaped,
            "</body>",
            "</html>",
        ]
    )


def save_report_files(
    report: AnalysisReport,
    output_dir: Path,
    *,
    formats: Sequence[Literal["markdown", "html"]] = ("markdown",),
) -> Dict[str, Path]:
    """Persist Markdown/HTML representations to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_text = render_markdown_report(report)

    saved: Dict[str, Path] = {}
    if "markdown" in formats:
        md_path = output_dir / f"{report.subject_id}_analysis_report.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        saved["markdown"] = md_path
    if "html" in formats:
        html_text = render_html_report(markdown_text, title=f"{report.subject_id} Analysis Report")
        html_path = output_dir / f"{report.subject_id}_analysis_report.html"
        html_path.write_text(html_text, encoding="utf-8")
        saved["html"] = html_path
    return saved


def _format_metric(stats: MetricStats, unit: Optional[str] = None) -> str:
    if stats.n == 0 or any(math.isnan(value) for value in (stats.mean, stats.std)):
        return "n/a"
    value = f"{stats.mean:.3f} ± {stats.std:.3f}"
    if unit:
        value = f"{value} {unit}"
    return f"{value} (n={stats.n})"


def _format_duration(start: datetime, end: datetime) -> str:
    seconds = max(0.0, (end - start).total_seconds())
    return f"{seconds:.2f} s"

