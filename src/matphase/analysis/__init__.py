"""
Analysis module for MatPhase.

Statistical analysis and reporting:
- Summary statistics
- Classification metrics
- Result aggregation
- Report generation
"""

from .contrast import (
    ContrastResult,
    ContrastSpec,
    TaskEvent,
    build_condition_frame_masks,
    compute_contrasts,
    save_contrast_artifacts,
)
from .classify import (
    ClassificationResult,
    build_pattern_feature_table,
    classify_patterns,
    save_classification_artifacts,
)
from .distribution import (
    MetricStats,
    SpiralDistributionMetrics,
    TemplateStats,
    compute_spiral_distribution_metrics,
    save_distribution_artifacts,
)
from .temporal_trends import (
    TemporalTrendResult,
    TrendStats,
    compute_temporal_trends,
    save_temporal_trend_csv,
    save_temporal_trend_plot,
)
from .storage import (
    SpiralAnalysisDataset,
    load_subject_bundle,
    save_subject_bundle_from_detection,
    write_storage_preview,
)
from .reporting import (
    AnalysisReport,
    ValidationRecord,
    render_markdown_report,
    render_html_report,
    save_report_files,
    summarize_classification,
    summarize_contrasts,
    summarize_distribution_metrics,
)

__all__ = [
    "ContrastResult",
    "ContrastSpec",
    "TaskEvent",
    "ClassificationResult",
    "MetricStats",
    "SpiralDistributionMetrics",
    "TemplateStats",
    "SpiralAnalysisDataset",
    "build_condition_frame_masks",
    "build_pattern_feature_table",
    "classify_patterns",
    "compute_contrasts",
    "compute_spiral_distribution_metrics",
    "save_classification_artifacts",
    "save_contrast_artifacts",
    "save_distribution_artifacts",
    "load_subject_bundle",
    "save_subject_bundle_from_detection",
    "write_storage_preview",
    "AnalysisReport",
    "ValidationRecord",
    "render_markdown_report",
    "render_html_report",
    "save_report_files",
    "summarize_classification",
    "summarize_contrasts",
    "summarize_distribution_metrics",
    "TemporalTrendResult",
    "TrendStats",
    "compute_temporal_trends",
    "save_temporal_trend_csv",
    "save_temporal_trend_plot",
]
