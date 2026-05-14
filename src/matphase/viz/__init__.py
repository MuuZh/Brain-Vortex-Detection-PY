"""
Visualization and export utilities for matphase detection pipeline.

This module provides:
- Phase field visualization (gradients, curl, magnitude)
- Spiral detection overlays and trajectory plots
- Statistical summary plots and comparisons
- Export utilities (CSV, JSON, PNG)
"""

from matphase.viz.phase_field import (
    plot_phase_field,
    plot_curl_field,
    plot_gradient_field,
    save_phase_field_snapshot,
)

from matphase.viz.detection import (
    plot_spiral_overlays,
    plot_spiral_trajectories,
    plot_pattern_statistics,
    save_detection_frames,
)

from matphase.viz.statistics import (
    plot_detection_summary,
    plot_surrogate_comparison,
    plot_threshold_curves,
)

from matphase.viz.export import (
    export_detection_csv,
    export_detection_json,
    export_batch_plots,
)

__all__ = [
    # Phase field visualization
    "plot_phase_field",
    "plot_curl_field",
    "plot_gradient_field",
    "save_phase_field_snapshot",
    # Detection visualization
    "plot_spiral_overlays",
    "plot_spiral_trajectories",
    "plot_pattern_statistics",
    "save_detection_frames",
    # Statistical plots
    "plot_detection_summary",
    "plot_surrogate_comparison",
    "plot_threshold_curves",
    # Export utilities
    "export_detection_csv",
    "export_detection_json",
    "export_batch_plots",
]
