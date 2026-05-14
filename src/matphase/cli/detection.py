"""
CLI commands for spiral detection and export.

Provides command-line interface for running detection pipelines and exporting results.
"""

import argparse
from pathlib import Path
from typing import Optional

from matphase.config import load_config
from matphase.utils import get_logger
from matphase.detect.spirals import SpiralDetectionResult
from matphase.viz.export import (
    export_detection_csv,
    export_detection_json,
    export_summary_report,
)
from matphase.viz.detection import save_detection_frames
from matphase.viz.statistics import plot_detection_summary


def add_detection_export_args(parser: argparse.ArgumentParser) -> None:
    """
    Add detection export arguments to parser.

    Parameters:
        parser: ArgumentParser to add arguments to
    """
    export_group = parser.add_argument_group('Detection Export Options')

    export_group.add_argument(
        '--export-csv',
        action='store_true',
        help='Export detection results to CSV format',
    )

    export_group.add_argument(
        '--export-json',
        action='store_true',
        help='Export detection results to JSON format',
    )

    export_group.add_argument(
        '--export-summary',
        action='store_true',
        help='Export human-readable summary report',
    )

    export_group.add_argument(
        '--export-plots',
        action='store_true',
        help='Export detection overlay plots for all timepoints',
    )

    export_group.add_argument(
        '--export-statistics',
        action='store_true',
        help='Export statistical summary plots',
    )

    export_group.add_argument(
        '--export-prefix',
        type=str,
        default='detection_results',
        help='Prefix for exported files (default: detection_results)',
    )

    export_group.add_argument(
        '--export-dir',
        type=Path,
        default=None,
        help='Directory for exported files (default: from config output_dir)',
    )

    export_group.add_argument(
        '--plot-timepoints',
        type=str,
        default=None,
        help='Timepoints to export plots for (e.g., "0,5,10" or "0-20")',
    )

    export_group.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars for batch exports',
    )

    export_group.add_argument(
        '--rotation-mode',
        choices=['both', 'ccw', 'cw'],
        default='both',
        help='Rotation selection for spiral detection (ccw = positive curl, cw = negative curl)',
    )


def parse_timepoints(timepoints_str: Optional[str], max_timepoints: int) -> Optional[list]:
    """
    Parse timepoints specification string.

    Parameters:
        timepoints_str: String like "0,5,10" or "0-20" or None
        max_timepoints: Maximum timepoint index

    Returns:
        List of timepoint indices or None for all
    """
    if timepoints_str is None:
        return None

    # Range specification: "0-20"
    if '-' in timepoints_str and ',' not in timepoints_str:
        start, end = timepoints_str.split('-')
        return list(range(int(start), min(int(end) + 1, max_timepoints)))

    # Comma-separated list: "0,5,10"
    elif ',' in timepoints_str:
        return [int(t) for t in timepoints_str.split(',') if int(t) < max_timepoints]

    # Single value
    else:
        t = int(timepoints_str)
        return [t] if t < max_timepoints else []


def export_detection_results(
    result: SpiralDetectionResult,
    curl_field,
    export_dir: Path,
    prefix: str = 'detection_results',
    export_csv: bool = True,
    export_json: bool = True,
    export_summary: bool = True,
    export_plots: bool = False,
    export_statistics: bool = True,
    plot_timepoints: Optional[list] = None,
    show_progress: bool = True,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Export detection results in multiple formats.

    Parameters:
        result: SpiralDetectionResult to export
        curl_field: Curl field array for visualization
        export_dir: Directory for exported files
        prefix: Filename prefix
        export_csv: Export CSV files
        export_json: Export JSON file
        export_summary: Export text summary report
        export_plots: Export detection overlay plots
        export_statistics: Export statistical plots
        plot_timepoints: List of timepoints for plots (None = all)
        show_progress: Show progress bars
        metadata: Optional metadata for reports

    Returns:
        Dictionary of exported file paths
    """
    logger = get_logger(__name__)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    exported_files = {}

    # CSV export
    if export_csv:
        logger.info("Exporting CSV files...")
        csv_path = export_dir / prefix
        export_detection_csv(result, csv_path, include_statistics=True)
        exported_files['csv_patterns'] = export_dir / f"{prefix}_patterns.csv"
        exported_files['csv_timepoints'] = export_dir / f"{prefix}_timepoints.csv"
        logger.info(f"  -> {exported_files['csv_patterns']}")
        logger.info(f"  -> {exported_files['csv_timepoints']}")

    # JSON export
    if export_json:
        logger.info("Exporting JSON file...")
        json_path = export_dir / f"{prefix}.json"
        export_detection_json(result, json_path, include_voxel_indices=False)
        exported_files['json'] = json_path
        logger.info(f"  -> {json_path}")

    # Summary report
    if export_summary:
        logger.info("Exporting summary report...")
        summary_path = export_dir / f"{prefix}_summary.txt"
        export_summary_report(result, summary_path, metadata=metadata)
        exported_files['summary'] = summary_path
        logger.info(f"  -> {summary_path}")

    # Detection plots
    if export_plots and curl_field is not None:
        logger.info("Exporting detection overlay plots...")
        plots_dir = export_dir / "plots"
        plot_paths = save_detection_frames(
            curl_field,
            result,
            plots_dir,
            prefix=f"{prefix}_overlay",
            timepoints=plot_timepoints,
            show_progress=show_progress,
        )
        exported_files['plots'] = plot_paths
        logger.info(f"  -> {len(plot_paths)} plots saved to {plots_dir}")

    # Statistical plots
    if export_statistics:
        logger.info("Exporting statistical plots...")
        import matplotlib.pyplot as plt

        stats_dir = export_dir / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # Detection summary plot
        summary_fig = plot_detection_summary(
            result.statistics,
            title=f"Detection Summary: {prefix}"
        )
        summary_path = stats_dir / f"{prefix}_statistics_summary.png"
        summary_fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(summary_fig)
        exported_files['stats_summary'] = summary_path
        logger.info(f"  -> {summary_path}")

    logger.info(f"Export complete: {len(exported_files)} file(s) created")
    return exported_files


def run_detection_export_cli(args):
    """
    Run detection export from command-line arguments.

    Parameters:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    logger = get_logger(__name__)

    # Check if any export option is enabled
    if not any([
        args.export_csv,
        args.export_json,
        args.export_summary,
        args.export_plots,
        args.export_statistics,
    ]):
        logger.warning("No export options specified. Use --export-csv, --export-json, etc.")
        return 0

    # Load configuration
    config = load_config(config_path=args.config if hasattr(args, 'config') else None)

    # Determine export directory
    export_dir = args.export_dir if args.export_dir else config.paths.output_dir / "exports"

    logger.info(f"Export directory: {export_dir}")

    # TODO: In a real pipeline, this would load detection results from a previous run
    # For now, this is a placeholder showing how the CLI would work
    logger.warning("Detection export CLI requires integration with full pipeline")
    logger.info("This functionality will be used by the main pipeline command")

    return 0
