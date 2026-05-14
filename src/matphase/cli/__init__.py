"""
Command-line interface for MatPhase.

Entry points for running analysis pipelines from the command line.
"""

import argparse
import sys
from pathlib import Path

from matphase.config import load_config
from matphase.utils import get_logger, setup_logging
from matphase.cli.detection import add_detection_export_args, run_detection_export_cli
from matphase.cli.analysis import register_analysis_subcommand

__all__ = ["main"]

__version__ = "0.1.0"


def print_config_summary(config) -> None:
    """
    Print a formatted summary of the configuration.

    Args:
        config: MatPhaseConfig instance
    """
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("MatPhase Configuration Summary")
    logger.info("=" * 60)

    # Paths
    logger.info("Paths:")
    logger.info(f"  Data directory:    {config.paths.data_dir}")
    logger.info(f"  Output directory:  {config.paths.output_dir}")
    logger.info(f"  CIFTI file:        {config.paths.cifti_file}")
    logger.info(f"  Geometry file:     {config.paths.geometry_file}")

    # Preprocessing
    logger.info("Preprocessing:")
    logger.info(f"  Sigma scales:      {config.preprocessing.sigma_scale}")
    logger.info(f"  Downsample rate:   {config.preprocessing.downsample_rate}")
    logger.info(f"  Sampling rate:     {config.preprocessing.temporal_sampling_rate} Hz")
    logger.info(
        f"  Bandpass filter:   {config.preprocessing.filter_low_freq}-"
        f"{config.preprocessing.filter_high_freq} Hz"
    )

    # Detection
    logger.info("Pattern Detection:")
    logger.info(f"  Min duration:      {config.detection.min_pattern_duration} frames")
    logger.info(f"  Min size:          {config.detection.min_pattern_size} pixels")
    logger.info(f"  Connectivity:      {config.detection.connectivity}")

    # Compute
    logger.info("Compute:")
    logger.info(f"  GPU enabled:       {config.compute.use_gpu}")
    logger.info(f"  Parallel jobs:     {config.compute.n_jobs}")

    # Output
    logger.info("Output:")
    logger.info(f"  Log level:         {config.output.log_level}")
    logger.info(f"  Figure format:     {config.output.figure_format}")

    # Hemisphere
    logger.info("Hemisphere:")
    logger.info(f"  Processing:        {config.hemisphere.side}")

    logger.info("=" * 60)


def parse_args(args=None):
    """
    Parse command-line arguments.

    Args:
        args: List of arguments to parse (None = sys.argv)

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="matphase",
        description="MatPhase - fMRI phase-field spiral detection toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"MatPhase {__version__}",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration YAML file (default: use built-in defaults)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override logging level from config",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (default: console only)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory from config",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit without processing",
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command")
    register_analysis_subcommand(subparsers)

    # Add detection export arguments (available for root command)
    add_detection_export_args(parser)

    return parser.parse_args(args)


def main(args=None):
    """
    Main entry point for the MatPhase CLI.

    Args:
        args: Command-line arguments (None = sys.argv)

    Returns:
        Exit code (0 = success, 1 = error)
    """
    try:
        # Parse arguments
        parsed_args = parse_args(args)

        # Load configuration
        config_overrides = {}

        # Build overrides from CLI arguments
        if parsed_args.log_level is not None:
            config_overrides.setdefault("output", {})["log_level"] = parsed_args.log_level

        if parsed_args.log_file is not None:
            config_overrides.setdefault("output", {})["log_file"] = parsed_args.log_file

        if parsed_args.data_dir is not None:
            config_overrides.setdefault("paths", {})["data_dir"] = parsed_args.data_dir

        if parsed_args.output_dir is not None:
            config_overrides.setdefault("paths", {})["output_dir"] = parsed_args.output_dir

        # Load config with overrides
        config = load_config(
            config_path=parsed_args.config,
            apply_env=True,
            **config_overrides,
        )

        # Setup logging based on config
        setup_logging(
            level=config.output.log_level,
            log_file=config.output.log_file,
        )

        logger = get_logger(__name__)
        logger.info(f"MatPhase v{__version__} starting...")

        # Print configuration summary for visibility
        print_config_summary(config)

        # Hand off to subcommand handlers (analysis, etc.)
        if getattr(parsed_args, "subcommand_handler", None):
            return parsed_args.subcommand_handler(parsed_args, config)

        # In dry-run mode just show config and exit
        if parsed_args.dry_run:
            logger.info("Dry run mode - exiting without processing")
            return 0

        if _requested_detection_exports(parsed_args):
            logger.info("Running detection export workflow...")
            return run_detection_export_cli(parsed_args)

        logger.info("Full pipeline execution not yet implemented")
        logger.info("Session 4: CLI skeleton successfully configured")
        logger.info("Future sessions will implement preprocessing, detection, and analysis")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def _requested_detection_exports(args: argparse.Namespace) -> bool:
    """Return True if any detection export flags were set."""

    return any(
        getattr(args, attr, False)
        for attr in (
            "export_csv",
            "export_json",
            "export_summary",
            "export_plots",
            "export_statistics",
        )
    )
