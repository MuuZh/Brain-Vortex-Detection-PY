#!/usr/bin/env python
"""
Batch runner for MatPhase detection bundles.

Runs multiple subjects with shared parameters while varying CIFTI path
and hemisphere per entry. Supports bounded concurrency.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() in {".json"}:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON manifest must be a list of objects.")
        return [dict(row) for row in data]

    # Assume CSV/TSV based on delimiter
    delimiter = "\t" if path.suffix.lower() in {".tsv"} else ","
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(row) for row in reader]


def _build_command(
    base_args: argparse.Namespace,
    row: Dict[str, str],
) -> List[str]:
    cmd = [
        "python",
        "scripts/run_full_detection_bundle.py",
        "--config",
        str(base_args.config),
        "--hemisphere",
        row["hemisphere"],
        "--cifti-file",
        row["cifti_file"],
    ]

    if base_args.bundle_root:
        cmd += ["--bundle-root", str(base_args.bundle_root)]
    if base_args.bundle_suffix:
        cmd += ["--bundle-suffix", str(base_args.bundle_suffix)]
    if base_args.output_dir:
        cmd += ["--output-dir", str(base_args.output_dir)]
    if base_args.data_dir:
        cmd += ["--data-dir", str(base_args.data_dir)]
    if base_args.show_progress:
        cmd.append("--show-progress")
    if base_args.rotation_mode:
        cmd += ["--rotation-mode", base_args.rotation_mode]

    # Optional per-row overrides
    if "bundle_root" in row and row["bundle_root"]:
        cmd += ["--bundle-root", row["bundle_root"]]
    if "bundle_suffix" in row and row["bundle_suffix"]:
        cmd += ["--bundle-suffix", row["bundle_suffix"]]
    if "log_file" in row and row["log_file"]:
        cmd += ["--log-file", row["log_file"]]

    return cmd


def _run_command(cmd: List[str]) -> int:
    proc = subprocess.Popen(cmd)
    return proc.wait()


def _iter_rows(manifest: List[Dict[str, str]]) -> Iterable[Dict[str, str]]:
    required = {"cifti_file", "hemisphere"}
    for idx, row in enumerate(manifest, start=1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {idx} is missing required fields: {missing}")
        yield row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for MatPhase detection over multiple subjects."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest (CSV/TSV/JSON) with columns at least: cifti_file, hemisphere.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/defaults.yaml"),
        help="Path to base configuration YAML.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of concurrent runs.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Default bundle root for all runs (can be overridden per row).",
    )
    parser.add_argument(
        "--bundle-suffix",
        type=str,
        default=None,
        help="Default bundle suffix for all runs (can be overridden per row).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override paths.output_dir for all runs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override paths.data_dir for all runs.",
    )
    parser.add_argument(
        "--rotation-mode",
        choices=["both", "ccw", "cw"],
        default=None,
        help="Override detection.rotation_mode for all runs.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Enable progress bars in child runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = _load_manifest(args.manifest)
    if not manifest:
        raise ValueError("Manifest is empty.")

    futures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for row in _iter_rows(manifest):
            cmd = _build_command(args, row)
            futures.append(executor.submit(_run_command, cmd))

        exit_codes = [f.result() for f in as_completed(futures)]

    failed = sum(code != 0 for code in exit_codes)
    if failed:
        raise SystemExit(f"{failed} run(s) failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
