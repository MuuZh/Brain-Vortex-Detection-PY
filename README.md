# MatPhase

Python toolkit for fMRI phase-field spiral detection and analysis.

## Overview

MatPhase detects spiral (vortex) patterns in fMRI phase fields using spatial interpolation, bandpass filtering, curl-based detection, and optional surrogate thresholds. This bundle includes only the pieces needed to run the detection pipeline and write subject bundles.

## Features

- CIFTI loading and flat-surface geometry handling
- Temporal + spatial filtering and Hilbert phase extraction
- Spiral detection with curl thresholds and optional expansion/compatibility filters
- Bundle writer (`metadata.json`, `patterns.parquet`, `frame_index.parquet`, `coords.feather`, optional `phase_cube.npy`)
- Batch runner for multiple subjects

## Project Status

Detection pipeline is ready for sharing; tests, analysis extras, and heavy data are omitted in this zip-friendly bundle.

## Installation

Prereqs: Python 3.11+, Conda recommended.

```powershell
# after unzipping
cd matphase_detection_bundle
conda run -n base pip install -e .
```

GPU/Viz extras (optional):

```powershell
conda run -n base pip install -e ".[gpu]"
conda run -n base pip install -e ".[viz]"
```

## Quick Start

See `USAGE.md` for detailed inputs/outputs and dimensions. Typical commands:

- Single subject/hemisphere bundle:
  ```powershell
  conda run -n base python scripts/run_full_detection_bundle.py ^
    --config configs/defaults.yaml ^
    --cifti-file <your.dtseries.nii> ^
    --hemisphere left ^
    --data-dir <data_dir> ^
    --output-dir <output_root> ^
    --bundle-root <bundle_root>
  ```
- Batch over a manifest:
  ```powershell
  conda run -n base python scripts/run_detection_batch.py ^
    --manifest subjects.csv ^
    --config configs/defaults.yaml ^
    --data-dir <data_dir> ^
    --output-dir <output_root> ^
    --bundle-root <bundle_root>
  ```

## Project Structure

- src/matphase/ (core library: preprocess, detect, analysis/storage)
- scripts/ (run_full_detection_bundle.py, run_detection_batch.py, run_full_analytic_cube.py)
- configs/defaults.yaml
- testdata/ (flat surfaces + parcellations only)
- USAGE.md, README.md
- pyproject.toml

## Requirements

Core: numpy, scipy, nibabel, pydantic, pyyaml, h5py (see versions in pyproject.toml).
Optional: cupy/taichi for GPU, matplotlib for plots.

## License

MIT License.
