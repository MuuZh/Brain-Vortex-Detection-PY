# MatPhase Detection Bundle Usage

## What This Bundle Contains
- `src/matphase/`: core preprocessing, detection, and storage code (includes `analysis/storage.py` needed for bundle writing).
- `scripts/`: runnable entry points:
  - `run_full_detection_bundle.py` (single subject/hemisphere pipeline)
  - `run_detection_batch.py` (batch over a manifest)
  - `run_full_analytic_cube.py` (optional: saves analytic phase cube only)
- `configs/defaults.yaml`: baseline configuration.
- `pyproject.toml`: dependency list.
- `testdata/`: lightweight geometry + parcellation assets (`L.flat.32k_fs_LR.surf.gii`, `R.flat.32k_fs_LR.surf.gii`, `left_brain_parcellation.npy`, `right_brain_parcellation.npy`, `parcellation_template.mat`).

## Prerequisites
- Python 3.11+ (recommend `conda run -n base`).
- Install package editable:
  ```powershell
  conda run -n base pip install -e .
  ```
- Ensure CUDA/TAICHI only if you set `compute.use_gpu=true` in config (not required by default).

## Required Inputs
1. **CIFTI dtseries** (`*.dtseries.nii`)
   - Dense time series; vertex dimension should match HCP fs_LR 32k surfaces (~64,984 cortical vertices total, 32,492 per hemisphere) and time dimension = frames.
   - File is referenced via `paths.cifti_file`; if relative, it is joined to `paths.data_dir`.
2. **Flat surfaces** (`L.flat.32k_fs_LR.surf.gii`, `R.flat.32k_fs_LR.surf.gii`)
   - Vertex count 32,492 per hemisphere; coordinates in mm; used for interpolation and plotting masks.
3. **Parcellation grids** (`left_brain_parcellation.npy`, `right_brain_parcellation.npy`)
   - Shape `(176, 251)`; NaN outside cortex. Grid produced from flat-surface coordinates with `downsample_rate=2` over the default x/y ranges in `defaults.yaml`.
4. **Optional template** (`parcellation_template.mat`)
   - MATLAB reference template (not needed for core pipeline).

## Key Output Artifacts (per hemisphere run)
- `<bundle_root>/<bundle_name>/metadata.json`
  - Includes schema version, grid height/width (default 176x251), frame_count, detection parameters, statistics, and extra metadata (hemisphere, thresholds).
- `<bundle_root>/<bundle_name>/patterns.parquet`
  - One row per detected spiral pattern (id, duration, mean size/power, bounding box, compatibility stats).
- `<bundle_root>/<bundle_name>/frame_index.parquet`
  - One row per pattern-frame with centroids, instantaneous size/power, optional expansion radius and compatibility flags. `coord_start`/`coord_end` index into `coords.feather`.
- `<bundle_root>/<bundle_name>/coords.feather`
  - Two-column (`y`, `x`) int16 coordinate pool; rows are stacked slices of voxel indices for each frame.
- `<bundle_root>/<bundle_name>/phase_cube.npy` (if `output.save_phase_cube=true`)
  - Float32 phase array shaped `(grid_height, grid_width, frame_count)`.
- `run.log` (if logging to file enabled).

### Interpreting detection outputs
- `metadata.json`
  - `grid_height`, `grid_width`, `frame_count`: dimensions of the downsampled phase field (default 176 x 251 x T).
  - `detection_statistics`: aggregate metrics (kept ratios, average radius, etc.).
  - `extra_metadata`: hemisphere, rotation_mode, curl_threshold, surrogate thresholds if used.

- `patterns.parquet` (one row per spiral pattern)
  - Columns: `pattern_id` (int), `rotation_direction` (`ccw`/`cw`), `curl_sign` (+/-), `start_frame`, `end_frame`, `duration`, `mean_size`, `mean_power`, `mean_peak_amp`, bounding box (`bbox_x0..bbox_t1`), `fraction_frames_passing`, `mean_compatibility_ratio`, `all_frames_filtered`.
  - Use to filter patterns globally (e.g., keep `duration >= 3` and `fraction_frames_passing > 0.5`).

- `frame_index.parquet` (one row per pattern per frame)
  - Keys: `pattern_id`, `frame_idx` (0-based within pattern), `abs_time` (frame index in cube), `coord_start`, `coord_end`, `voxel_count`.
  - Geometry: `centroid_x/y`, `weighted_centroid_x/y`, `instantaneous_size`, `instantaneous_power`, `instantaneous_peak_amp`, `instantaneous_width`, optional `expansion_radius`, `compatibility_ratio`, `compatibility_pass`.
  - `coord_start`/`coord_end` slice into `coords.feather` to reconstruct the mask for that frame.

- `coords.feather`
  - Columns `y`, `x` (int16). Concatenated coordinate pool for all frames of all patterns. For a `frame_index` row, slice `coords[coord_start:coord_end]` to get pixel locations belonging to that pattern at `abs_time`.

- `phase_cube.npy` (if saved)
  - Shape `(grid_height, grid_width, frame_count)`, dtype float32, values are wrapped phase in radians. Aligns with `abs_time` in `frame_index`.

#### Linking files together
1. Load `patterns.parquet` to choose `pattern_id`s.
2. Subset `frame_index.parquet` by `pattern_id`.
3. For each row, slice `coords = coords_array[coord_start:coord_end]` to get `(y, x)` indices.
4. Create a binary mask `(grid_height, grid_width)` and set those indices to 1; `abs_time` gives the frame in `phase_cube.npy`.

#### Minimal Python snippet (offline)
```python
import json, pandas as pd, numpy as np
from pathlib import Path

bundle = Path("path/to/bundle")
patterns = pd.read_parquet(bundle/"patterns.parquet")
frame_index = pd.read_parquet(bundle/"frame_index.parquet")
coords = pd.read_feather(bundle/"coords.feather")[['y','x']].to_numpy()
meta = json.loads((bundle/"metadata.json").read_text())
H, W = meta["grid_height"], meta["grid_width"]

pid = patterns.query("duration>=3").pattern_id.iloc[0]
row = frame_index[frame_index.pattern_id==pid].iloc[0]
mask = np.zeros((H, W), dtype=bool)
mask_coords = coords[row.coord_start:row.coord_end]
mask[mask_coords[:,0], mask_coords[:,1]] = True
```

## Running a Single Subject
```powershell
conda run -n base python scripts/run_full_detection_bundle.py \
  --config configs/defaults.yaml \
  --hemisphere left \
  --cifti-file DMT_DMT_S01_Atlas_s0.dtseries.nii \
  --data-dir <path_to_data_dir> \
  --output-dir <path_to_output_root> \
  --bundle-root <path_to_bundle_root> \
  --bundle-suffix subj01_left \
  --show-progress
```

Notes:
- `--hemisphere both` processes left then right and writes two bundle folders.
- Override thresholds/rotation if needed (e.g., `--curl-threshold 0.85`, `--rotation-mode ccw`).
- Sampling rate comes from CIFTI metadata; override with `--sampling-rate` or `--tr`.

## Batch Runs
1. Prepare a manifest CSV/TSV/JSON with at least columns: `cifti_file`, `hemisphere` (values: `left`, `right`, or `both` not allowed; supply two rows instead).

Sample CSV:
```csv
cifti_file,hemisphere,bundle_root,bundle_suffix
E:/resarch_data/fMRI/fMRI_DMT/dtseries/DMT_DMT_dtseries/DMT_DMT_S01_Atlas_s0.dtseries.nii,left,detect_results,DMTDMT01L
E:/resarch_data/fMRI/fMRI_DMT/dtseries/DMT_DMT_dtseries/DMT_DMT_S01_Atlas_s0.dtseries.nii,right,detect_results,DMTDMT01R
E:/resarch_data/fMRI/fMRI_DMT/dtseries/DMT_DMT_dtseries/DMT_DMT_S02_Atlas_s0.dtseries.nii,left,detect_results,DMTDMT02L
E:/resarch_data/fMRI/fMRI_DMT/dtseries/DMT_DMT_dtseries/DMT_DMT_S02_Atlas_s0.dtseries.nii,right,detect_results,DMTDMT02R
```

2. Run:
```powershell
conda run -n base python scripts/run_detection_batch.py \
  --manifest subjects.csv \
  --config configs/defaults.yaml \
  --bundle-root <path_to_bundle_root> \
  --output-dir <path_to_output_root> \
  --data-dir <path_to_data_dir> \
  --max-workers 2 \
  --show-progress
```

Optional per-row fields: `bundle_root`, `bundle_suffix`, `log_file`.

## Data/Dimension Reference
- **Interpolation grid**: defaults from `configs/defaults.yaml` give x/y ranges (-250..250 mm left, -270..230 mm right; y ranges similar) with `downsample_rate=2`, producing a 176 (y) x 251 (x) grid per hemisphere.
- **CIFTI frames**: frame count in outputs equals the time dimension of the input dtseries after any temporal filtering (no trimming by default).
- **Coordinate system**: grid indices `(y, x)` are zero-based array positions into the downsampled phase field; `coords.feather` stores these indices, not physical mm coordinates.

## Quick Checklist
- Place CIFTI + surfaces + parcellations under one `data_dir`, or pass absolute paths.
- Adjust `output_dir` and `bundle_root` to writable locations.
- Keep `output.save_phase_cube=true` if you need phase cubes for diagnostics; set false to save space.
- For GPU runs, toggle `compute.use_gpu` and ensure CuPy/Taichi installed. Note: Not yet fully developed.

## Troubleshooting
- **Missing geometry/parcellation**: verify `paths.surface_left/right` and parcellation filenames are reachable from `data_dir`.
- **No spirals detected**: try lowering `--curl-threshold` or disabling surrogate thresholds (`--use-surrogate-thresholds` off via config toggle).
- **Shape errors**: confirm your CIFTI uses fs_LR 32k vertices; other meshes require matching surface/parcellation inputs.