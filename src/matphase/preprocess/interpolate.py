"""
Spatial interpolation and cortical mask generation for fMRI preprocessing.

This module provides functions to:
1. Generate cortical boundary masks using alpha shapes or convex hulls
2. Interpolate irregular surface vertex data to regular 2D grids
3. Apply masks to interpolated data
4. Handle coordinate transformation to positive axis system

Corresponds to MATLAB: load_fMRI.m (lines 26-44), spaceFreq_fMRI.m

IMPORTANT: Default behavior matches MATLAB: interpolation can be performed in
physical coordinates, then mapped to the positive-axis grid for parcellation
masks. Set `coordinate_system` to control whether vertices are shifted before
interpolation.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from matphase.utils.logging import get_logger
logger = get_logger(__name__)


try:
    from alphashape import alphashape as compute_alphashape
    ALPHASHAPE_AVAILABLE = True
except ImportError:
    ALPHASHAPE_AVAILABLE = False


def shift_coordinates_to_positive(
    xy: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float]
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Shift coordinates to positive axis (0-based).

    Matches MATLAB's poly2mask behavior: b(:,1)-min(xCord)+1
    The +1 offset is critical for proper grid alignment with parcellation masks.

    Args:
        xy: (n_points, 2) array of (x, y) coordinates
        x_range: (min, max) for x coordinates in physical space
        y_range: (min, max) for y coordinates in physical space

    Returns:
        xy_shifted: (n_points, 2) coordinates shifted to start from 1 (MATLAB convention)
        shifts: (x_shift, y_shift) - values subtracted from coordinates
    """
    # MATLAB: b(:,1)-min(xCord)+1 → shift by -min + 1
    x_shift = -x_range[0] + 1
    y_shift = -y_range[0] + 1

    xy_shifted = xy.copy()
    xy_shifted[:, 0] += x_shift
    xy_shifted[:, 1] += y_shift

    return xy_shifted, (x_shift, y_shift)


def generate_coordinate_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    downsample_rate: int = 2,
    return_physical: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate regular 2D coordinate grid for interpolation.

    By default, returns 0-based positive axis coordinates to align with
    parcellation masks. Set return_physical=True to get physical coordinates.

    Args:
        x_range: (min, max) for x coordinates in physical space
        y_range: (min, max) for y coordinates in physical space
        downsample_rate: spacing between grid points
        return_physical: if True, return physical coordinates; if False, return 0-based

    Returns:
        x_coords: (n_x,) array of x coordinates
        y_coords: (n_y,) array of y coordinates
        X: (n_y, n_x) meshgrid of x coordinates
        Y: (n_y, n_x) meshgrid of y coordinates
    """
    # Calculate grid dimensions
    x_length = x_range[1] - x_range[0]
    y_length = y_range[1] - y_range[0]

    if return_physical:
        # Physical coordinates (negative to positive)
        x_coords = np.arange(x_range[0], x_range[1] + 1, downsample_rate)
        y_coords = np.arange(y_range[0], y_range[1] + 1, downsample_rate)
    else:
        # 0-based positive coordinates (matches parcellation indexing)
        x_coords = np.arange(0, x_length + 1, downsample_rate)
        y_coords = np.arange(0, y_length + 1, downsample_rate)

    # MATLAB meshgrid uses (x, y) ordering; numpy meshgrid defaults to 'xy' indexing
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')

    return x_coords, y_coords, X, Y


def generate_cortical_mask(
    positions: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    downsample_rate: int = 2,
    alpha: float = 4.0,
    method: Literal["alphashape", "convexhull"] = "alphashape",
    parcellation_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Generate binary mask defining valid cortical region.

    If parcellation_mask is provided, it is used directly (after validation).
    Otherwise, uses alpha shape or convex hull to compute boundary.

    Args:
        positions: (n_vertices, 2 or 3) - vertex coordinates (uses x, y only)
        x_range: (min, max) for x coordinates in physical space
        y_range: (min, max) for y coordinates in physical space
        downsample_rate: grid spacing
        alpha: alpha parameter for alpha shape (larger = simpler boundary)
        method: "alphashape" or "convexhull"
        parcellation_mask: (n_y, n_x) pre-computed mask (1.0=valid, NaN=invalid)

    Returns:
        mask: (n_y, n_x) - binary mask (1.0=valid, NaN=invalid)

    Note:
        If parcellation_mask is provided, positions/alpha/method are ignored.
    """
    # If parcellation mask provided, use it directly
    if parcellation_mask is not None:
        # Validate dtype and ensure NaN convention
        mask = parcellation_mask.astype(np.float32)
        if not np.any(np.isnan(mask)):
            warnings.warn(
                "Parcellation mask contains no NaN values - "
                "all pixels marked as valid"
            )
        return mask

    # Otherwise, compute mask from positions
    # Extract x, y coordinates
    if positions.shape[1] == 3:
        xy = positions[:, :2]  # Use only x, y
    else:
        xy = positions

    # Shift coordinates to positive axis
    xy_shifted, (x_shift, y_shift) = shift_coordinates_to_positive(
        xy, x_range, y_range)

    # Generate coordinate grid (0-based)
    x_coords, y_coords, X, Y = generate_coordinate_grid(
        x_range, y_range, downsample_rate, return_physical=False
    )

    # Compute boundary
    if method == "alphashape" and ALPHASHAPE_AVAILABLE:
        # Use alpha shape for concave hull
        try:
            alpha_shape = compute_alphashape(xy_shifted, alpha=alpha)

            # Get boundary polygon
            if hasattr(alpha_shape, 'exterior'):
                # Shapely Polygon
                boundary_coords = np.array(alpha_shape.exterior.coords)
            elif hasattr(alpha_shape, 'boundary'):
                # Alpha shape object with boundary
                boundary_coords = np.array(alpha_shape.boundary.coords)
            else:
                warnings.warn(
                    "Could not extract boundary from alpha shape, falling back to convex hull")
                method = "convexhull"
        except Exception as e:
            warnings.warn(
                f"Alpha shape failed: {e}, falling back to convex hull")
            method = "convexhull"

    if method == "convexhull" or not ALPHASHAPE_AVAILABLE:
        # Use convex hull as fallback
        if not ALPHASHAPE_AVAILABLE and method == "alphashape":
            warnings.warn(
                "alphashape package not available, using convex hull")

        hull = ConvexHull(xy_shifted)
        boundary_indices = hull.vertices
        boundary_coords = xy_shifted[boundary_indices]
        # Close the polygon
        boundary_coords = np.vstack([boundary_coords, boundary_coords[0]])

    # Rasterize boundary to mask
    mask = rasterize_polygon_mask(boundary_coords, X, Y)

    # Convert 0 to NaN (MATLAB convention)
    mask = mask.astype(np.float32)
    mask[mask == 0] = np.nan

    return mask


def rasterize_polygon_mask(
    boundary_coords: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    Rasterize polygon boundary to binary mask.

    Args:
        boundary_coords: (n_boundary, 2) - polygon vertices (x, y)
        X: (n_y, n_x) - meshgrid of x coordinates
        Y: (n_y, n_x) - meshgrid of y coordinates

    Returns:
        mask: (n_y, n_x) - binary mask (1=inside, 0=outside)
    """
    from matplotlib.path import Path

    # Flatten grid
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Create path and test containment
    path = Path(boundary_coords)
    mask_flat = path.contains_points(points)

    # Reshape to grid
    mask = mask_flat.reshape(X.shape).astype(np.float32)

    return mask


def interpolate_to_grid(
    signal: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray | None,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "cubic",
    n_jobs: int = 1,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    coordinate_system: Literal["positive", "physical"] = "positive",
    show_progress: bool = False,
) -> np.ndarray:
    """
    Interpolate surface data to regular 2D grid.

    Automatically filters out vertices with NaN values (e.g., medial wall)
    before interpolation. Coordinates are shifted to positive axis system
    if x_range and y_range are provided.

    Args:
        signal: (n_vertices, n_timepoints) - time series data
        positions: (n_vertices, 2 or 3) - vertex coordinates (uses x, y)
        faces: (n_faces, 3) triangle indices; required for triangulation methods
        x_coords: (n_x,) - grid x coordinates (0-based for positive mode or physical)
        y_coords: (n_y,) - grid y coordinates (0-based for positive mode or physical)
        mask: (n_y, n_x) - cortical mask (optional, 1.0=valid, NaN=invalid)
        method: 'linear', 'cubic', 'nearest', or 'tri_linear'
        n_jobs: number of parallel jobs for timepoint loop (1=serial)
        x_range: (min, max) physical x coordinates (for coordinate shifting)
        y_range: (min, max) physical y coordinates (for coordinate shifting)
        coordinate_system: "positive" to shift vertices to 0-based coordinates
            (MATLAB poly2mask parity), or "physical" to interpolate directly in
            the original coordinate system.
        show_progress: display tqdm progress when n_jobs==1

    Returns:
        interpolated: (n_y, n_x, n_timepoints) - gridded data

    Note:
        - Vertices with NaN coordinates or NaN signal values are automatically
          excluded from interpolation (e.g., medial wall vertices)
        - If x_range and y_range are provided, positions are shifted to
          positive axis to align with 0-based grid coordinates
        - Grid points outside the convex hull of valid vertices will be NaN
    """
    n_vertices, n_timepoints = signal.shape

    # Extract x, y coordinates
    if positions.shape[1] == 3:
        xy = positions[:, :2]
    else:
        xy = positions

    # Filter out vertices with NaN values (e.g., medial wall)
    # Check for NaN in either coordinates or signal (across all timepoints)
    valid_coords = ~(np.isnan(xy[:, 0]) | np.isnan(xy[:, 1]))
    valid_signal = ~np.any(np.isnan(signal), axis=1)
    valid_vertices = valid_coords & valid_signal

    n_valid = valid_vertices.sum()
    n_invalid = n_vertices - n_valid

    if method == "tri_linear":
        # For triangulation, keep full vertex set (faces index into original ordering)
        xy_valid = xy
        signal_valid = signal
        if coordinate_system == "positive" and x_range is not None and y_range is not None:
            xy_shifted, (x_shift, y_shift) = shift_coordinates_to_positive(xy, x_range, y_range)
        else:
            xy_shifted = xy
            if coordinate_system == "positive":
                warnings.warn(
                    "No coordinate ranges provided - assuming positions are already "
                    "in 0-based coordinate system"
                )
    else:
        if n_invalid > 0:
            # Filter to valid vertices only
            xy_valid = xy[valid_vertices]
            signal_valid = signal[valid_vertices, :]
        else:
            xy_valid = xy
            signal_valid = signal

        # Shift coordinates to positive axis if ranges provided
        if coordinate_system == "positive" and x_range is not None and y_range is not None:
            xy_shifted, (x_shift, y_shift) = shift_coordinates_to_positive(
                xy_valid, x_range, y_range)
        else:
            xy_shifted = xy_valid
            if coordinate_system == "positive":
                warnings.warn(
                    "No coordinate ranges provided - assuming positions are already "
                    "in 0-based coordinate system"
                )

    # Generate meshgrid
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
    n_y, n_x = X.shape

    # Validate mask shape if provided
    if mask is not None:
        if mask.shape != (n_y, n_x):
            raise ValueError(
                f"Mask shape {mask.shape} does not match grid shape ({n_y}, {n_x}). "
                f"Expected mask to have same dimensions as coordinate grid."
            )

    # Prepare output array
    interpolated = np.zeros((n_y, n_x, n_timepoints), dtype=np.float32)
    time_iter = range(n_timepoints)
    if n_jobs == 1 and show_progress:
        from tqdm import tqdm

        time_iter = tqdm(time_iter, desc="Interpolating", unit="frame", total=n_timepoints)

    # Interpolation backends
    if method == "tri_linear":
        # Triangulation-based linear interpolation (uses surface faces)
        if faces is None:
            raise ValueError(
                "faces must be provided for tri_linear interpolation")
        import matplotlib.tri as mtri
        logger.info("Using triangulation-based linear interpolation")

        triang = mtri.Triangulation(
            xy_shifted[:, 0], xy_shifted[:, 1], faces)

        def interpolate_timepoint_tri(t_idx: int) -> np.ndarray:
            z = signal[:, t_idx]
            bad_vert = np.isnan(z) | np.isnan(xy_shifted).any(axis=1)
            if bad_vert.any():
                bad_tri = bad_vert[triang.triangles].any(axis=1)
                tri_masked = mtri.Triangulation(
                    triang.x, triang.y, triangles=triang.triangles, mask=bad_tri
                )
            else:
                tri_masked = triang
            interpolator = mtri.LinearTriInterpolator(tri_masked, z)
            return interpolator(X, Y)

        if n_jobs == 1:
            for t in time_iter:
                interpolated[:, :, t] = interpolate_timepoint_tri(t)
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(interpolate_timepoint_tri)(t) for t in range(n_timepoints)
            )
            for t, result in enumerate(results):
                interpolated[:, :, t] = result
    else:
        # griddata backend
        logger.info("Using griddata interpolation with method '%s'", method)
        if n_jobs == 1:
            for t in time_iter:
                interpolated[:, :, t] = griddata(
                    xy_shifted, signal_valid[:, t], (X, Y), method=method, fill_value=np.nan
                )
        else:
            from joblib import Parallel, delayed

            def interpolate_timepoint(t):
                return griddata(
                    xy_shifted, signal_valid[:,
                                             t], (X, Y), method=method, fill_value=np.nan
                )

            results = Parallel(n_jobs=n_jobs)(
                delayed(interpolate_timepoint)(t) for t in range(n_timepoints)
            )

            for t, result in enumerate(results):
                interpolated[:, :, t] = result

    # Apply mask if provided
    if mask is not None:
        # Broadcast mask across time dimension
        interpolated = interpolated * mask[:, :, np.newaxis]

    return interpolated


def interpolate_to_grid_batch(
    signal: np.ndarray,
    positions: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    faces: np.ndarray | None = None,
    downsample_rate: int = 2,
    alpha: float = 4.0,
    method: str = "cubic",
    n_jobs: int = 1,
    return_mask: bool = False,
    parcellation_mask: np.ndarray | None = None,
    coordinate_system: Literal["positive", "physical"] = "physical",
    drop_last_nan_row: bool | None = None,
    show_progress: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Complete interpolation pipeline: generate/load mask + interpolate data.

    Convenience function combining generate_cortical_mask() and interpolate_to_grid().
    If parcellation_mask is provided, it is used instead of computing boundary.

    Args:
        signal: (n_vertices, n_timepoints) - time series data
        positions: (n_vertices, 2 or 3) - vertex coordinates
        faces: (n_faces, 3) surface triangles; required when method='tri_linear'
        x_range: (min, max) for x coordinates in physical space
        y_range: (min, max) for y coordinates in physical space
        downsample_rate: grid spacing
        alpha: alpha shape parameter
        method: interpolation method ('linear', 'cubic', 'nearest', 'tri_linear')
        n_jobs: number of parallel jobs
        return_mask: if True, return (interpolated, mask) tuple
        parcellation_mask: (n_y, n_x) pre-computed mask from parcellation
        coordinate_system: "physical" to interpolate in original coordinates
            (MATLAB parity), or "positive" to shift to 0-based before interpolation.
        drop_last_nan_row: if True, drop the final row after masking (mirrors MATLAB
            parcellation templates with padded NaN row). Defaults to True when a
            parcellation mask is provided, otherwise False.
        show_progress: display tqdm progress when n_jobs==1

    Returns:
        interpolated: (n_y, n_x, n_timepoints) - gridded data
        mask: (n_y, n_x) - cortical mask (if return_mask=True)
    """
    # Generate coordinate grid (physical or positive depending on mode)
    return_physical = coordinate_system == "physical"
    x_coords, y_coords, _, _ = generate_coordinate_grid(
        x_range, y_range, downsample_rate, return_physical=return_physical
    )

    # Generate or use provided cortical mask
    mask = generate_cortical_mask(
        positions, x_range, y_range, downsample_rate, alpha,
        parcellation_mask=parcellation_mask
    )

    # Interpolate data
    interpolated = interpolate_to_grid(
        signal,
        positions,
        faces,
        x_coords,
        y_coords,
        mask,
        method,
        n_jobs,
        x_range=x_range,
        y_range=y_range,
        coordinate_system=coordinate_system,
        show_progress=show_progress,
    )

    # Mirror MATLAB: templates include a padded NaN row; default to dropping it
    if drop_last_nan_row is None:
        drop_last_nan_row = parcellation_mask is not None
    if drop_last_nan_row:
        interpolated = interpolated[:-1, :, :]
        mask = mask[:-1, :]

    if return_mask:
        return interpolated, mask
    else:
        return interpolated


def get_nan_statistics(data: np.ndarray, name: str = "data") -> dict[str, float]:
    """
    Compute NaN statistics for validation.

    Args:
        data: array to analyze
        name: label for reporting

    Returns:
        dict with keys: 'total_elements', 'nan_count', 'nan_fraction', 'valid_count'
    """
    total = data.size
    nan_count = np.isnan(data).sum()
    nan_fraction = nan_count / total if total > 0 else 0.0
    valid_count = total - nan_count

    return {
        'name': name,
        'total_elements': total,
        'nan_count': int(nan_count),
        'nan_fraction': float(nan_fraction),
        'valid_count': int(valid_count)
    }
