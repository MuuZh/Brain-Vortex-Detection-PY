"""Data validation utilities for CIFTI/GIFTI I/O.

This module provides sanity checks for:
- Mask coverage: Verify CIFTI vertex count matches expected surface resolution
- Grid bounds: Ensure interpolation coordinate ranges cover surface extent
"""

from pathlib import Path
from typing import Optional, Tuple
import warnings

import numpy as np

from .cifti import load_cifti
from .surface import load_surface


def check_mask_coverage(
    cifti_path: str | Path,
    surface_path: str | Path,
    structure: str = "CORTEX_LEFT",
    expected_vertices: Optional[int] = None,
    tolerance: float = 0.05,
    warn_only: bool = True,
) -> dict:
    """Check if CIFTI data covers expected surface area.

    Validates that the number of vertices in the CIFTI file matches the
    expected surface resolution. Typical values for HCP 32k surfaces:
    - Left cortex: ~29,696 vertices (excluding medial wall)
    - Right cortex: ~29,716 vertices (excluding medial wall)

    Args:
        cifti_path: Path to CIFTI time series file
        surface_path: Path to GIFTI surface file
        structure: Brain structure to check (e.g., "CORTEX_LEFT")
        expected_vertices: Expected number of non-medial-wall vertices.
                           If None, uses surface total vertex count.
        tolerance: Acceptable relative deviation from expected (default 5%)
        warn_only: If True, issue warning on mismatch. If False, raise ValueError.

    Returns:
        Dictionary with validation results:
        - "cifti_vertices": Number of vertices in CIFTI file
        - "surface_vertices": Total vertices in surface mesh
        - "expected_vertices": Expected vertex count
        - "coverage_ratio": Ratio of CIFTI to expected vertices
        - "passed": Whether validation passed

    Raises:
        ValueError: If validation fails and warn_only=False
        FileNotFoundError: If files don't exist

    Example:
        >>> result = check_mask_coverage(
        ...     "data.dtseries.nii",
        ...     "L.flat.32k_fs_LR.surf.gii",
        ...     structure="CORTEX_LEFT",
        ...     expected_vertices=29696
        ... )
        >>> print(f"Coverage: {result['coverage_ratio']:.1%}")
    """
    # Convert to Path
    cifti_path = Path(cifti_path)
    surface_path = Path(surface_path)

    if not cifti_path.exists():
        raise FileNotFoundError(f"CIFTI file not found: {cifti_path}")
    if not surface_path.exists():
        raise FileNotFoundError(f"Surface file not found: {surface_path}")

    # Load CIFTI and surface
    ts = load_cifti(cifti_path)
    mesh = load_surface(surface_path)

    # Get CIFTI vertex count for structure
    if structure not in ts.metadata.brain_structures:
        raise ValueError(
            f"Structure '{structure}' not found in CIFTI file. "
            f"Available: {ts.metadata.brain_structures}"
        )

    cifti_vertex_count = ts.get_structure_data(structure).shape[0]
    surface_vertex_count = mesh.vertices.shape[0]

    # Determine expected count
    if expected_vertices is None:
        # Use surface total as expected (assumes no medial wall exclusion)
        expected_count = surface_vertex_count
    else:
        expected_count = expected_vertices

    # Calculate coverage ratio
    coverage_ratio = cifti_vertex_count / expected_count

    # Check tolerance
    deviation = abs(coverage_ratio - 1.0)
    passed = deviation <= tolerance

    # Prepare result
    result = {
        "cifti_vertices": cifti_vertex_count,
        "surface_vertices": surface_vertex_count,
        "expected_vertices": expected_count,
        "coverage_ratio": coverage_ratio,
        "passed": passed,
        "deviation": deviation,
    }

    # Issue warning or raise error if failed
    if not passed:
        message = (
            f"Mask coverage check failed for structure '{structure}':\n"
            f"  CIFTI vertices: {cifti_vertex_count}\n"
            f"  Expected vertices: {expected_count}\n"
            f"  Coverage ratio: {coverage_ratio:.2%}\n"
            f"  Deviation: {deviation:.2%} (tolerance: {tolerance:.2%})\n"
            f"  Surface total: {surface_vertex_count} vertices"
        )

        if warn_only:
            warnings.warn(message, UserWarning)
        else:
            raise ValueError(message)

    return result


def check_grid_bounds(
    surface_path: str | Path,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    margin: float = 0.0,
    warn_on_overflow: bool = True,
) -> dict:
    """Verify interpolation grid parameters cover surface extent.

    Ensures that the specified interpolation coordinate ranges are large
    enough to cover the surface geometry. Useful for preprocessing steps
    that interpolate surface data onto regular grids.

    Args:
        surface_path: Path to GIFTI surface file
        x_range: (x_min, x_max) coordinate range for interpolation grid
        y_range: (y_min, y_max) coordinate range for interpolation grid
        margin: Required margin beyond surface bounds (default 0.0)
        warn_on_overflow: If True, warn if surface extends beyond grid bounds

    Returns:
        Dictionary with validation results:
        - "surface_bounds": {"x": (min, max), "y": (min, max)}
        - "grid_bounds": {"x": x_range, "y": y_range}
        - "margins": {"x": (left, right), "y": (bottom, top)}
        - "passed": Whether grid covers surface with required margin
        - "overflow_vertices": Number of vertices outside grid bounds

    Example:
        >>> result = check_grid_bounds(
        ...     "L.flat.32k_fs_LR.surf.gii",
        ...     x_range=(-250, 250),
        ...     y_range=(-150, 200),
        ...     margin=10.0
        ... )
        >>> if result["passed"]:
        ...     print("Grid covers surface with margin")
    """
    # Convert to Path
    surface_path = Path(surface_path)

    if not surface_path.exists():
        raise FileNotFoundError(f"Surface file not found: {surface_path}")

    # Load surface
    mesh = load_surface(surface_path)
    vertices = mesh.vertices  # Shape: (N, 3)

    # Extract x, y coordinates (assume 2D projection for flat surfaces)
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]

    # Compute surface bounds
    surface_bounds = {
        "x": (float(x_coords.min()), float(x_coords.max())),
        "y": (float(y_coords.min()), float(y_coords.max())),
    }

    grid_bounds = {
        "x": x_range,
        "y": y_range,
    }

    # Calculate margins (positive = grid extends beyond surface)
    margins = {
        "x": (
            x_range[0] - surface_bounds["x"][0],  # left margin
            surface_bounds["x"][1] - x_range[1],  # right margin (negative if grid too small)
        ),
        "y": (
            y_range[0] - surface_bounds["y"][0],  # bottom margin
            surface_bounds["y"][1] - y_range[1],  # top margin (negative if grid too small)
        ),
    }

    # Correct margin sign: should be negative if grid extends beyond surface
    margins["x"] = (-margins["x"][0], margins["x"][1])
    margins["y"] = (-margins["y"][0], margins["y"][1])

    # Check if grid covers surface with required margin
    passed = (
        margins["x"][0] >= margin and margins["x"][1] >= margin and
        margins["y"][0] >= margin and margins["y"][1] >= margin
    )

    # Count vertices outside grid bounds
    overflow_mask = (
        (x_coords < x_range[0]) | (x_coords > x_range[1]) |
        (y_coords < y_range[0]) | (y_coords > y_range[1])
    )
    overflow_count = int(overflow_mask.sum())

    result = {
        "surface_bounds": surface_bounds,
        "grid_bounds": grid_bounds,
        "margins": margins,
        "passed": passed,
        "overflow_vertices": overflow_count,
    }

    # Issue warning if vertices overflow
    if overflow_count > 0 and warn_on_overflow:
        pct_overflow = 100.0 * overflow_count / len(x_coords)
        message = (
            f"Grid bounds check: {overflow_count} vertices ({pct_overflow:.2f}%) "
            f"fall outside grid bounds.\n"
            f"  Surface bounds: x={surface_bounds['x']}, y={surface_bounds['y']}\n"
            f"  Grid bounds: x={x_range}, y={y_range}\n"
            f"  Margins: left={margins['x'][0]:.1f}, right={margins['x'][1]:.1f}, "
            f"bottom={margins['y'][0]:.1f}, top={margins['y'][1]:.1f}\n"
            f"  Required margin: {margin}"
        )
        warnings.warn(message, UserWarning)

    return result


def validate_preprocessing_compatibility(
    cifti_path: str | Path,
    surface_path: str | Path,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    structure: str = "CORTEX_LEFT",
    expected_vertices: Optional[int] = None,
) -> dict:
    """Comprehensive validation for preprocessing pipeline.

    Combines mask coverage and grid bounds checks to ensure data is
    compatible with preprocessing pipeline (interpolation, filtering, etc.).

    Args:
        cifti_path: Path to CIFTI time series file
        surface_path: Path to GIFTI surface file
        x_range: Interpolation grid x-coordinate range
        y_range: Interpolation grid y-coordinate range
        structure: Brain structure to validate
        expected_vertices: Expected non-medial-wall vertex count

    Returns:
        Dictionary with combined validation results:
        - "mask_coverage": Result from check_mask_coverage()
        - "grid_bounds": Result from check_grid_bounds()
        - "passed": Whether both checks passed

    Example:
        >>> result = validate_preprocessing_compatibility(
        ...     "data.dtseries.nii",
        ...     "L.flat.32k_fs_LR.surf.gii",
        ...     x_range=(-250, 250),
        ...     y_range=(-150, 200),
        ...     structure="CORTEX_LEFT",
        ...     expected_vertices=29696
        ... )
        >>> if result["passed"]:
        ...     print("Data is compatible with preprocessing pipeline")
    """
    # Run mask coverage check
    mask_result = check_mask_coverage(
        cifti_path=cifti_path,
        surface_path=surface_path,
        structure=structure,
        expected_vertices=expected_vertices,
        warn_only=True,
    )

    # Run grid bounds check
    grid_result = check_grid_bounds(
        surface_path=surface_path,
        x_range=x_range,
        y_range=y_range,
        warn_on_overflow=True,
    )

    # Combine results
    result = {
        "mask_coverage": mask_result,
        "grid_bounds": grid_result,
        "passed": mask_result["passed"] and grid_result["passed"],
    }

    return result
