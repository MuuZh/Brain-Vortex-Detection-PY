"""
Parcellation template loading and mask generation utilities.

Provides functions to:
1. Load parcellation templates from .npy files
2. Convert parcellation arrays to binary masks
3. Handle downsampling and NaN regions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_parcellation(path: Path | str) -> np.ndarray:
    """
    Load parcellation template from .npy file.

    Args:
        path: Path to parcellation .npy file

    Returns:
        parcellation: (176, 251) array with ROI labels (may contain NaN)

    Raises:
        ValueError: If file not found, invalid format, or incorrect shape
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Parcellation file not found: {path}")

    # Load array
    parcellation = np.load(path, allow_pickle=False)

    # Validate shape
    if parcellation.ndim != 2:
        raise ValueError(
            f"Expected 2D parcellation array, got shape {parcellation.shape}"
        )

    # Validate data type (should be numeric) - check before shape to get better error messages
    if not np.issubdtype(parcellation.dtype, np.number):
        raise ValueError(
            f"Expected numeric parcellation array, got dtype {parcellation.dtype}"
        )

    # Validate expected shape (176, 251) for downsample_rate=2
    if parcellation.shape != (176, 251):
        raise ValueError(
            f"Expected parcellation shape (176, 251), got {parcellation.shape}"
        )

    return parcellation


def parcellation_to_mask(parcellation: np.ndarray) -> np.ndarray:
    """
    Convert parcellation array to binary mask.

    Args:
        parcellation: (176, 251) array with ROI labels

    Returns:
        mask: (176, 251) binary mask (1.0=valid, NaN=invalid)

    Notes:
        - Valid regions are where parcellation is not NaN (~np.isnan -> 1)
        - Invalid regions are set to NaN (MATLAB convention)
        - Output dtype is float32
        - Input parcellation should already be downsampled to grid resolution
    """
    # Create mask: valid where not NaN
    mask = np.ones_like(parcellation, dtype=np.float32)
    mask[np.isnan(parcellation)] = np.nan

    return mask


def validate_parcellation_shape(
    parcellation: np.ndarray,
    expected_shape: tuple[int, int],
    name: str = "parcellation"
) -> None:
    """
    Validate that parcellation has expected shape.

    Args:
        parcellation: parcellation array to validate
        expected_shape: (n_y, n_x) expected shape
        name: name for error message

    Raises:
        ValueError: If shape mismatch
    """
    if parcellation.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: expected {expected_shape}, "
            f"got {parcellation.shape}"
        )
