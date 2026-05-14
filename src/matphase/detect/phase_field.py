"""
Phase field computation for spiral wave detection.

This module provides functions to compute spatial gradients and curl
(vorticity) from phase fields derived from fMRI data. The implementation
follows MATLAB reference code (spiral_detection_surfilt.m) conventions.

Key concepts:
- Phase gradient: Spatial rate of change of phase (2D vector field)
- Curl: Vorticity measure for detecting rotational patterns (spirals)
- Normalization: Unit vector fields for direction analysis

Author: matphase refactor
Date: 2025-11-09
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np
from tqdm import tqdm
import warnings


@dataclass
class PhaseFieldResult:
    """
    Result container for phase field computations.

    Attributes
    ----------
    gradient_x : np.ndarray
        Phase gradient in x-direction (columns), shape (rows, cols, timepoints)
    gradient_y : np.ndarray
        Phase gradient in y-direction (rows), shape (rows, cols, timepoints)
    magnitude : np.ndarray
        Gradient magnitude, shape (rows, cols, timepoints)
    normalized_x : np.ndarray
        Normalized gradient in x-direction (unit vectors)
    normalized_y : np.ndarray
        Normalized gradient in y-direction (unit vectors)
    curl : Optional[np.ndarray]
        Curl (vorticity) of normalized gradient field, shape (rows, cols, timepoints)
    n_nan : int
        Number of NaN values in input phase field
    magnitude_stats : dict
        Statistics of gradient magnitude (mean, std, min, max)
    method : str
        Computation method used ('central_difference')
    """
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    magnitude: np.ndarray
    normalized_x: np.ndarray
    normalized_y: np.ndarray
    curl: Optional[np.ndarray] = None
    n_nan: int = 0
    magnitude_stats: dict = None
    method: str = "central_difference"

    def __post_init__(self):
        """Compute magnitude statistics if not provided."""
        if self.magnitude_stats is None:
            self.magnitude_stats = {
                "mean": float(np.nanmean(self.magnitude)),
                "std": float(np.nanstd(self.magnitude)),
                "min": float(np.nanmin(self.magnitude)),
                "max": float(np.nanmax(self.magnitude)),
            }


def angle_subtract(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Subtract angles with proper wrapping to [-pi, pi].

    Equivalent to MATLAB anglesubtract.m. Handles phase periodicity
    correctly for angular data in radians.

    Parameters
    ----------
    x : np.ndarray
        First angle array (radians)
    y : np.ndarray
        Second angle array (radians)

    Returns
    -------
    np.ndarray
        x - y wrapped to [-pi, pi]

    Notes
    -----
    Uses modulo method: mod(x - y + pi, 2*pi) - pi
    This ensures the result is in the principal range.

    Examples
    --------
    >>> angle_subtract(np.pi, -np.pi)
    0.0
    >>> angle_subtract(0.1, -0.1)
    0.2
    >>> angle_subtract(3.0, -3.0)  # wraps around
    -0.283...
    """
    # Modulo method (matches MATLAB anglesubtract.m METHOD 1)
    diff = np.mod(x - y + np.pi, 2 * np.pi) - np.pi
    return diff


def compute_phase_gradient(
    phase: np.ndarray,
    spacing: float = 1.0,
    method: Literal["central_difference"] = "central_difference",
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial gradient of phase field using angular difference.

    Implements central difference scheme matching MATLAB code:
    - vPhaseX(iX, 2:end-1, iTime) = anglesubtract(phase(iX, 3:end, iTime), phase(iX, 1:end-2, iTime)) / 2
    - vPhaseY(2:end-1, iY, iTime) = anglesubtract(phase(3:end, iY, iTime), phase(1:end-2, iY, iTime)) / 2

    Parameters
    ----------
    phase : np.ndarray
        Phase field array, shape (rows, cols) or (rows, cols, timepoints)
        Phase values in radians, typically in range [-pi, pi]
    spacing : float, optional
        Grid spacing for gradient calculation (default: 1.0)
        In MATLAB code, this is implicitly 1 (index-based)
    method : str, optional
        Gradient computation method (default: 'central_difference')
        Currently only supports central differences
    show_progress : bool, optional
        Show progress bar for batch processing (default: False)

    Returns
    -------
    gradient_x : np.ndarray
        Phase gradient in x-direction (columns), same shape as input
        Edges (first/last column) are NaN
    gradient_y : np.ndarray
        Phase gradient in y-direction (rows), same shape as input
        Edges (first/last row) are NaN

    Notes
    -----
    - Uses angle_subtract() for proper phase wrapping
    - Central differences: f'(i) ≈ (f(i+1) - f(i-1)) / 2
    - Edge pixels set to NaN (not computed by central difference)
    - Input NaNs are propagated to output

    References
    ----------
    MATLAB: spiral_detection_surfilt.m, lines 58-65
    """
    # Validate input
    if phase.ndim not in [2, 3]:
        raise ValueError(f"Phase must be 2D or 3D array, got shape {phase.shape}")

    # Ensure 3D for uniform processing
    if phase.ndim == 2:
        phase = phase[..., np.newaxis]

    rows, cols, n_timepoints = phase.shape

    # Initialize gradient arrays (filled with NaN)
    gradient_x = np.full_like(phase, np.nan, dtype=np.float64)
    gradient_y = np.full_like(phase, np.nan, dtype=np.float64)

    # Setup progress bar
    iterator = range(n_timepoints)
    if show_progress and n_timepoints > 10:
        iterator = tqdm(iterator, desc="Computing phase gradients", unit="frame")

    # Compute gradients for each timepoint
    for t in iterator:
        phase_t = phase[:, :, t]

        # X-gradient (along columns): central difference
        # Matches: vPhaseX(iX, 2:end-1, iTime) = anglesubtract(...) / 2
        for i in range(rows):
            if cols >= 3:  # Need at least 3 columns for central difference
                # Central difference: (phase[i, j+1] - phase[i, j-1]) / 2
                forward = phase_t[i, 2:]       # phase(iX, 3:end)
                backward = phase_t[i, :-2]     # phase(iX, 1:end-2)
                gradient_x[i, 1:-1, t] = angle_subtract(forward, backward) / (2.0 * spacing)

        # Y-gradient (along rows): central difference
        # Matches: vPhaseY(2:end-1, iY, iTime) = anglesubtract(...) / 2
        for j in range(cols):
            if rows >= 3:  # Need at least 3 rows for central difference
                # Central difference: (phase[i+1, j] - phase[i-1, j]) / 2
                forward = phase_t[2:, j]       # phase(3:end, iY)
                backward = phase_t[:-2, j]     # phase(1:end-2, iY)
                gradient_y[1:-1, j, t] = angle_subtract(forward, backward) / (2.0 * spacing)

    return gradient_x, gradient_y


def normalize_vector_field(
    vx: np.ndarray,
    vy: np.ndarray,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize 2D vector field to unit vectors.

    Matches MATLAB code:
    - Vx_norm = -vPhaseX / sqrt(vPhaseX^2 + vPhaseY^2)
    - Vy_norm = -vPhaseY / sqrt(vPhaseX^2 + vPhaseY^2)

    Parameters
    ----------
    vx : np.ndarray
        X-component of vector field
    vy : np.ndarray
        Y-component of vector field
    inplace : bool, optional
        Modify input arrays in-place (default: False)

    Returns
    -------
    vx_norm : np.ndarray
        Normalized x-component (unit vectors)
    vy_norm : np.ndarray
        Normalized y-component (unit vectors)
    magnitude : np.ndarray
        Original magnitude before normalization

    Notes
    -----
    - MATLAB code negates gradients: this matches the convention
      that phase gradients point "downhill" in phase
    - Zero-magnitude vectors result in NaN (division by zero)
    - Input NaNs propagate to output

    References
    ----------
    MATLAB: spiral_detection_surfilt.m, lines 66-67
    """
    # Compute magnitude
    magnitude = np.sqrt(vx**2 + vy**2)

    # Avoid in-place modification of input if requested
    if not inplace:
        vx = vx.copy()
        vy = vy.copy()

    # Normalize (with NaN-safe division)
    # Note: MATLAB code negates gradients (matches reference convention)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')

        vx_norm = -vx / magnitude
        vy_norm = -vy / magnitude

    return vx_norm, vy_norm, magnitude


def compute_curl_2d(
    vx: np.ndarray,
    vy: np.ndarray,
    spacing: float = 1.0,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Compute curl (vorticity) of 2D vector field.

    For a 2D vector field (vx, vy), the curl is a scalar:
    curl_z = ∂vy/∂x - ∂vx/∂y

    Positive curl indicates counterclockwise rotation (anticlockwise spirals).
    Negative curl indicates clockwise rotation.

    Matches MATLAB curl() function behavior.

    Parameters
    ----------
    vx : np.ndarray
        X-component of vector field, shape (rows, cols) or (rows, cols, timepoints)
    vy : np.ndarray
        Y-component of vector field, same shape as vx
    spacing : float, optional
        Grid spacing (default: 1.0)
    show_progress : bool, optional
        Show progress bar for batch processing (default: False)

    Returns
    -------
    curl_z : np.ndarray
        Curl (vorticity) field, same shape as input
        Positive = counterclockwise, negative = clockwise

    Notes
    -----
    - Uses NumPy gradient for spatial derivatives (central differences)
    - Edge pixels may have reduced accuracy due to boundary conditions
    - NaN values in input propagate to output
    - MATLAB curl() uses meshgrid convention: [curlz, cav] = curl(x, y, vx, vy)

    References
    ----------
    MATLAB: spiral_detection_surfilt.m, line 80
    MATLAB doc: https://www.mathworks.com/help/matlab/ref/curl.html
    """
    # Validate input
    if vx.shape != vy.shape:
        raise ValueError(f"Vector components must have same shape: {vx.shape} vs {vy.shape}")

    # Ensure 3D for uniform processing
    if vx.ndim == 2:
        vx = vx[..., np.newaxis]
        vy = vy[..., np.newaxis]

    rows, cols, n_timepoints = vx.shape
    curl_z = np.zeros_like(vx)

    # Setup progress bar
    iterator = range(n_timepoints)
    if show_progress and n_timepoints > 10:
        iterator = tqdm(iterator, desc="Computing curl", unit="frame")

    # Compute curl for each timepoint
    for t in iterator:
        # Extract 2D slice
        vx_t = vx[:, :, t]
        vy_t = vy[:, :, t]

        # Compute partial derivatives
        # ∂vy/∂x: gradient along columns (axis=1)
        dvy_dx = np.gradient(vy_t, spacing, axis=1)

        # ∂vx/∂y: gradient along rows (axis=0)
        dvx_dy = np.gradient(vx_t, spacing, axis=0)

        # Curl_z = ∂vy/∂x - ∂vx/∂y
        curl_z[:, :, t] = dvy_dx - dvx_dy

    return curl_z


def compute_phase_field(
    phase: np.ndarray,
    spacing: float = 1.0,
    compute_curl: bool = True,
    show_progress: bool = False,
) -> PhaseFieldResult:
    """
    Complete phase field analysis: gradient, normalization, and curl.

    This is the main entry point for phase field computation, combining:
    1. Phase gradient computation (with angular wrapping)
    2. Vector field normalization
    3. Curl (vorticity) computation

    Parameters
    ----------
    phase : np.ndarray
        Phase field array, shape (rows, cols) or (rows, cols, timepoints)
        Phase values in radians, range [-pi, pi]
    spacing : float, optional
        Grid spacing in physical units (default: 1.0)
        Should match downsample_rate from preprocessing
    compute_curl : bool, optional
        Whether to compute curl field (default: True)
    show_progress : bool, optional
        Show progress bars for batch operations (default: False)

    Returns
    -------
    PhaseFieldResult
        Complete phase field analysis results

    Examples
    --------
    >>> # Single frame
    >>> phase = np.random.uniform(-np.pi, np.pi, (100, 100))
    >>> result = compute_phase_field(phase)
    >>> print(result.curl.shape)
    (100, 100, 1)

    >>> # Time series
    >>> phase = np.random.uniform(-np.pi, np.pi, (176, 251, 240))
    >>> result = compute_phase_field(phase, spacing=2.0, show_progress=True)
    >>> print(f"Curl range: [{result.curl.min():.2f}, {result.curl.max():.2f}]")

    Notes
    -----
    - Workflow matches MATLAB spiral_detection_surfilt.m lines 46-80
    - Phase gradients use angle_subtract for proper wrapping
    - Normalized vectors are negated (MATLAB convention)
    - Curl detects rotational patterns (spirals)

    See Also
    --------
    compute_phase_gradient : Phase gradient computation
    normalize_vector_field : Vector normalization
    compute_curl_2d : Curl computation
    """
    # Count NaN values
    n_nan = int(np.sum(np.isnan(phase)))

    # 1. Compute phase gradients
    gradient_x, gradient_y = compute_phase_gradient(
        phase=phase,
        spacing=spacing,
        show_progress=show_progress,
    )

    # 2. Normalize vector field
    normalized_x, normalized_y, magnitude = normalize_vector_field(
        vx=gradient_x,
        vy=gradient_y,
    )

    # 3. Compute curl (optional)
    curl = None
    if compute_curl:
        curl = compute_curl_2d(
            vx=normalized_x,
            vy=normalized_y,
            spacing=spacing,
            show_progress=show_progress,
        )

    # 4. Package results
    result = PhaseFieldResult(
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        magnitude=magnitude,
        normalized_x=normalized_x,
        normalized_y=normalized_y,
        curl=curl,
        n_nan=n_nan,
        method="central_difference",
    )

    return result


def get_phase_field_statistics(result: PhaseFieldResult) -> dict:
    """
    Extract comprehensive statistics from phase field results.

    Parameters
    ----------
    result : PhaseFieldResult
        Phase field computation results

    Returns
    -------
    dict
        Statistics dictionary with keys:
        - n_nan: Number of NaN values
        - gradient_magnitude: {mean, std, min, max}
        - curl: {mean, std, min, max, positive_fraction, negative_fraction}
    """
    stats = {
        "n_nan": result.n_nan,
        "gradient_magnitude": result.magnitude_stats,
    }

    if result.curl is not None:
        curl_valid = result.curl[~np.isnan(result.curl)]
        stats["curl"] = {
            "mean": float(np.mean(curl_valid)) if curl_valid.size > 0 else np.nan,
            "std": float(np.std(curl_valid)) if curl_valid.size > 0 else np.nan,
            "min": float(np.min(curl_valid)) if curl_valid.size > 0 else np.nan,
            "max": float(np.max(curl_valid)) if curl_valid.size > 0 else np.nan,
            "positive_fraction": float(np.sum(curl_valid > 0) / curl_valid.size) if curl_valid.size > 0 else np.nan,
            "negative_fraction": float(np.sum(curl_valid < 0) / curl_valid.size) if curl_valid.size > 0 else np.nan,
        }

    return stats
