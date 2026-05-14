"""
Threshold computation and application for spiral detection.

This module provides utilities for applying statistical thresholds to phase field
and curl data, supporting both fixed thresholds and surrogate-based dynamic thresholds.

Key functions:
- apply_curl_threshold: Filter curl field by minimum magnitude
- compute_expansion_field: Calculate divergence (expansion/contraction)
- apply_expansion_threshold: Filter by maximum expansion rate
- compute_phase_coherence: Calculate local phase gradient magnitude
- apply_phase_coherence_threshold: Filter by minimum coherence
- compute_detection_thresholds_from_surrogates: Estimate thresholds from surrogate distribution

MATLAB Reference:
- spiral_detection_surfilt.m lines 15-16, 83-84
- Curl threshold: cav_filt_pos(cav_filt_pos<1) = 0 → minimum curl = 1.0
- Expansion threshold: expansion_threshold = 1
- Phase difference threshold: phase_df_threshold = pi/6 (~0.5236)
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from tqdm import tqdm


@dataclass
class ThresholdResult:
    """
    Result of threshold application.

    Attributes:
        filtered_data: Thresholded data array (values below threshold set to 0 or NaN)
        binary_mask: Boolean mask (True where threshold exceeded)
        threshold_value: Threshold value applied
        n_passed: Number of non-NaN values that passed threshold
        n_total: Total number of non-NaN values before thresholding
        pass_fraction: Fraction of non-NaN values passing threshold
        statistics: Dict of summary statistics (mean, std, min, max of passed values)
    """

    filtered_data: np.ndarray
    binary_mask: np.ndarray
    threshold_value: float
    n_passed: int
    n_total: int
    pass_fraction: float
    statistics: dict[str, float]


def apply_curl_threshold(
    curl_field: np.ndarray,
    threshold: float = 1.0,
    absolute: bool = True,
    rotation_mode: Literal["both", "ccw", "cw"] = "both",
    fill_value: float = 0.0,
    return_binary: bool = False,
) -> ThresholdResult:
    """
    Apply curl magnitude threshold to detect spiral candidates.

    MATLAB: cav_filt_pos(cav_filt_pos<1) = 0
    Default threshold of 1.0 filters out weak vorticity, retaining only strong spirals.

    Parameters:
        curl_field: Curl (vorticity) array, shape (n_y, n_x, n_t)
        threshold: Minimum absolute curl magnitude (default: 1.0 from MATLAB)
        absolute: If True and rotation_mode='both', use absolute curl magnitude.
        rotation_mode: 'ccw' (positive curl), 'cw' (negative curl), or 'both' to
                       include both rotations. When not 'both', the `absolute`
                       argument is ignored and the appropriate sign is enforced.
        fill_value: Value to use for sub-threshold regions (0.0 or np.nan)
        return_binary: If True, return 1/0 mask; if False, preserve curl values

    Returns:
        ThresholdResult with filtered curl field and statistics
    """
    # Handle NaN values
    valid_mask = ~np.isnan(curl_field)
    n_total = np.sum(valid_mask)

    if rotation_mode not in {"both", "ccw", "cw"}:
        raise ValueError("rotation_mode must be 'both', 'ccw', or 'cw'")

    # Apply threshold
    if rotation_mode == "both":
        if absolute:
            passed_mask = np.abs(curl_field) >= threshold
        else:
            passed_mask = curl_field >= threshold
    elif rotation_mode == "ccw":
        passed_mask = curl_field >= threshold
    else:  # rotation_mode == "cw"
        passed_mask = curl_field <= -threshold

    # Combine with valid mask
    passed_mask = passed_mask & valid_mask

    # Create filtered output
    if return_binary:
        filtered_data = passed_mask.astype(float)
    else:
        filtered_data = np.where(passed_mask, curl_field, fill_value)

    # Preserve NaN from original invalid_mask in output
    filtered_data = np.where(valid_mask, filtered_data, np.nan)

    # Compute statistics on passed values
    passed_values = curl_field[passed_mask]
    n_passed = len(passed_values)
    pass_fraction = n_passed / n_total if n_total > 0 else 0.0

    statistics = {
        "mean": float(np.mean(passed_values)) if n_passed > 0 else 0.0,
        "std": float(np.std(passed_values)) if n_passed > 0 else 0.0,
        "min": float(np.min(passed_values)) if n_passed > 0 else 0.0,
        "max": float(np.max(passed_values)) if n_passed > 0 else 0.0,
    }

    return ThresholdResult(
        filtered_data=filtered_data,
        binary_mask=passed_mask,
        threshold_value=threshold,
        n_passed=n_passed,
        n_total=n_total,
        pass_fraction=pass_fraction,
        statistics=statistics,
    )


def compute_expansion_field(
    phase_gradient_x: np.ndarray,
    phase_gradient_y: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """
    Compute expansion field (divergence of phase gradient).

    Expansion measures radial outflow/inflow. Spirals should have low expansion
    (rotation-dominated, not source/sink dominated).

    Divergence: div(V) = ∂Vx/∂x + ∂Vy/∂y

    Parameters:
        phase_gradient_x: Phase gradient in x-direction, shape (n_y, n_x, n_t)
        phase_gradient_y: Phase gradient in y-direction, shape (n_y, n_x, n_t)
        dx: Grid spacing in x-direction (default: 1.0)
        dy: Grid spacing in y-direction (default: 1.0)

    Returns:
        expansion_field: Divergence array, shape (n_y, n_x, n_t)
    """
    # Check array size
    if phase_gradient_x.shape[0] < 2 or phase_gradient_x.shape[1] < 2:
        # For very small arrays, return zeros (cannot compute gradient)
        return np.zeros_like(phase_gradient_x)

    # Compute partial derivatives using central differences
    # ∂Vx/∂x
    dvx_dx = np.gradient(phase_gradient_x, dx, axis=1)
    # ∂Vy/∂y
    dvy_dy = np.gradient(phase_gradient_y, dy, axis=0)

    # Divergence = sum of partial derivatives
    expansion_field = dvx_dx + dvy_dy

    return expansion_field


def apply_expansion_threshold(
    expansion_field: np.ndarray,
    threshold: float = 1.0,
    absolute: bool = True,
    fill_value: float = 0.0,
) -> ThresholdResult:
    """
    Apply expansion threshold to filter out source/sink patterns.

    MATLAB: expansion_threshold = 1 (lines 15, 192 in spiral_detection_surfilt.m)
    Low expansion indicates rotation-dominated flow (spirals).

    Parameters:
        expansion_field: Divergence array, shape (n_y, n_x, n_t)
        threshold: Maximum allowed expansion magnitude (default: 1.0 from MATLAB)
        absolute: If True, filter by |expansion| < threshold (both expansion/contraction)
        fill_value: Value for regions exceeding threshold (0.0 or np.nan)

    Returns:
        ThresholdResult with filtered expansion field (passed = low expansion)
    """
    valid_mask = ~np.isnan(expansion_field)
    n_total = np.sum(valid_mask)

    # Low expansion = spiral candidate (opposite logic from curl threshold)
    if absolute:
        passed_mask = np.abs(expansion_field) <= threshold
    else:
        passed_mask = np.abs(expansion_field) <= threshold

    passed_mask = passed_mask & valid_mask

    # Create filtered output (keep original values where passed)
    filtered_data = np.where(passed_mask, expansion_field, fill_value)

    # Preserve NaN from original invalid_mask
    filtered_data = np.where(valid_mask, filtered_data, np.nan)

    # Statistics
    passed_values = expansion_field[passed_mask]
    n_passed = len(passed_values)
    pass_fraction = n_passed / n_total if n_total > 0 else 0.0

    statistics = {
        "mean": float(np.mean(passed_values)) if n_passed > 0 else 0.0,
        "std": float(np.std(passed_values)) if n_passed > 0 else 0.0,
        "min": float(np.min(passed_values)) if n_passed > 0 else 0.0,
        "max": float(np.max(passed_values)) if n_passed > 0 else 0.0,
    }

    return ThresholdResult(
        filtered_data=filtered_data,
        binary_mask=passed_mask,
        threshold_value=threshold,
        n_passed=n_passed,
        n_total=n_total,
        pass_fraction=pass_fraction,
        statistics=statistics,
    )


def compute_phase_coherence(
    phase_gradient_x: np.ndarray,
    phase_gradient_y: np.ndarray,
) -> np.ndarray:
    """
    Compute phase gradient coherence (magnitude of phase gradient).

    High coherence indicates smooth phase fields. Spirals typically have
    coherent phase structure radiating from center.

    Coherence = sqrt(∇φ_x² + ∇φ_y²)

    Parameters:
        phase_gradient_x: Phase gradient in x-direction, shape (n_y, n_x, n_t)
        phase_gradient_y: Phase gradient in y-direction, shape (n_y, n_x, n_t)

    Returns:
        coherence: Phase gradient magnitude, shape (n_y, n_x, n_t)
    """
    coherence = np.sqrt(phase_gradient_x**2 + phase_gradient_y**2)
    return coherence


def apply_phase_coherence_threshold(
    phase_gradient_x: np.ndarray,
    phase_gradient_y: np.ndarray,
    threshold: float = np.pi / 6,  # MATLAB: phase_df_threshold = pi/6
    fill_value: float = 0.0,
) -> ThresholdResult:
    """
    Apply phase coherence threshold to detect smooth phase fields.

    MATLAB: phase_df_threshold = pi/6 (~0.5236 radians = 30 degrees)
    High gradient magnitude indicates strong phase structure.

    Parameters:
        phase_gradient_x: Phase gradient in x-direction, shape (n_y, n_x, n_t)
        phase_gradient_y: Phase gradient in y-direction, shape (n_y, n_x, n_t)
        threshold: Minimum phase gradient magnitude (default: π/6 from MATLAB)
        fill_value: Value for sub-threshold regions (0.0 or np.nan)

    Returns:
        ThresholdResult with binary mask (True where coherence >= threshold)
    """
    coherence = compute_phase_coherence(phase_gradient_x, phase_gradient_y)

    valid_mask = ~np.isnan(coherence)
    n_total = np.sum(valid_mask)

    # High coherence = spiral candidate
    passed_mask = coherence >= threshold
    passed_mask = passed_mask & valid_mask

    # Filtered output preserves coherence values
    filtered_data = np.where(passed_mask, coherence, fill_value)

    passed_values = coherence[passed_mask]
    n_passed = len(passed_values)
    pass_fraction = n_passed / n_total if n_total > 0 else 0.0

    statistics = {
        "mean": float(np.mean(passed_values)) if n_passed > 0 else 0.0,
        "std": float(np.std(passed_values)) if n_passed > 0 else 0.0,
        "min": float(np.min(passed_values)) if n_passed > 0 else 0.0,
        "max": float(np.max(passed_values)) if n_passed > 0 else 0.0,
    }

    return ThresholdResult(
        filtered_data=filtered_data,
        binary_mask=passed_mask,
        threshold_value=threshold,
        n_passed=n_passed,
        n_total=n_total,
        pass_fraction=pass_fraction,
        statistics=statistics,
    )


def compute_detection_thresholds_from_surrogates(
    surrogate_curl_fields: list[np.ndarray],
    percentile: float = 95.0,
    compute_expansion: bool = False,
    phase_gradients: Optional[tuple[list[np.ndarray], list[np.ndarray]]] = None,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Compute detection thresholds from surrogate distribution.

    Estimates statistical significance thresholds by analyzing surrogate data
    (null hypothesis). For example, 95th percentile → p < 0.05.

    Parameters:
        surrogate_curl_fields: List of surrogate curl arrays, each (n_y, n_x, n_t)
        percentile: Percentile for threshold (e.g., 95.0 for p<0.05, 99.0 for p<0.01)
        compute_expansion: If True, also compute expansion threshold (requires phase_gradients)
        phase_gradients: Optional tuple of (surrogate_grad_x_list, surrogate_grad_y_list)
                         for expansion/coherence thresholds
        show_progress: Show progress bar for surrogate processing

    Returns:
        thresholds: Dict with keys 'curl', optionally 'expansion', 'phase_coherence'
    """
    n_surrogates = len(surrogate_curl_fields)

    if n_surrogates == 0:
        raise ValueError("No surrogate data provided")

    # Collect maximum curl values from each surrogate
    max_curl_values = []

    iterator = (
        tqdm(surrogate_curl_fields, desc="Computing curl thresholds")
        if show_progress and n_surrogates > 10
        else surrogate_curl_fields
    )

    for surrogate_curl in iterator:
        valid_curl = surrogate_curl[~np.isnan(surrogate_curl)]
        if len(valid_curl) > 0:
            max_curl_values.append(np.max(np.abs(valid_curl)))

    # Compute percentile threshold
    curl_threshold = float(np.percentile(max_curl_values, percentile))

    thresholds = {"curl": curl_threshold}

    # Optional: compute expansion threshold
    if compute_expansion and phase_gradients is not None:
        surrogate_grad_x_list, surrogate_grad_y_list = phase_gradients
        max_expansion_values = []

        iterator = (
            tqdm(
                zip(surrogate_grad_x_list, surrogate_grad_y_list),
                total=n_surrogates,
                desc="Computing expansion thresholds",
            )
            if show_progress and n_surrogates > 10
            else zip(surrogate_grad_x_list, surrogate_grad_y_list)
        )

        for grad_x, grad_y in iterator:
            expansion = compute_expansion_field(grad_x, grad_y)
            valid_exp = expansion[~np.isnan(expansion)]
            if len(valid_exp) > 0:
                max_expansion_values.append(np.max(np.abs(valid_exp)))

        expansion_threshold = float(np.percentile(max_expansion_values, percentile))
        thresholds["expansion"] = expansion_threshold

        # Also compute phase coherence threshold
        max_coherence_values = []
        for grad_x, grad_y in zip(surrogate_grad_x_list, surrogate_grad_y_list):
            coherence = compute_phase_coherence(grad_x, grad_y)
            valid_coh = coherence[~np.isnan(coherence)]
            if len(valid_coh) > 0:
                max_coherence_values.append(np.max(valid_coh))

        coherence_threshold = float(np.percentile(max_coherence_values, percentile))
        thresholds["phase_coherence"] = coherence_threshold

    return thresholds


def apply_combined_threshold(
    curl_field: np.ndarray,
    curl_threshold: float = 1.0,
    phase_gradient_x: Optional[np.ndarray] = None,
    phase_gradient_y: Optional[np.ndarray] = None,
    expansion_threshold: Optional[float] = None,
    phase_coherence_threshold: Optional[float] = None,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, dict[str, ThresholdResult]]:
    """
    Apply multiple thresholds sequentially for robust spiral detection.

    Workflow:
    1. Apply curl threshold (minimum vorticity)
    2. (Optional) Apply expansion threshold (maximum divergence)
    3. (Optional) Apply phase coherence threshold (minimum gradient magnitude)

    Parameters:
        curl_field: Curl array, shape (n_y, n_x, n_t)
        curl_threshold: Minimum curl magnitude (default: 1.0)
        phase_gradient_x: Optional phase gradient x-component for expansion/coherence
        phase_gradient_y: Optional phase gradient y-component for expansion/coherence
        expansion_threshold: Optional maximum expansion (None = skip this filter)
        phase_coherence_threshold: Optional minimum coherence (None = skip)
        fill_value: Value for filtered regions (0.0 or np.nan)

    Returns:
        filtered_curl: Final filtered curl field after all thresholds
        results: Dict mapping threshold names to ThresholdResult objects
    """
    results = {}

    # Step 1: Curl threshold
    curl_result = apply_curl_threshold(
        curl_field, threshold=curl_threshold, fill_value=fill_value
    )
    filtered_curl = curl_result.filtered_data
    combined_mask = curl_result.binary_mask
    results["curl"] = curl_result

    # Step 2: Expansion threshold (optional)
    if expansion_threshold is not None and phase_gradient_x is not None:
        expansion_field = compute_expansion_field(phase_gradient_x, phase_gradient_y)
        exp_result = apply_expansion_threshold(
            expansion_field, threshold=expansion_threshold, fill_value=fill_value
        )
        combined_mask = combined_mask & exp_result.binary_mask
        results["expansion"] = exp_result

    # Step 3: Phase coherence threshold (optional)
    if phase_coherence_threshold is not None and phase_gradient_x is not None:
        coh_result = apply_phase_coherence_threshold(
            phase_gradient_x,
            phase_gradient_y,
            threshold=phase_coherence_threshold,
            fill_value=fill_value,
        )
        combined_mask = combined_mask & coh_result.binary_mask
        results["phase_coherence"] = coh_result

    # Apply combined mask to curl field
    filtered_curl = np.where(combined_mask, curl_field, fill_value)

    return filtered_curl, results
