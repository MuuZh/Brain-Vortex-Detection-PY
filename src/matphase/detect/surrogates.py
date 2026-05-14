"""Surrogate data generation for null model statistical testing.

This module implements Fourier phase randomization to generate surrogate
datasets that preserve the power spectrum while destroying phase relationships.
Used for null hypothesis testing in spiral wave detection.

MATLAB Reference
----------------
- load_fMRI.m lines 82-113: 3D FFT phase randomization
- Uses fftn() + fftshift() to center DC component
- Randomizes phase with uniform distribution [-π, π]
- Enforces conjugate symmetry for real-valued output
- Note: Line 110 uses exp(1i*phaseData + 1i*phaseDataRand) which ADDS
  original and random phases (unusual - standard approach is replacement)

Key Operations
--------------
1. 3D FFT of input data (NaN replaced with 0)
2. Extract magnitude and phase from frequency domain
3. Randomize phase values while preserving conjugate symmetry
4. Inverse FFT to obtain surrogate time series
5. Preserve original data shape and NaN structure

Example
-------
>>> import numpy as np
>>> from matphase.detect.surrogates import generate_surrogate_fft
>>>
>>> # Generate synthetic grid data (176 x 251 x 217 timepoints)
>>> data = np.random.randn(176, 251, 217)
>>> data[0:20, :, :] = np.nan  # Add cortical mask
>>>
>>> # Generate single surrogate realization
>>> result = generate_surrogate_fft(data, random_seed=42)
>>>
>>> print(result.surrogate.shape)  # (176, 251, 217)
>>> print(result.preserves_spectrum)  # True
>>> print(result.random_seed)  # 42
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from tqdm import tqdm


@dataclass
class SurrogateResult:
    """Result container for surrogate generation.

    Attributes
    ----------
    surrogate : np.ndarray
        Surrogate data with same shape as input. NaN structure preserved.
    original_shape : tuple[int, ...]
        Shape of original input data.
    n_nan_replaced : int
        Number of NaN values replaced with 0 before FFT.
    random_seed : int | None
        Random seed used (None if non-reproducible).
    method : str
        Generation method ("fft_phase_randomization").
    preserves_spectrum : bool
        Whether power spectrum is preserved (always True for FFT method).
    phase_mode : str
        Phase randomization mode ("replace" or "add").
        - "replace": Standard approach, replace original phase with random phase
        - "add": MATLAB approach, add random phase to original phase
    """
    surrogate: np.ndarray
    original_shape: tuple[int, ...]
    n_nan_replaced: int
    random_seed: int | None
    method: str
    preserves_spectrum: bool
    phase_mode: str


def generate_surrogate_fft(
    data: np.ndarray,
    random_seed: int | None = None,
    phase_mode: Literal["replace", "add"] = "replace",
    enforce_symmetry: bool = True
) -> SurrogateResult:
    """Generate surrogate data via 3D Fourier phase randomization.

    Implements MATLAB load_fMRI.m lines 82-113 phase randomization logic.
    Preserves power spectrum while destroying temporal and spatial phase
    relationships.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array (n_y, n_x, n_t). May contain NaN values.
        Note: n_t should be odd for symmetric FFT indexing (MATLAB convention).
    random_seed : int or None, default=None
        Random seed for reproducibility. If None, non-reproducible.
    phase_mode : {"replace", "add"}, default="replace"
        Phase randomization mode:
        - "replace": Standard approach, use only random phase (recommended)
        - "add": MATLAB approach, add random phase to original phase
    enforce_symmetry : bool, default=True
        Enforce conjugate symmetry to guarantee real-valued output.
        Should always be True for real input data.

    Returns
    -------
    result : SurrogateResult
        Container with surrogate data and metadata.

    Raises
    ------
    ValueError
        If data is not 3D.

    Notes
    -----
    MATLAB Implementation (load_fMRI.m lines 82-113):
    1. Replace NaN with 0 (line 98)
    2. Compute midpoint: floor(size/2)+1, convert to linear index
    3. FFT: freqData = fftshift(fftn(dataIn))
    4. Extract: phaseData = angle(freqData), absData = abs(freqData)
    5. Randomize first half: phaseDataRand(1:indMid-1) = rand * 2π - π
    6. Enforce symmetry: phaseDataRand(indMid+i) = -phaseDataRand(indMid-i)
    7. Reconstruct: freqDataNew = ifftshift(absData .* exp(1i*phaseData + 1i*phaseDataRand))
    8. Inverse FFT: dataOut = real(ifftn(freqDataNew))

    Key differences from standard phase randomization:
    - MATLAB adds phases: exp(1i*phaseData + 1i*phaseDataRand)
    - Standard replaces: exp(1i*phaseDataRand)
    - Default here is "replace" (standard), use phase_mode="add" for MATLAB parity

    Algorithm Details:
    - fftshift() centers DC component at array center
    - Linear indexing: MATLAB sub2ind(size, mid_y, mid_x, mid_t)
    - Conjugate symmetry: F(-k) = conj(F(k)) for real signals
    - Phase randomization destroys phase coherence while preserving amplitude

    Performance:
    - Typical: ~Y ms for 175x251x217 array
    - Bottleneck: fftn/ifftn operations
    - Future: Consider GPU acceleration (CuPy) for large datasets

    References
    ----------
    Theiler et al. (1992). Testing for nonlinearity in time series: the method
    of surrogate data. Physica D, 58(1-4), 77-94.

    Examples
    --------
    >>> # Standard phase replacement (recommended)
    >>> result = generate_surrogate_fft(data, random_seed=42, phase_mode='replace')
    >>>
    >>> # MATLAB parity (phase addition)
    >>> result = generate_surrogate_fft(data, random_seed=42, phase_mode='add')
    """
    # Validate input
    if data.ndim != 3:
        raise ValueError(f"Expected 3D input (n_y, n_x, n_t), got shape {data.shape}")

    n_y, n_x, n_t = data.shape
    original_shape = data.shape

    # Replace NaN with 0 (MATLAB line 98)
    nan_mask = np.isnan(data)
    n_nan_replaced = np.sum(nan_mask)
    data_filled = np.where(nan_mask, 0.0, data)

    # Initialize RNG
    rng = np.random.default_rng(random_seed)

    # 3D FFT (MATLAB lines 102-104)
    freq_data = np.fft.fftn(data_filled)
    magnitude = np.abs(freq_data)

    # Generate random phase uniformly distributed in [-π, π]
    phase_random = rng.uniform(-np.pi, np.pi, size=(n_y, n_x, n_t))

    # Enforce Hermitian symmetry for real-valued output
    # For real input: F(-k) = conj(F(k)), which means phase(-k) = -phase(k]
    # and magnitude[-k] = magnitude[k]
    if enforce_symmetry:
        # Identify Nyquist frequencies (self-conjugate bins)
        # These occur at index 0 (DC) and index N/2 (Nyquist) for even dimensions
        # These bins must have zero phase (purely real) to maintain Hermitian symmetry

        # DC component (always present)
        phase_random[0, 0, 0] = 0

        # Nyquist frequencies for even-sized dimensions
        # y-axis Nyquist plane
        if n_y % 2 == 0:
            phase_random[n_y // 2, :, :] = 0

        # x-axis Nyquist plane
        if n_x % 2 == 0:
            phase_random[:, n_x // 2, :] = 0

        # t-axis Nyquist plane
        if n_t % 2 == 0:
            phase_random[:, :, n_t // 2] = 0

        # Enforce conjugate symmetry for all other bins
        # For 3D FFT output, frequency k at index [iy, ix, it]
        # has conjugate -k at index [(N_y-iy) % N_y, (N_x-ix) % N_x, (N_t-it) % N_t]
        for iy in range(n_y):
            for ix in range(n_x):
                for it in range(n_t):
                    # Compute conjugate indices
                    iy_conj = (n_y - iy) % n_y
                    ix_conj = (n_x - ix) % n_x
                    it_conj = (n_t - it) % n_t

                    # Skip self-conjugate bins (already handled above)
                    if (iy, ix, it) == (iy_conj, ix_conj, it_conj):
                        continue

                    # Only process if we're in the "first half" to avoid double-setting
                    # Use lexicographic ordering to define "first half"
                    if (iy, ix, it) < (iy_conj, ix_conj, it_conj):
                        # Current random phase
                        phase_val = phase_random[iy, ix, it]
                        # Set conjugate to negative phase
                        phase_random[iy_conj, ix_conj, it_conj] = -phase_val

    # Reconstruct frequency data with randomized phase
    if phase_mode == "add":
        # MATLAB approach: add original + random phase (line 110)
        phase_original = np.angle(freq_data)
        freq_data_new = magnitude * np.exp(1j * (phase_original + phase_random))
    elif phase_mode == "replace":
        # Standard approach: replace original phase with random phase
        freq_data_new = magnitude * np.exp(1j * phase_random)
    else:
        raise ValueError(f"Invalid phase_mode: {phase_mode}. Choose 'replace' or 'add'.")

    # Inverse FFT (MATLAB line 111)
    surrogate_complex = np.fft.ifftn(freq_data_new)

    # Verify Hermitian symmetry: imaginary part should be negligible
    max_imag = np.max(np.abs(np.imag(surrogate_complex)))
    tolerance = 1e-8  # Numerical noise from FFT symmetry enforcement
    if enforce_symmetry and max_imag > tolerance:
        import warnings
        warnings.warn(
            f"Hermitian symmetry not preserved: max imaginary component = {max_imag:.2e}. "
            "This indicates a bug in phase randomization symmetry enforcement.",
            RuntimeWarning
        )

    # Take real part (should be negligible imaginary if symmetry enforced correctly)
    surrogate = np.real(surrogate_complex)

    # Restore NaN structure from original data
    surrogate[nan_mask] = np.nan

    # Package result
    result = SurrogateResult(
        surrogate=surrogate,
        original_shape=original_shape,
        n_nan_replaced=n_nan_replaced,
        random_seed=random_seed,
        method="fft_phase_randomization",
        preserves_spectrum=True,
        phase_mode=phase_mode
    )

    return result


def generate_surrogate_batch(
    data: np.ndarray,
    n_surrogates: int = 1,
    random_seed: int | None = None,
    phase_mode: Literal["replace", "add"] = "replace",
    show_progress: bool = True
) -> list[SurrogateResult]:
    """Generate batch of surrogate realizations.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array (n_y, n_x, n_t). May contain NaN values.
    n_surrogates : int, default=1
        Number of surrogate realizations to generate.
    random_seed : int or None, default=None
        Base random seed. If provided, each realization uses seed+i.
        If None, all realizations are non-reproducible.
    phase_mode : {"replace", "add"}, default="replace"
        Phase randomization mode (see generate_surrogate_fft).
    show_progress : bool, default=True
        Show tqdm progress bar for batch generation.

    Returns
    -------
    results : list[SurrogateResult]
        List of surrogate results, one per realization.

    Notes
    -----
    - If random_seed is provided, realization i uses seed = random_seed + i
    - Progress bar shown when n_surrogates > 1 and show_progress=True
    - Each realization is independent with different random phase pattern

    Examples
    --------
    >>> # Generate 100 surrogate realizations with reproducible seeds
    >>> results = generate_surrogate_batch(data, n_surrogates=100, random_seed=42)
    >>> surrogates = np.stack([r.surrogate for r in results], axis=-1)
    >>> print(surrogates.shape)  # (176, 251, 217, 100)
    """
    if n_surrogates < 1:
        raise ValueError(f"n_surrogates must be >= 1, got {n_surrogates}")

    results = []

    # Show progress for multiple surrogates
    iterator = range(n_surrogates)
    if show_progress and n_surrogates > 1:
        iterator = tqdm(iterator, desc="Generating surrogates", unit="realization")

    for i in iterator:
        # Increment seed for each realization if base seed provided
        seed = (random_seed + i) if random_seed is not None else None

        result = generate_surrogate_fft(
            data,
            random_seed=seed,
            phase_mode=phase_mode,
            enforce_symmetry=True
        )

        results.append(result)

    return results


def validate_power_spectrum_preservation(
    original: np.ndarray,
    surrogate: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> tuple[bool, float]:
    """Validate that surrogate preserves power spectrum of original.

    Parameters
    ----------
    original : np.ndarray
        Original data (n_y, n_x, n_t). NaN values handled by replacement.
    surrogate : np.ndarray
        Surrogate data with same shape. NaN values handled identically.
    rtol : float, default=1e-5
        Relative tolerance for np.allclose comparison.
    atol : float, default=1e-8
        Absolute tolerance for np.allclose comparison.

    Returns
    -------
    is_preserved : bool
        True if power spectra match within tolerance.
    max_relative_error : float
        Maximum relative error across all frequencies.

    Notes
    -----
    Power spectrum = abs(FFT(data))^2
    Phase randomization preserves magnitude, hence power spectrum.
    Small numerical errors expected due to floating-point arithmetic.

    Examples
    --------
    >>> result = generate_surrogate_fft(data, random_seed=42)
    >>> is_valid, error = validate_power_spectrum_preservation(data, result.surrogate)
    >>> assert is_valid, f"Power spectrum not preserved: max error = {error}"
    """
    # Replace NaN with 0 (same as generation step)
    original_filled = np.where(np.isnan(original), 0.0, original)
    surrogate_filled = np.where(np.isnan(surrogate), 0.0, surrogate)

    # Compute power spectra
    fft_original = np.fft.fftn(original_filled)
    fft_surrogate = np.fft.fftn(surrogate_filled)

    power_original = np.abs(fft_original) ** 2
    power_surrogate = np.abs(fft_surrogate) ** 2

    # Check if spectra match
    is_preserved = np.allclose(power_original, power_surrogate, rtol=rtol, atol=atol)

    # Compute maximum relative error
    # Only consider bins with significant power to avoid division by near-zero
    power_threshold = np.max(power_original) * 1e-10
    significant_bins = power_original > power_threshold

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_error = np.abs(power_surrogate - power_original) / (power_original + power_threshold)

    # Only compute max error over significant bins
    if np.any(significant_bins):
        max_relative_error = np.max(relative_error[significant_bins])
    else:
        max_relative_error = 0.0

    return is_preserved, max_relative_error


# Module exports
__all__ = [
    "SurrogateResult",
    "generate_surrogate_fft",
    "generate_surrogate_batch",
    "validate_power_spectrum_preservation",
]
