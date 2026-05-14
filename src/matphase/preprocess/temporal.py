"""
Temporal filtering and phase extraction for fMRI preprocessing.

This module provides bandpass filtering and phase extraction with support for
multiple methods via a strategy pattern. Current implementation uses Hilbert
transform; Generalized Phase (GP) method is now supported for wideband signals.

Key components:
- PhaseExtractionResult: Unified output dataclass
- PhaseExtractor: Abstract base class for extraction strategies
- HilbertPhaseExtractor: Current implementation using scipy.signal.hilbert
- GeneralizedPhaseExtractor: Future placeholder for GP method
- temporal_bandpass_filter: Main entry point for filtering + phase extraction

Example:
    >>> from matphase.preprocess import temporal_bandpass_filter
    >>> result = temporal_bandpass_filter(
    ...     data=interpolated_grid,  # (176, 251, 1200)
    ...     sampling_rate=1.389,     # Hz
    ...     freq_low=0.01,
    ...     freq_high=0.1,
    ...     phase_method="hilbert"
    ... )
    >>> phase = result.phase         # (176, 251, 1200) in [-pi, pi]
    >>> amplitude = result.amplitude # (176, 251, 1200)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, hilbert
from scipy.interpolate import CubicSpline
import warnings
from tqdm import tqdm

from matphase.utils import get_logger


# ============================================================================
# Output Data Structures
# ============================================================================

@dataclass
class PhaseExtractionResult:
    """Results from phase extraction methods.

    Provides a unified interface for outputs from different phase extraction
    methods (Hilbert, Generalized Phase, etc.). All methods return required
    fields (bandpassed, phase, amplitude); optional fields depend on method.

    Attributes:
        bandpassed: Bandpass filtered signal, shape (n_channels, n_timepoints)
            or (n_y, n_x, n_timepoints) if 3D input
        phase: Instantaneous phase in radians, range [-pi, pi),
            same shape as bandpassed
        amplitude: Amplitude envelope (positive values), same shape as bandpassed
        analytic: Complex analytic signal (optional, Hilbert method only),
            same shape as bandpassed
        inst_freq: Instantaneous frequency in Hz (optional, GP method),
            same shape as bandpassed
        neg_freq_mask: Boolean mask marking negative frequency corrections
            (optional, GP method), same shape as bandpassed
        method: Name of extraction method used ("hilbert" or "generalized_phase")
    """
    bandpassed: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray
    analytic: np.ndarray | None = None
    inst_freq: np.ndarray | None = None
    neg_freq_mask: np.ndarray | None = None
    method: str = "hilbert"


# ============================================================================
# Phase Extraction Strategy Interface
# ============================================================================

class PhaseExtractor(ABC):
    """Abstract base class for phase extraction methods.

    Defines the interface for phase extraction strategies. Subclasses implement
    specific methods (Hilbert, Generalized Phase, wavelet-based, etc.) while
    maintaining consistent input/output contracts.
    """

    @abstractmethod
    def extract(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        **kwargs
    ) -> PhaseExtractionResult:
        """Extract phase, amplitude, and optional outputs from bandpassed signal.

        Args:
            signal: Bandpass-filtered signal, shape (n_channels, n_timepoints)
            sampling_rate: Sampling rate in Hz
            **kwargs: Method-specific parameters

        Returns:
            PhaseExtractionResult with required fields (bandpassed, phase,
            amplitude) and method-specific optional fields
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name for configuration and logging."""
        pass


# ============================================================================
# Concrete Implementations
# ============================================================================

class HilbertPhaseExtractor(PhaseExtractor):
    """Phase extraction using scipy.signal.hilbert transform.

    Implements the Hilbert transform approach for narrowband signals. For a
    real signal x(t), the analytic signal is:
        z(t) = x(t) + i * H[x(t)]
    where H[·] is the Hilbert transform. Phase and amplitude are:
        phase(t) = angle(z(t))     in [-pi, pi]
        amplitude(t) = |z(t)|

    This method assumes the signal is narrowband (i.e., bandpass filtered
    before phase extraction). For wideband signals, consider Generalized Phase.

    Args:
        return_analytic: Whether to return complex analytic signal in results
        show_progress: Show tqdm progress bar for multi-channel processing
        output_dtype: Precision for phase/amplitude outputs ("float32" or "float64")
    """

    def __init__(self, return_analytic: bool = True, show_progress: bool = False,
                 output_dtype: str = "float32"):
        self.return_analytic = return_analytic
        self.show_progress = show_progress
        self.output_dtype = np.float64 if output_dtype == "float64" else np.float32

    def extract(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        **kwargs
    ) -> PhaseExtractionResult:
        """Extract phase via Hilbert transform.

        Args:
            signal: Bandpassed signal, shape (n_channels, n_timepoints)
            sampling_rate: Sampling rate in Hz (not used for Hilbert, kept
                for interface consistency with other methods)
            **kwargs: Ignored (for interface compatibility)

        Returns:
            PhaseExtractionResult with bandpassed=signal, phase, amplitude,
            and optionally analytic signal
        """
        n_channels, n_timepoints = signal.shape

        # Allocate output arrays
        analytic_signal = np.zeros(
            (n_channels, n_timepoints), dtype=np.complex128)
        phase = np.zeros((n_channels, n_timepoints), dtype=self.output_dtype)
        amplitude = np.zeros((n_channels, n_timepoints),
                             dtype=self.output_dtype)

        # Process each channel
        iterator = range(n_channels)
        if self.show_progress and n_channels > 100:
            iterator = tqdm(iterator, desc="Hilbert transform", unit="channel")

        for i in iterator:
            if np.all(np.isnan(signal[i, :])):
                # Skip all-NaN channels (e.g., masked-out regions)
                analytic_signal[i, :] = np.nan
                phase[i, :] = np.nan
                amplitude[i, :] = np.nan
            else:
                # Compute Hilbert transform
                analytic_signal[i, :] = hilbert(signal[i, :])

                # Extract phase and amplitude
                phase[i, :] = np.angle(analytic_signal[i, :])  # [-pi, pi]
                amplitude[i, :] = np.abs(analytic_signal[i, :])

        return PhaseExtractionResult(
            bandpassed=signal,
            phase=phase,
            amplitude=amplitude,
            analytic=analytic_signal if self.return_analytic else None,
            method="hilbert"
        )

    @property
    def name(self) -> str:
        return "hilbert"


class GeneralizedPhaseExtractor(PhaseExtractor):
    """Phase extraction using Generalized Phase (GP) method.

    The Generalized Phase method computes phase for non-narrowband signals and
    corrects negative-frequency segments by masking, interpolating, and
    re-wrapping the phase. Optional GP-specific bandpass filtering can be
    enabled via filter_range to mimic the Julia reference implementation.

    Args:
        filter_range: Optional (low, high) Hz for GP frequency range,
            None uses bandpass filter range
        phase_correction_threshold: Threshold for negative frequency detection
        neg_freq_extension: Extend each detected negative-frequency run by this
            multiple of its length (default 3)
        return_inst_freq: Include instantaneous frequency in results
        return_neg_freq_mask: Include negative frequency correction mask
        show_progress: Show tqdm progress bar for processing
        output_dtype: Precision for outputs ("float32" or "float64")
        apply_internal_filter: Force enabling/disabling internal GP bandpass.
            Default behavior is to filter only when filter_range is provided.

    References:
        [Add GP method references when implemented]
    """

    def __init__(
        self,
        filter_range: tuple[float, float] | None = None,
        phase_correction_threshold: float = 0.0,
        neg_freq_extension: int = 3,
        return_inst_freq: bool = True,
        return_neg_freq_mask: bool = False,
        show_progress: bool = False,
        output_dtype: str = "float32",
        apply_internal_filter: bool | None = None,
    ):
        self.filter_range = filter_range
        self.phase_correction_threshold = phase_correction_threshold
        self.neg_freq_extension = max(1, int(neg_freq_extension))
        self.return_inst_freq = return_inst_freq
        self.return_neg_freq_mask = return_neg_freq_mask
        self.show_progress = show_progress
        self.output_dtype = np.float64 if output_dtype == "float64" else np.float32
        # Backward compatibility: if apply_internal_filter is None, enable only when filter_range is set
        self.apply_internal_filter = apply_internal_filter

    def extract(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        t: np.ndarray | None = None,
        **kwargs
    ) -> PhaseExtractionResult:
        """Extract generalized phase from signal.

        Args:
            signal: Bandpassed signal, shape (n_channels, n_timepoints)
            sampling_rate: Sampling rate in Hz
            t: Optional time vector, shape (n_timepoints,)
            **kwargs: Additional GP-specific parameters

        Returns:
            PhaseExtractionResult with GP-specific outputs

        Raises:
        """
        n_channels, n_timepoints = signal.shape
        dt = 1.0 / sampling_rate if sampling_rate > 0 else np.nan

        # Decide whether to apply internal bandpass (GP-specific). Default: only when filter_range is set.
        use_internal_filter = (
            self.apply_internal_filter
            if self.apply_internal_filter is not None
            else self.filter_range is not None
        )

        bandpassed = np.asarray(signal, dtype=np.float64, copy=True)
        if use_internal_filter and self.filter_range is not None:
            low, high = self.filter_range
            nyquist = sampling_rate / 2.0
            if high >= nyquist:
                raise ValueError(
                    f"GP filter high cutoff ({high} Hz) must be < Nyquist ({nyquist} Hz)"
                )
            if low >= high:
                raise ValueError(
                    f"GP filter low cutoff ({low} Hz) must be < high cutoff ({high} Hz)"
                )
            b, a = butter(4, [low / nyquist, high / nyquist],
                          btype="bandpass", output="ba")
            iterator = range(n_channels)
            if self.show_progress and n_channels > 500:
                iterator = tqdm(
                    iterator, desc="GP internal bandpass", unit="channel")
            for i in iterator:
                if np.all(np.isnan(bandpassed[i, :])):
                    bandpassed[i, :] = np.nan
                    continue
                bandpassed[i, :] = filtfilt(
                    b, a, bandpassed[i, :], padtype="odd")

        analytic_signal = np.zeros(
            (n_channels, n_timepoints), dtype=np.complex128)
        phase = np.zeros((n_channels, n_timepoints), dtype=self.output_dtype)
        amplitude = np.zeros((n_channels, n_timepoints),
                             dtype=self.output_dtype)
        inst_freq = np.full((n_channels, n_timepoints),
                            np.nan, dtype=self.output_dtype)
        neg_freq_mask = np.zeros((n_channels, n_timepoints), dtype=bool)

        iterator = range(n_channels)
        if self.show_progress and n_channels > 1000:
            iterator = tqdm(iterator, desc="Generalized Phase", unit="channel")

        for i in iterator:
            channel = bandpassed[i, :]
            if np.all(np.isnan(channel)):
                analytic_signal[i, :] = np.nan
                phase[i, :] = np.nan
                amplitude[i, :] = np.nan
                continue

            analytic_signal[i, :] = hilbert(channel)
            raw_phase = np.angle(analytic_signal[i, :])
            amplitude[i, :] = np.abs(analytic_signal[i, :]).astype(
                self.output_dtype, copy=False)

            # Instantaneous frequency (Hz) via consecutive phase differences
            freq = np.zeros_like(raw_phase, dtype=self.output_dtype)
            if n_timepoints > 1 and not np.isnan(dt):
                freq[:-1] = np.angle(
                    analytic_signal[i, 1:] *
                    np.conjugate(analytic_signal[i, :-1])
                ) / (2 * np.pi * dt)
                freq[-1] = freq[-2] if n_timepoints > 2 else freq[:-
                                                                  1].mean() if n_timepoints > 1 else 0.0
            inst_freq[i, :] = freq if self.return_inst_freq else np.nan

            # Determine dominant direction and identify negative-frequency segments
            dir_sign = np.sign(np.nanmean(freq)) if np.any(
                ~np.isnan(freq)) else 0.0
            idx = dir_sign * freq < self.phase_correction_threshold
            if idx.size > 0:
                idx[0] = False  # follow Julia logic

            # Label runs of True and extend by neg_freq_extension factor
            if np.any(idx):
                labels = np.zeros_like(idx, dtype=int)
                current_label = 0
                for t in range(1, n_timepoints):
                    if idx[t]:
                        if not idx[t - 1]:
                            current_label += 1
                        labels[t] = current_label
                for k in range(1, current_label + 1):
                    run_idx = np.where(labels == k)[0]
                    if run_idx.size == 0:
                        continue
                    run_length = run_idx[-1] - run_idx[0] + 1
                    extend_to = min(
                        n_timepoints,
                        run_idx[0] + run_length * self.neg_freq_extension
                    )
                    idx[run_idx[0]:extend_to] = True
                neg_freq_mask[i, :] = idx

            # Unwrap -> mask negative freq -> interpolate gaps -> rewrap to [-pi, pi]
            unwrapped = np.unwrap(raw_phase)
            if np.any(idx):
                unwrapped[idx] = np.nan
                valid = ~np.isnan(unwrapped)
                if np.count_nonzero(valid) >= 2:
                    cs = CubicSpline(np.flatnonzero(
                        valid), unwrapped[valid], bc_type="natural", extrapolate=True)
                    unwrapped[~valid] = cs(np.flatnonzero(~valid))
            wrapped = ((unwrapped + np.pi) % (2 * np.pi)) - np.pi
            phase[i, :] = wrapped.astype(self.output_dtype, copy=False)

            # Mask inst_freq where negative-frequency corrections were applied
            if self.return_inst_freq:
                inst_freq[i, idx] = np.nan

        return PhaseExtractionResult(
            bandpassed=bandpassed.astype(self.output_dtype, copy=False),
            phase=phase,
            amplitude=amplitude,
            analytic=analytic_signal,
            inst_freq=inst_freq if self.return_inst_freq else None,
            neg_freq_mask=neg_freq_mask if self.return_neg_freq_mask else None,
            method="generalized_phase"
        )

    @property
    def name(self) -> str:
        return "generalized_phase"


# ============================================================================
# Main Filtering Function
# ============================================================================

def temporal_bandpass_filter(
    data: np.ndarray,
    sampling_rate: float,
    freq_low: float = 0.01,
    freq_high: float = 0.1,
    filter_order: int = 4,
    phase_method: Literal["hilbert", "generalized_phase"] = "hilbert",
    return_analytic: bool = True,
    phase_extractor: PhaseExtractor | None = None,
    demean: bool = True,
    show_progress: bool = True,
    filter_method: Literal["sosfiltfilt", "filtfilt"] = "sosfiltfilt",
    filter_padtype: Literal["odd", "even", "constant"] = "odd",
    filter_precision: Literal["float32", "float64"] = "float32",
    **phase_kwargs
) -> PhaseExtractionResult:
    """Apply temporal bandpass filter and extract phase/amplitude.

    Main entry point for temporal filtering in the preprocessing pipeline.
    Supports multiple phase extraction methods via strategy pattern. Handles
    both 2D (n_pixels, n_timepoints) and 3D (n_y, n_x, n_timepoints) input.

    Processing steps:
    1. Reshape 3D input to 2D if needed
    2. Design 4th-order Butterworth bandpass filter (SOS format)
    3. Demean each channel (optional)
    4. Apply zero-phase filtering (sosfiltfilt)
    5. Extract phase and amplitude using selected method
    6. Reshape outputs to match input shape

    Args:
        data: Input signal, shape (n_y, n_x, n_timepoints) for interpolated
            grids or (n_pixels, n_timepoints). Can contain NaN values.
        sampling_rate: Sampling rate in Hz (e.g., 1.389 Hz for TR=0.72s)
        freq_low: Bandpass lower cutoff frequency in Hz (default: 0.01)
        freq_high: Bandpass upper cutoff frequency in Hz (default: 0.1)
        filter_order: Butterworth filter order (default: 4)
        phase_method: Phase extraction method, "hilbert" or "generalized_phase"
        return_analytic: Return complex analytic signal (Hilbert method only)
        phase_extractor: Optional custom PhaseExtractor instance (overrides
            phase_method parameter)
        demean: Demean each channel before filtering (default: True, matches MATLAB)
        show_progress: Show tqdm progress bar for filtering (default: True)
        filter_method: Filtering method, "sosfiltfilt" (default) or "filtfilt"
            sosfiltfilt uses SOS format (scipy default), filtfilt uses ba format
            for more control over padding. For MATLAB parity testing.
        filter_padtype: Padding type for filtfilt, "odd" (default), "even", or "constant"
            - odd: odd extension (reflects with slope)
            - even: even extension (mirrors signal)
            - constant: pads with edge values (closest to MATLAB default)
            Only applies when filter_method="filtfilt"
        filter_precision: Output precision, "float32" (default, fast) or "float64"
            (MATLAB-like precision). Note: Hilbert uses complex128 regardless.
        **phase_kwargs: Additional keyword arguments for phase extraction method
            (e.g., filter_range, smoothing_window for GP)

    Returns:
        PhaseExtractionResult with required fields (bandpassed, phase, amplitude)
        and method-specific optional fields. Output shape matches input shape.

    Raises:
        ValueError: If data shape is invalid, or freq_high >= Nyquist frequency
    Example:
        >>> # Basic usage with Hilbert method
        >>> result = temporal_bandpass_filter(
        ...     data=interpolated_data,  # (176, 251, 1200)
        ...     sampling_rate=1.389,
        ...     freq_low=0.01,
        ...     freq_high=0.1,
        ...     phase_method="hilbert"
        ... )
        >>> phase = result.phase  # (176, 251, 1200) in [-pi, pi]
        >>>
        >>> # GP method with optional internal bandpass
        >>> result_gp = temporal_bandpass_filter(
        ...     data=interpolated_data,
        ...     sampling_rate=1.389,
        ...     phase_method="generalized_phase",
        ...     filter_range=(0.01, 0.1)
        ... )
    """
    logger = get_logger(__name__)
    logger.info(
        "Temporal filter: sampling_rate=%.6f Hz (TR=%.6f s), bandpass=[%.4f, %.4f] Hz, order=%d",
        sampling_rate,
        (1.0 / sampling_rate) if sampling_rate > 0 else float("inf"),
        freq_low,
        freq_high,
        filter_order,
    )

    # Validate and reshape input
    original_shape = data.shape
    if data.ndim == 3:
        n_y, n_x, n_timepoints = data.shape
        data_2d = data.reshape(-1, n_timepoints)  # (n_pixels, n_timepoints)
    elif data.ndim == 2:
        data_2d = data
        n_timepoints = data.shape[1]
    else:
        raise ValueError(
            f"Data must be 2D (n_pixels, n_timepoints) or "
            f"3D (n_y, n_x, n_timepoints), got shape {data.shape}"
        )

    n_channels = data_2d.shape[0]

    # Design Butterworth bandpass filter (second-order sections for stability)
    nyquist = sampling_rate / 2.0
    if freq_high >= nyquist:
        raise ValueError(
            f"High cutoff frequency ({freq_high} Hz) must be < Nyquist "
            f"frequency ({nyquist} Hz). Check sampling_rate={sampling_rate} Hz."
        )

    if freq_low >= freq_high:
        raise ValueError(
            f"Low cutoff ({freq_low} Hz) must be < high cutoff ({freq_high} Hz)"
        )

    # Normalize frequencies to Nyquist
    low_norm = freq_low / nyquist
    high_norm = freq_high / nyquist

    # Determine output dtype based on precision setting
    output_dtype = np.float64 if filter_precision == "float64" else np.float32

    # Design filter based on selected method
    if filter_method == "sosfiltfilt":
        # SOS format (more numerically stable, scipy default)
        sos = butter(
            filter_order,
            [low_norm, high_norm],
            btype='bandpass',
            output='sos'
        )
        filter_coefs = sos
        logger.info("Using sosfiltfilt with SOS format")
    else:  # filtfilt
        # ba format (for more control over padding)
        b, a = butter(
            filter_order,
            [low_norm, high_norm],
            btype='bandpass',
            output='ba'
        )
        filter_coefs = (b, a)
        logger.info(f"Using filtfilt with ba format, padtype={filter_padtype}")

    # Apply bandpass filter to each channel
    bandpassed = np.zeros_like(data_2d, dtype=output_dtype)

    iterator = range(n_channels)
    if show_progress and n_channels > 1000:
        iterator = tqdm(iterator, desc="Bandpass filtering", unit="channel")

    for i in iterator:
        if np.all(np.isnan(data_2d[i, :])):
            # Skip all-NaN channels (masked regions, medial wall, etc.)
            bandpassed[i, :] = np.nan
        else:
            # Demean if requested (matches MATLAB convention)
            if demean:
                signal = data_2d[i, :] - np.nanmean(data_2d[i, :])
            else:
                signal = data_2d[i, :]

            # Apply zero-phase filtering (forward-backward filter)
            if filter_method == "sosfiltfilt":
                bandpassed[i, :] = sosfiltfilt(filter_coefs, signal)
            else:  # filtfilt
                b, a = filter_coefs
                bandpassed[i, :] = filtfilt(
                    b, a, signal, padtype=filter_padtype)

    # Select phase extraction method
    if phase_extractor is None:
        if phase_method == "hilbert":
            phase_extractor = HilbertPhaseExtractor(
                return_analytic=return_analytic,
                show_progress=show_progress,
                output_dtype=filter_precision
            )
            logger.info("Using Hilbert phase extraction method")
        elif phase_method == "generalized_phase":
            phase_extractor = GeneralizedPhaseExtractor(
                show_progress=show_progress,
                **phase_kwargs
            )
            logger.info("Using Generalized Phase extraction method")
        else:
            raise ValueError(
                f"Unknown phase_method '{phase_method}'. "
                f"Valid options: 'hilbert', 'generalized_phase'"
            )

    # Extract phase and amplitude
    result = phase_extractor.extract(bandpassed, sampling_rate, **phase_kwargs)

    # Reshape outputs to match original input shape
    if data.ndim == 3:
        result.bandpassed = result.bandpassed.reshape(original_shape)
        result.phase = result.phase.reshape(original_shape)
        result.amplitude = result.amplitude.reshape(original_shape)

        if result.analytic is not None:
            result.analytic = result.analytic.reshape(original_shape)
        if result.inst_freq is not None:
            result.inst_freq = result.inst_freq.reshape(original_shape)
        if result.neg_freq_mask is not None:
            result.neg_freq_mask = result.neg_freq_mask.reshape(original_shape)

    return result


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_phase_range(
    phase: np.ndarray,
    tolerance: float = 1e-6
) -> dict[str, bool | float | int | str]:
    """Validate that phase values are in expected [-pi, pi] range.

    Checks that all non-NaN phase values fall within [-pi, pi] ± tolerance.
    Useful for verifying phase extraction outputs.

    Args:
        phase: Phase array in radians (any shape)
        tolerance: Tolerance for range checking (default: 1e-6)

    Returns:
        dict with keys:
            - valid: bool, True if all values in range
            - min: float, minimum non-NaN value
            - max: float, maximum non-NaN value
            - out_of_range_count: int, number of values outside range
            - message: str, validation message

    Example:
        >>> result = temporal_bandpass_filter(data, fs=1.389)
        >>> validation = validate_phase_range(result.phase)
        >>> assert validation['valid'], validation['message']
    """
    valid_mask = ~np.isnan(phase)
    valid_phase = phase[valid_mask]

    if len(valid_phase) == 0:
        return {
            'valid': True,
            'min': np.nan,
            'max': np.nan,
            'out_of_range_count': 0,
            'message': 'No valid phase values (all NaN)'
        }

    phase_min = float(valid_phase.min())
    phase_max = float(valid_phase.max())

    # Count values outside [-pi, pi] ± tolerance
    out_of_range = np.sum(
        (valid_phase < -np.pi - tolerance) | (valid_phase > np.pi + tolerance)
    )

    is_valid = (out_of_range == 0)

    if is_valid:
        message = f"OK: phase in [{phase_min:.6f}, {phase_max:.6f}]"
    else:
        message = (
            f"ERROR: {out_of_range} values out of [-pi, pi] range. "
            f"Actual range: [{phase_min:.6f}, {phase_max:.6f}]"
        )

    return {
        'valid': is_valid,
        'min': phase_min,
        'max': phase_max,
        'out_of_range_count': int(out_of_range),
        'message': message
    }
