"""Spiral wave detection and analysis tools.

This package provides utilities for detecting and characterizing spiral waves
in phase field data, including surrogate null model generation for statistical
testing.

Modules
-------
phase_field : Phase gradient and curl computations for spiral detection
spirals : Connected-components based spatiotemporal pattern detection
surrogates : Fourier phase randomization for null model generation
thresholds : Statistical thresholding for spiral detection (curl, expansion, coherence)
"""

from matphase.detect.phase_field import (
    PhaseFieldResult,
    angle_subtract,
    compute_phase_gradient,
    normalize_vector_field,
    compute_curl_2d,
    compute_phase_field,
    get_phase_field_statistics,
)
from matphase.detect.spirals import (
    SpiralPattern,
    SpiralDetectionResult,
    detect_spirals,
    detect_spirals_directional,
    filter_patterns_by_curl_strength,
    get_pattern_trajectories,
    get_pattern_statistics_summary,
)
from matphase.detect.surrogates import (
    SurrogateResult,
    generate_surrogate_fft,
    generate_surrogate_batch,
    validate_power_spectrum_preservation,
)
from matphase.detect.thresholds import (
    ThresholdResult,
    apply_curl_threshold,
    compute_expansion_field,
    apply_expansion_threshold,
    compute_phase_coherence,
    apply_phase_coherence_threshold,
    compute_detection_thresholds_from_surrogates,
    apply_combined_threshold,
)

__all__ = [
    # Phase field functions
    "PhaseFieldResult",
    "angle_subtract",
    "compute_phase_gradient",
    "normalize_vector_field",
    "compute_curl_2d",
    "compute_phase_field",
    "get_phase_field_statistics",
    # Spiral detection functions
    "SpiralPattern",
    "SpiralDetectionResult",
    "detect_spirals",
    "detect_spirals_directional",
    "filter_patterns_by_curl_strength",
    "get_pattern_trajectories",
    "get_pattern_statistics_summary",
    # Surrogate functions
    "SurrogateResult",
    "generate_surrogate_fft",
    "generate_surrogate_batch",
    "validate_power_spectrum_preservation",
    # Threshold functions
    "ThresholdResult",
    "apply_curl_threshold",
    "compute_expansion_field",
    "apply_expansion_threshold",
    "compute_phase_coherence",
    "apply_phase_coherence_threshold",
    "compute_detection_thresholds_from_surrogates",
    "apply_combined_threshold",
]
