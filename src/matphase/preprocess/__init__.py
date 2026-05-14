"""
Preprocessing module for MatPhase.

Provides signal processing operations:
- Temporal filtering and phase extraction
- Spatial interpolation and filtering
- Surrogate data generation
- Denoising utilities
"""

from matphase.preprocess.interpolate import (
    shift_coordinates_to_positive,
    generate_coordinate_grid,
    generate_cortical_mask,
    interpolate_to_grid,
    interpolate_to_grid_batch,
    get_nan_statistics,
)

from matphase.preprocess.temporal import (
    PhaseExtractionResult,
    PhaseExtractor,
    HilbertPhaseExtractor,
    GeneralizedPhaseExtractor,
    temporal_bandpass_filter,
    validate_phase_range,
)

from matphase.preprocess.spatial import (
    SpatialFilterResult,
    nanconv2d,
    create_gaussian_kernel,
    gaussian_pyramid,
    difference_of_gaussians,
    spatial_bandpass_filter,
    validate_spatial_filter_result,
)

__all__ = [
    # Interpolation
    "shift_coordinates_to_positive",
    "generate_coordinate_grid",
    "generate_cortical_mask",
    "interpolate_to_grid",
    "interpolate_to_grid_batch",
    "get_nan_statistics",
    # Temporal filtering
    "PhaseExtractionResult",
    "PhaseExtractor",
    "HilbertPhaseExtractor",
    "GeneralizedPhaseExtractor",
    "temporal_bandpass_filter",
    "validate_phase_range",
    # Spatial filtering
    "SpatialFilterResult",
    "nanconv2d",
    "create_gaussian_kernel",
    "gaussian_pyramid",
    "difference_of_gaussians",
    "spatial_bandpass_filter",
    "validate_spatial_filter_result",
]
