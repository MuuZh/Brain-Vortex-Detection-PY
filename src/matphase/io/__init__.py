"""
Data I/O module for MatPhase.

Handles loading and saving of neuroimaging data formats:
- CIFTI (Connectivity Informatics Technology Initiative)
- GIFTI (Geometry format)
- Parcellation schemes
- Data validation utilities
"""

from .cifti import load_cifti, CiftiTimeSeries, CiftiMetadata
from .surface import load_surface, load_hemisphere_pair, SurfaceMesh
from .validation import (
    check_mask_coverage,
    check_grid_bounds,
    validate_preprocessing_compatibility,
)
from .parcellation import (
    load_parcellation,
    parcellation_to_mask,
    validate_parcellation_shape,
)

__all__ = [
    "load_cifti",
    "CiftiTimeSeries",
    "CiftiMetadata",
    "load_surface",
    "load_hemisphere_pair",
    "SurfaceMesh",
    "check_mask_coverage",
    "check_grid_bounds",
    "validate_preprocessing_compatibility",
    "load_parcellation",
    "parcellation_to_mask",
    "validate_parcellation_shape",
]
