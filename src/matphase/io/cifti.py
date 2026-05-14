"""CIFTI-2 file I/O for dense time series data.

This module provides functions to load CIFTI-2 (.dtseries.nii) files containing
fMRI time series data mapped to cortical surface vertices (grayordinates).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import nibabel as nib
from nibabel import cifti2


@dataclass
class CiftiMetadata:
    """Metadata extracted from CIFTI-2 file header.

    Attributes:
        n_grayordinates: Total number of grayordinates (vertices + subcortical voxels)
        n_timepoints: Number of time samples
        sampling_rate: Temporal sampling rate in Hz
        tr: Repetition time in seconds (1 / sampling_rate)
        brain_structures: List of brain structure names (e.g., 'CORTEX_LEFT')
        structure_indices: Dict mapping structure names to grayordinate index arrays
        vertex_indices: Dict mapping structure names to surface vertex indices (for surface structures)
        surface_n_vertices: Dict mapping structure names to total surface vertices (including medial wall)
    """
    n_grayordinates: int
    n_timepoints: int
    sampling_rate: float
    tr: float
    brain_structures: list[str]
    structure_indices: dict[str, np.ndarray] = field(default_factory=dict)
    vertex_indices: dict[str, np.ndarray] = field(default_factory=dict)
    surface_n_vertices: dict[str, int] = field(default_factory=dict)


@dataclass
class CiftiTimeSeries:
    """Dense CIFTI-2 time series data.

    Attributes:
        data: Time series array, shape (n_grayordinates, n_timepoints)
        time: Time axis in seconds, shape (n_timepoints,)
        metadata: CIFTI metadata (sampling rate, structures, etc.)
        has_nans: Whether data contains any NaN values
        nan_fraction: Fraction of values that are NaN (0.0 to 1.0)
    """
    data: np.ndarray
    time: np.ndarray
    metadata: CiftiMetadata
    has_nans: bool = False
    nan_fraction: float = 0.0

    def get_structure_data(self, structure: str) -> np.ndarray:
        """Extract data for a specific brain structure.

        Args:
            structure: Structure name (e.g., 'CORTEX_LEFT', 'CORTEX_RIGHT')

        Returns:
            Time series for the structure, shape (n_vertices, n_timepoints)

        Raises:
            KeyError: If structure not found in metadata
        """
        if structure not in self.metadata.structure_indices:
            available = list(self.metadata.structure_indices.keys())
            raise KeyError(
                f"Structure '{structure}' not found. "
                f"Available structures: {available}"
            )

        indices = self.metadata.structure_indices[structure]
        return self.data[indices, :]

    def get_full_surface_data(self, structure: str) -> np.ndarray:
        """Extract data for surface structure with medial wall filled as NaN.

        For surface structures (cortex), this returns a full vertex array including
        medial wall vertices as NaN. This is useful for visualization on standard
        surfaces like HCP 32k mesh where CIFTI data excludes medial wall.

        Args:
            structure: Structure name (e.g., 'CORTEX_LEFT', 'CORTEX_RIGHT')

        Returns:
            Time series array, shape (total_surface_vertices, n_timepoints)
            where total_surface_vertices includes medial wall (e.g., 32492 for 32k mesh)

        Raises:
            KeyError: If structure not found or not a surface structure
            ValueError: If structure has no vertex mapping information

        Example:
            >>> ts = load_cifti('data.dtseries.nii')
            >>> # Regular data: 29696 cortical vertices (medial wall excluded)
            >>> left_data = ts.get_structure_data('CORTEX_LEFT')
            >>> print(left_data.shape)
            (29696, 240)
            >>> # Full surface: 32492 vertices (medial wall as NaN)
            >>> left_full = ts.get_full_surface_data('CORTEX_LEFT')
            >>> print(left_full.shape)
            (32492, 240)
            >>> print(np.isnan(left_full).sum(axis=1).sum())  # ~2796 NaN vertices
            2796
        """
        if structure not in self.metadata.vertex_indices:
            raise KeyError(
                f"Structure '{structure}' not found or is not a surface structure. "
                f"Surface structures with vertex mapping: {list(self.metadata.vertex_indices.keys())}"
            )

        if structure not in self.metadata.surface_n_vertices:
            raise ValueError(
                f"No surface vertex count information for '{structure}'"
            )

        # Get CIFTI data (excluding medial wall)
        cifti_data = self.get_structure_data(structure)
        n_cifti_vertices, n_timepoints = cifti_data.shape

        # Get mapping information
        vertex_indices = self.metadata.vertex_indices[structure]
        total_vertices = self.metadata.surface_n_vertices[structure]

        # Create full array with NaN
        full_data = np.full((total_vertices, n_timepoints), np.nan, dtype=np.float32)

        # Fill in CIFTI data at specified vertex indices
        full_data[vertex_indices, :] = cifti_data

        return full_data


class CiftiLoadError(Exception):
    """Exception raised when CIFTI file cannot be loaded."""
    pass


def load_cifti(
    filepath: str | Path,
    *,
    expected_tr: Optional[float] = None,
    warn_on_nan: bool = True,
    nan_threshold: float = 0.1
) -> CiftiTimeSeries:
    """Load CIFTI-2 dense time series file.

    Args:
        filepath: Path to .dtseries.nii file
        expected_tr: Expected TR in seconds (warns if mismatch, None to skip check)
        warn_on_nan: Whether to warn if NaN values detected
        nan_threshold: Warn if NaN fraction exceeds this threshold (0.0 to 1.0)

    Returns:
        CiftiTimeSeries object with data and metadata

    Raises:
        CiftiLoadError: If file cannot be loaded or has invalid structure
        FileNotFoundError: If file does not exist

    Example:
        >>> ts = load_cifti('subject01.dtseries.nii', expected_tr=1.4)
        >>> print(ts.data.shape)
        (91282, 1200)
        >>> left_cortex = ts.get_structure_data('CORTEX_LEFT')
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CIFTI file not found: {filepath}")

    # Load CIFTI-2 image
    try:
        img = nib.load(filepath)
    except Exception as e:
        raise CiftiLoadError(f"Failed to load CIFTI file: {e}") from e

    if not isinstance(img, cifti2.Cifti2Image):
        raise CiftiLoadError(
            f"File is not a valid CIFTI-2 image (got {type(img).__name__})"
        )

    # Extract data array
    data = img.get_fdata(dtype=np.float32)

    # Parse header axes
    # CIFTI dtseries files can have axes in either order:
    # Format 1: (grayordinates, timepoints) - axis 0=BrainModel, axis 1=Series
    # Format 2: (timepoints, grayordinates) - axis 0=Series, axis 1=BrainModel
    header = img.header
    try:
        axis0 = header.get_axis(0)
        axis1 = header.get_axis(1)

        # Detect axis order
        if isinstance(axis0, cifti2.BrainModelAxis) and isinstance(axis1, cifti2.SeriesAxis):
            brain_axis = axis0
            time_axis = axis1
            data_needs_transpose = False
        elif isinstance(axis0, cifti2.SeriesAxis) and isinstance(axis1, cifti2.BrainModelAxis):
            brain_axis = axis1
            time_axis = axis0
            data_needs_transpose = True
        else:
            raise CiftiLoadError(
                f"Invalid CIFTI axes: got {type(axis0).__name__} and {type(axis1).__name__}, "
                f"expected BrainModelAxis and SeriesAxis in either order"
            )
    except IndexError as e:
        raise CiftiLoadError(f"Invalid CIFTI header structure: {e}") from e

    # Transpose data if needed to ensure (grayordinates, timepoints) format
    if data_needs_transpose:
        data = data.T

    # Extract temporal information
    sampling_rate = 1.0 / time_axis.step  # Hz
    tr = time_axis.step  # seconds
    n_timepoints = time_axis.size
    time = np.arange(n_timepoints) * tr + time_axis.start

    # Validate TR if expected value provided
    if expected_tr is not None:
        tr_diff = abs(tr - expected_tr)
        if tr_diff > 0.01:  # 10ms tolerance
            warnings.warn(
                f"CIFTI TR ({tr:.3f}s) differs from expected ({expected_tr:.3f}s) "
                f"by {tr_diff:.3f}s",
                UserWarning
            )

    # Extract brain structures
    n_grayordinates = brain_axis.size
    structure_indices = {}
    brain_structures = []
    vertex_indices = {}
    surface_n_vertices = {}

    for name, slice_obj, _ in brain_axis.iter_structures():
        # Simplify structure names (remove CIFTI_STRUCTURE_ prefix)
        if name.startswith('CIFTI_STRUCTURE_'):
            clean_name = name[16:]  # Remove prefix
        else:
            clean_name = name

        brain_structures.append(clean_name)

        # Convert slice to indices
        if isinstance(slice_obj, slice):
            start = slice_obj.start or 0
            stop = slice_obj.stop if slice_obj.stop is not None else n_grayordinates
            step = slice_obj.step or 1
            indices = np.arange(start, stop, step)
        else:
            indices = np.array(slice_obj)

        structure_indices[clean_name] = indices

        # Extract vertex mapping for surface structures
        # Check if this structure has surface vertex information
        if name in brain_axis.nvertices:
            # This is a surface structure
            total_nverts = brain_axis.nvertices[name]
            surface_n_vertices[clean_name] = total_nverts

            # Get vertex indices (which surface vertices are used)
            vertex_array = brain_axis.vertex[slice_obj]
            vertex_indices[clean_name] = np.array(vertex_array, dtype=np.int32)

    # Check for NaN values
    nan_mask = np.isnan(data)
    has_nans = nan_mask.any()
    nan_fraction = nan_mask.mean() if has_nans else 0.0

    if warn_on_nan and has_nans:
        if nan_fraction > nan_threshold:
            warnings.warn(
                f"CIFTI data contains {nan_fraction*100:.2f}% NaN values "
                f"(threshold: {nan_threshold*100:.1f}%)",
                UserWarning
            )

    # Validate data shape
    if data.shape != (n_grayordinates, n_timepoints):
        raise CiftiLoadError(
            f"Data shape mismatch: expected ({n_grayordinates}, {n_timepoints}), "
            f"got {data.shape}"
        )

    # Build metadata
    metadata = CiftiMetadata(
        n_grayordinates=n_grayordinates,
        n_timepoints=n_timepoints,
        sampling_rate=sampling_rate,
        tr=tr,
        brain_structures=brain_structures,
        structure_indices=structure_indices,
        vertex_indices=vertex_indices,
        surface_n_vertices=surface_n_vertices
    )

    return CiftiTimeSeries(
        data=data,
        time=time,
        metadata=metadata,
        has_nans=has_nans,
        nan_fraction=nan_fraction
    )


def validate_cifti_structures(
    ts: CiftiTimeSeries,
    required_structures: Optional[list[str]] = None
) -> tuple[bool, list[str]]:
    """Validate that CIFTI contains required brain structures.

    Args:
        ts: CiftiTimeSeries object
        required_structures: List of required structure names (None = no validation)

    Returns:
        Tuple of (is_valid, missing_structures)

    Example:
        >>> ts = load_cifti('data.dtseries.nii')
        >>> valid, missing = validate_cifti_structures(
        ...     ts, ['CORTEX_LEFT', 'CORTEX_RIGHT']
        ... )
        >>> if not valid:
        ...     print(f"Missing structures: {missing}")
    """
    if required_structures is None:
        return True, []

    available = set(ts.metadata.structure_indices.keys())
    required = set(required_structures)
    missing = list(required - available)

    return len(missing) == 0, sorted(missing)
