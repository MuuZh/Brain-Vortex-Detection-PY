"""Map CIFTI data to full surface vertex arrays (including medial wall NaN vertices)."""

import numpy as np
from typing import Optional
import nibabel as nib
from nibabel import cifti2

from .cifti import CiftiTimeSeries


def map_cifti_to_full_surface(
    cifti_ts: CiftiTimeSeries,
    structure: str,
    expected_n_vertices: int = 32492
) -> np.ndarray:
    """Map CIFTI data to full surface array, filling medial wall with NaN.

    CIFTI files exclude medial wall vertices to save space. This function creates
    a full-resolution surface array (e.g., 32492 vertices for HCP 32k mesh) and
    fills in the CIFTI data at the correct vertex indices, leaving medial wall as NaN.

    Args:
        cifti_ts: CiftiTimeSeries object from load_cifti()
        structure: Structure name (e.g., 'CORTEX_LEFT', 'CORTEX_RIGHT')
        expected_n_vertices: Expected total number of vertices in full surface (default: 32492 for 32k mesh)

    Returns:
        Array of shape (expected_n_vertices, n_timepoints) with NaN for medial wall

    Raises:
        KeyError: If structure not found
        ValueError: If vertex indices exceed expected surface size

    Example:
        >>> ts = load_cifti('data.dtseries.nii')
        >>> # ts has 29696 CORTEX_LEFT vertices (medial wall excluded)
        >>> full_data = map_cifti_to_full_surface(ts, 'CORTEX_LEFT', 32492)
        >>> # full_data has 32492 vertices, with NaN for medial wall
        >>> print(full_data.shape)
        (32492, 240)
        >>> print(np.isnan(full_data).sum(axis=1).max())  # ~2796 NaN vertices
        2796
    """
    # Get CIFTI data for structure
    cifti_data = cifti_ts.get_structure_data(structure)  # Shape: (n_cifti_vertices, n_timepoints)
    n_cifti_vertices, n_timepoints = cifti_data.shape

    # Get the original CIFTI image to access vertex indices
    # We need to reconstruct this from the metadata
    # For now, assume sequential mapping and compute medial wall size
    n_medial_wall = expected_n_vertices - n_cifti_vertices

    if n_cifti_vertices > expected_n_vertices:
        raise ValueError(
            f"CIFTI has more vertices ({n_cifti_vertices}) than expected surface ({expected_n_vertices}). "
            f"Check if expected_n_vertices is correct."
        )

    # Create full array filled with NaN
    full_data = np.full((expected_n_vertices, n_timepoints), np.nan, dtype=np.float32)

    # IMPORTANT: We need vertex indices from BrainModelAxis to know which vertices are used
    # For now, we'll provide a helper function that needs the actual CIFTI file path
    # This is a limitation - we need to enhance load_cifti() to return vertex indices

    raise NotImplementedError(
        "This function requires vertex indices from BrainModelAxis. "
        "Use map_cifti_to_full_surface_from_file() instead, which takes the CIFTI file path."
    )


def map_cifti_to_full_surface_from_file(
    cifti_path: str,
    structure: str,
    expected_n_vertices: int = 32492
) -> tuple[np.ndarray, np.ndarray]:
    """Map CIFTI data to full surface array with vertex index mapping.

    Args:
        cifti_path: Path to CIFTI .dtseries.nii file
        structure: Structure name (e.g., 'CIFTI_STRUCTURE_CORTEX_LEFT')
        expected_n_vertices: Expected total vertices in surface

    Returns:
        Tuple of (full_data, vertex_indices):
            - full_data: Shape (expected_n_vertices, n_timepoints) with NaN for medial wall
            - vertex_indices: Array of vertex indices used in CIFTI

    Example:
        >>> full_data, indices = map_cifti_to_full_surface_from_file(
        ...     'data.dtseries.nii',
        ...     'CIFTI_STRUCTURE_CORTEX_LEFT',
        ...     32492
        ... )
        >>> print(full_data.shape, len(indices))
        (32492, 240) 29696
    """
    img = nib.load(cifti_path)
    data = img.get_fdata(dtype=np.float32)

    # Get brain axis (handle both axis orders)
    axis0 = img.header.get_axis(0)
    axis1 = img.header.get_axis(1)

    if isinstance(axis0, cifti2.BrainModelAxis):
        brain_axis = axis0
        needs_transpose = False
    elif isinstance(axis1, cifti2.BrainModelAxis):
        brain_axis = axis1
        needs_transpose = True
    else:
        raise ValueError("No BrainModelAxis found in CIFTI file")

    if needs_transpose:
        data = data.T  # Ensure (grayordinates, timepoints)

    # Find the structure
    structure_found = False
    for name, slice_obj, brain_model in brain_axis.iter_structures():
        if name == structure:
            structure_found = True

            # Extract CIFTI data for this structure
            if isinstance(slice_obj, slice):
                start = slice_obj.start or 0
                stop = slice_obj.stop
                cifti_data = data[start:stop, :]
            else:
                cifti_data = data[slice_obj, :]

            n_cifti_vertices, n_timepoints = cifti_data.shape

            # Get vertex indices from brain_axis
            # The brain_axis.vertex array contains vertex indices for each grayordinate
            # For surface structures, vertex[slice] gives the surface vertex indices
            total_nverts = brain_axis.nvertices.get(structure, 0)

            if total_nverts == 0:
                # Fallback: assume contiguous
                vertex_indices = np.arange(n_cifti_vertices, dtype=np.int32)
            else:
                # Extract vertex indices for this structure's slice
                vertex_array = brain_axis.vertex[slice_obj]
                vertex_indices = np.array(vertex_array, dtype=np.int32)

            # Validate
            if len(vertex_indices) != n_cifti_vertices:
                raise ValueError(
                    f"Vertex indices count ({len(vertex_indices)}) != CIFTI vertices ({n_cifti_vertices})"
                )

            if vertex_indices.max() >= expected_n_vertices:
                raise ValueError(
                    f"Max vertex index ({vertex_indices.max()}) >= expected vertices ({expected_n_vertices})"
                )

            # Create full array
            full_data = np.full((expected_n_vertices, n_timepoints), np.nan, dtype=np.float32)

            # Fill in CIFTI data at specified indices
            full_data[vertex_indices, :] = cifti_data

            return full_data, vertex_indices

    if not structure_found:
        available = [name for name, _, _ in brain_axis.iter_structures()]
        raise KeyError(
            f"Structure '{structure}' not found. Available: {available}"
        )
