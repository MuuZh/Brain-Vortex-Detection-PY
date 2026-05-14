"""GIFTI surface geometry I/O for cortical mesh data.

This module provides functions to load GIFTI (.surf.gii) surface files containing
3D vertex coordinates and triangular face indices for cortical surface meshes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import warnings

import numpy as np
import nibabel as nib
from nibabel import gifti


# GIFTI intent codes (from NIFTI-1 standard)
INTENT_POINTSET = 1008  # 3D vertex coordinates
INTENT_TRIANGLE = 1009  # Triangular face indices


@dataclass
class SurfaceMesh:
    """Cortical surface mesh geometry.

    Attributes:
        vertices: 3D vertex coordinates, shape (n_vertices, 3)
        faces: Triangular face indices, shape (n_faces, 3)
        hemisphere: Hemisphere identifier ('left', 'right', or None)
        n_vertices: Number of vertices
        n_faces: Number of triangular faces
    """
    vertices: np.ndarray
    faces: np.ndarray
    hemisphere: Optional[Literal['left', 'right']] = None

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        """Number of triangular faces in the mesh."""
        return self.faces.shape[0]

    def get_vertex_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute bounding box of vertex coordinates.

        Returns:
            Tuple of (min_coords, max_coords), each shape (3,) for (x, y, z)
        """
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate mesh structure.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check vertex array
        if self.vertices.ndim != 2:
            errors.append(f"Vertices must be 2D, got shape {self.vertices.shape}")
        elif self.vertices.shape[1] != 3:
            errors.append(f"Vertices must have 3 columns (x,y,z), got {self.vertices.shape[1]}")

        # Check face array
        if self.faces.ndim != 2:
            errors.append(f"Faces must be 2D, got shape {self.faces.shape}")
        elif self.faces.shape[1] != 3:
            errors.append(f"Faces must have 3 columns (triangular), got {self.faces.shape[1]}")

        # Check face indices are within vertex range
        if self.faces.size > 0:
            max_index = self.faces.max()
            if max_index >= self.n_vertices:
                errors.append(
                    f"Face indices out of range: max index {max_index} >= {self.n_vertices} vertices"
                )
            if self.faces.min() < 0:
                errors.append(f"Face indices contain negative values: min={self.faces.min()}")

        # Check for NaN/Inf
        if np.any(~np.isfinite(self.vertices)):
            nan_count = np.sum(~np.isfinite(self.vertices))
            errors.append(f"Vertices contain {nan_count} NaN/Inf values")

        return len(errors) == 0, errors


class SurfaceLoadError(Exception):
    """Exception raised when surface file cannot be loaded."""
    pass


def load_surface(
    filepath: str | Path,
    *,
    hemisphere: Optional[Literal['left', 'right']] = None,
    validate_mesh: bool = True
) -> SurfaceMesh:
    """Load GIFTI surface geometry file.

    Args:
        filepath: Path to .surf.gii file
        hemisphere: Hemisphere identifier ('left' or 'right'), or None to auto-detect from filename
        validate_mesh: Whether to validate mesh structure after loading

    Returns:
        SurfaceMesh object with vertices and faces

    Raises:
        SurfaceLoadError: If file cannot be loaded or has invalid structure
        FileNotFoundError: If file does not exist

    Example:
        >>> mesh = load_surface('L.flat.32k_fs_LR.surf.gii', hemisphere='left')
        >>> print(mesh.vertices.shape)
        (32492, 3)
        >>> print(mesh.faces.shape)
        (59013, 3)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Surface file not found: {filepath}")

    # Auto-detect hemisphere from filename if not specified
    if hemisphere is None:
        filename = filepath.name.lower()
        if filename.startswith('l.') or '.l.' in filename or filename.startswith('left'):
            hemisphere = 'left'
        elif filename.startswith('r.') or '.r.' in filename or filename.startswith('right'):
            hemisphere = 'right'
        else:
            warnings.warn(
                f"Could not auto-detect hemisphere from filename: {filepath.name}",
                UserWarning
            )

    # Load GIFTI image
    try:
        img = nib.load(filepath)
    except Exception as e:
        raise SurfaceLoadError(f"Failed to load GIFTI file: {e}") from e

    if not isinstance(img, gifti.GiftiImage):
        raise SurfaceLoadError(
            f"File is not a valid GIFTI image (got {type(img).__name__})"
        )

    # Extract vertex and face arrays
    vertices = None
    faces = None

    for darray in img.darrays:
        intent = darray.intent

        if intent == INTENT_POINTSET:
            # Vertex coordinates
            vertices = darray.data
            if vertices.dtype != np.float32:
                vertices = vertices.astype(np.float32)

        elif intent == INTENT_TRIANGLE:
            # Face indices
            faces = darray.data
            if faces.dtype != np.int32:
                faces = faces.astype(np.int32)

    # Validate that we found required arrays
    if vertices is None:
        raise SurfaceLoadError(
            f"No vertex data (INTENT_POINTSET={INTENT_POINTSET}) found in GIFTI file"
        )

    if faces is None:
        raise SurfaceLoadError(
            f"No face data (INTENT_TRIANGLE={INTENT_TRIANGLE}) found in GIFTI file"
        )

    # Create mesh object
    mesh = SurfaceMesh(
        vertices=vertices,
        faces=faces,
        hemisphere=hemisphere
    )

    # Validate mesh structure
    if validate_mesh:
        is_valid, errors = mesh.validate()
        if not is_valid:
            error_msg = "\n  - ".join(["Mesh validation failed:"] + errors)
            raise SurfaceLoadError(error_msg)

    return mesh


def load_hemisphere_pair(
    left_path: str | Path,
    right_path: str | Path,
    *,
    validate_mesh: bool = True
) -> tuple[SurfaceMesh, SurfaceMesh]:
    """Load both left and right hemisphere surface meshes.

    Args:
        left_path: Path to left hemisphere .surf.gii file
        right_path: Path to right hemisphere .surf.gii file
        validate_mesh: Whether to validate mesh structures

    Returns:
        Tuple of (left_mesh, right_mesh)

    Raises:
        SurfaceLoadError: If either file cannot be loaded
        FileNotFoundError: If either file does not exist

    Example:
        >>> left, right = load_hemisphere_pair(
        ...     'L.flat.32k_fs_LR.surf.gii',
        ...     'R.flat.32k_fs_LR.surf.gii'
        ... )
        >>> print(f"Left: {left.n_vertices} vertices, Right: {right.n_vertices} vertices")
    """
    left_mesh = load_surface(left_path, hemisphere='left', validate_mesh=validate_mesh)
    right_mesh = load_surface(right_path, hemisphere='right', validate_mesh=validate_mesh)

    return left_mesh, right_mesh


def get_surface_path(
    base_dir: str | Path,
    hemisphere: Literal['left', 'right'],
    *,
    pattern: str = "{hemi}.flat.32k_fs_LR.surf.gii"
) -> Path:
    """Construct surface file path for a given hemisphere.

    Args:
        base_dir: Directory containing surface files
        hemisphere: 'left' or 'right'
        pattern: Filename pattern with {hemi} placeholder (default: HCP 32k format)

    Returns:
        Path to surface file

    Example:
        >>> path = get_surface_path('testdata', 'left')
        >>> print(path)
        testdata/L.flat.32k_fs_LR.surf.gii
    """
    base_dir = Path(base_dir)
    hemi_code = 'L' if hemisphere == 'left' else 'R'
    filename = pattern.format(hemi=hemi_code)
    return base_dir / filename
