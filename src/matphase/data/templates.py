"""Parcellation template loading with caching.

This module provides utilities for loading standard brain parcellation templates
(e.g., Glasser, Schaefer) from CIFTI-2 label files. Templates are cached after
first load to avoid repeated disk I/O.

Supported formats:
- .dlabel.nii (CIFTI-2 dense label files)
- Future: .pscalar.nii (CIFTI-2 parcellated scalar)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import nibabel as nib


# Global cache for loaded templates
_TEMPLATE_CACHE: Dict[str, "Parcellation"] = {}


@dataclass
class Parcellation:
    """Represents a brain parcellation template.

    Attributes:
        name: Template name (e.g., "Glasser360", "Schaefer100")
        labels: Dictionary mapping structure names to label arrays
                Shape: (n_vertices,) with integer ROI IDs
                Example: {"CORTEX_LEFT": np.array([1, 1, 2, 2, ...]), ...}
        roi_names: List of ROI names in order (index 0 = background/medial wall)
        roi_ids: List of ROI IDs corresponding to roi_names
        metadata: Additional metadata from CIFTI file
    """
    name: str
    labels: Dict[str, np.ndarray]
    roi_names: List[str]
    roi_ids: List[int]
    metadata: Dict

    def get_roi_mask(self, structure: str, roi_id: int) -> np.ndarray:
        """Get boolean mask for vertices belonging to a specific ROI.

        Args:
            structure: Brain structure name (e.g., "CORTEX_LEFT")
            roi_id: ROI identifier (integer label value)

        Returns:
            Boolean array with True for vertices in the ROI

        Example:
            >>> parcellation = load_template("Glasser360")
            >>> v1_mask = parcellation.get_roi_mask("CORTEX_LEFT", 1)
            >>> print(f"V1 has {v1_mask.sum()} vertices")
        """
        if structure not in self.labels:
            raise ValueError(
                f"Structure '{structure}' not found. "
                f"Available: {list(self.labels.keys())}"
            )
        return self.labels[structure] == roi_id

    def get_roi_vertices(self, structure: str, roi_id: int) -> np.ndarray:
        """Get indices of vertices belonging to a specific ROI.

        Args:
            structure: Brain structure name (e.g., "CORTEX_LEFT")
            roi_id: ROI identifier (integer label value)

        Returns:
            Integer array of vertex indices
        """
        mask = self.get_roi_mask(structure, roi_id)
        return np.where(mask)[0]

    def get_structure_roi_count(self, structure: str) -> int:
        """Count unique ROIs in a structure (excluding background).

        Args:
            structure: Brain structure name

        Returns:
            Number of unique ROIs (excluding 0/medial wall)
        """
        if structure not in self.labels:
            raise ValueError(
                f"Structure '{structure}' not found. "
                f"Available: {list(self.labels.keys())}"
            )
        unique_labels = np.unique(self.labels[structure])
        # Exclude 0 (typically background/medial wall)
        return np.sum(unique_labels > 0)


def load_template(
    template_name_or_path: str | Path,
    cache: bool = True,
    force_reload: bool = False
) -> Parcellation:
    """Load a parcellation template from CIFTI label file.

    Templates are cached by default to avoid repeated disk I/O. The cache is
    keyed by the absolute path or template name.

    Args:
        template_name_or_path: Template name (e.g., "Glasser360") or path to
                                .dlabel.nii file
        cache: Whether to cache the template after loading
        force_reload: If True, reload from disk even if cached

    Returns:
        Parcellation object with ROI labels and metadata

    Raises:
        FileNotFoundError: If template file not found
        ValueError: If file is not a valid CIFTI label file

    Example:
        >>> # Load by name (searches standard locations)
        >>> template = load_template("Glasser360")
        >>>
        >>> # Load from explicit path
        >>> template = load_template("data/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii")
        >>>
        >>> # Force reload (bypass cache)
        >>> template = load_template("Glasser360", force_reload=True)
    """
    # Convert to Path if string
    if isinstance(template_name_or_path, str):
        template_path = Path(template_name_or_path)
    else:
        template_path = template_name_or_path

    # Generate cache key
    cache_key = str(template_path.resolve()) if template_path.exists() else str(template_path)

    # Check cache
    if not force_reload and cache and cache_key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[cache_key]

    # If not a valid path, try to resolve from standard locations
    if not template_path.exists():
        resolved_path = _resolve_template_path(template_path)
        if resolved_path is None:
            raise FileNotFoundError(
                f"Template not found: {template_path}. "
                f"Available templates: {list_available_templates()}"
            )
        template_path = resolved_path

    # Load CIFTI label file
    try:
        cifti_img = nib.load(str(template_path))
    except Exception as e:
        raise ValueError(f"Failed to load CIFTI file: {e}") from e

    # Verify it's a label file
    if not hasattr(cifti_img, "header") or cifti_img.header is None:
        raise ValueError(f"Not a valid CIFTI file: {template_path}")

    # Extract label axis (LabelAxis) and brain model axis (BrainModelAxis)
    header = cifti_img.header
    axes = [header.get_axis(i) for i in range(header.number_of_mapped_indices)]

    # Find LabelAxis and BrainModelAxis
    label_axis = None
    brain_axis = None
    for axis in axes:
        axis_type = type(axis).__name__
        if "Label" in axis_type:
            label_axis = axis
        elif "BrainModel" in axis_type:
            brain_axis = axis

    if label_axis is None or brain_axis is None:
        raise ValueError(
            f"Could not find LabelAxis or BrainModelAxis in {template_path}. "
            f"Found axes: {[type(ax).__name__ for ax in axes]}"
        )

    # Extract data
    data = cifti_img.get_fdata()  # Shape: (grayordinates,) or (grayordinates, 1)
    if data.ndim == 2:
        data = data[:, 0]  # Take first column if 2D

    # Extract ROI names and IDs from LabelAxis
    # LabelAxis contains label tables mapping integer IDs to names
    roi_names = []
    roi_ids = []

    # Get label table for first map (index 0)
    try:
        label_table = label_axis.label[0]  # Access first label map
        for label_id, (label_name, rgba) in label_table.items():
            roi_ids.append(label_id)
            roi_names.append(label_name)
    except (AttributeError, IndexError, KeyError) as e:
        warnings.warn(f"Could not extract label names: {e}. Using generic names.")
        unique_ids = np.unique(data.astype(int))
        roi_ids = unique_ids.tolist()
        roi_names = [f"ROI_{i}" for i in roi_ids]

    # Extract labels per structure
    labels_dict = {}
    for structure_name, data_slice, brain_model_type in brain_axis.iter_structures():
        # Remove CIFTI_STRUCTURE_ prefix for consistency
        clean_name = structure_name.replace("CIFTI_STRUCTURE_", "")

        # Extract labels for this structure
        structure_labels = data[data_slice].astype(int)
        labels_dict[clean_name] = structure_labels

    # Create metadata
    metadata = {
        "file_path": str(template_path),
        "n_structures": len(labels_dict),
        "total_grayordinates": len(data),
    }

    # Create Parcellation object
    template_name = template_path.stem  # Use filename without extension
    parcellation = Parcellation(
        name=template_name,
        labels=labels_dict,
        roi_names=roi_names,
        roi_ids=roi_ids,
        metadata=metadata,
    )

    # Cache if requested
    if cache:
        _TEMPLATE_CACHE[cache_key] = parcellation

    return parcellation


def _resolve_template_path(template_name: Path) -> Optional[Path]:
    """Resolve template name to file path by searching standard locations.

    Searches:
    1. Current working directory
    2. testdata/
    3. data/
    4. templates/

    Args:
        template_name: Template name or partial path

    Returns:
        Resolved Path if found, None otherwise
    """
    # Search paths (relative to current working directory)
    search_dirs = [
        Path.cwd(),
        Path.cwd() / "testdata",
        Path.cwd() / "data",
        Path.cwd() / "templates",
    ]

    # Common filename patterns
    if not template_name.suffix:
        # Try adding .dlabel.nii extension
        patterns = [
            template_name.with_suffix(".dlabel.nii"),
            Path(f"{template_name}.dlabel.nii"),
        ]
    else:
        patterns = [template_name]

    # Search
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            candidate = search_dir / pattern
            if candidate.exists():
                return candidate

    return None


def list_available_templates() -> List[str]:
    """List template names currently cached in memory.

    Returns:
        List of cached template names/paths

    Example:
        >>> load_template("Glasser360")
        >>> print(list_available_templates())
        ['Glasser360']
    """
    return list(_TEMPLATE_CACHE.keys())


def clear_template_cache(template_name: Optional[str] = None) -> None:
    """Clear cached templates to free memory.

    Args:
        template_name: If provided, clear only this template.
                       If None, clear all cached templates.

    Example:
        >>> clear_template_cache("Glasser360")  # Clear one
        >>> clear_template_cache()  # Clear all
    """
    global _TEMPLATE_CACHE
    if template_name is None:
        _TEMPLATE_CACHE.clear()
    else:
        _TEMPLATE_CACHE.pop(template_name, None)
