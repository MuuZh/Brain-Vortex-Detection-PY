"""Shared helpers for analysis modules."""

from __future__ import annotations

from typing import List, Sequence, Union

from matphase.analysis.storage import SpiralAnalysisDataset
from matphase.detect.spirals import SpiralDetectionResult


DatasetInput = Union[
    SpiralAnalysisDataset,
    SpiralDetectionResult,
    Sequence[Union[SpiralAnalysisDataset, SpiralDetectionResult]],
]


def normalize_dataset_sequence(datasets: DatasetInput) -> List[SpiralAnalysisDataset]:
    """
    Normalize mixed dataset inputs into an in-memory list of SpiralAnalysisDataset objects.
    """

    if isinstance(datasets, (SpiralAnalysisDataset, SpiralDetectionResult)):
        dataset_items = [datasets]
    else:
        dataset_items = list(datasets)

    if not dataset_items:
        raise ValueError("At least one dataset or detection result is required.")

    normalized: List[SpiralAnalysisDataset] = []
    for item in dataset_items:
        if isinstance(item, SpiralAnalysisDataset):
            normalized.append(item)
        elif isinstance(item, SpiralDetectionResult):
            normalized.append(
                SpiralAnalysisDataset.from_detection_result(item, cifti_file="in-memory.dtseries.nii")
            )
        else:
            raise TypeError(f"Unsupported dataset type: {type(item)!r}")
    return normalized

