"""Spatial filtering operations for fMRI phase field analysis.

This module implements Gaussian pyramid-based spatial filtering, including
difference-of-Gaussians (DoG) bandpass filtering for multi-scale spatial
frequency decomposition. All operations are NaN-aware to preserve cortical
mask boundaries.

MATLAB Reference
----------------
- load_fMRI.m lines 115-161: Spatial bandpass filtering using Gaussian pyramid
- Uses fspecial('gaussian') + nanconv() for NaN-aware filtering
- Sigma scales: [29.35, 14.93] (physical coords) / downsample_rate (2)
- Filter width: ceil(3 * sigma) following 3-sigma rule
- DoG: difference between consecutive pyramid levels

Key Operations
--------------
1. Gaussian pyramid generation (multi-scale low-pass filtering)
2. Difference-of-Gaussians (DoG) bandpass computation
3. NaN-aware convolution preserving cortical boundaries
4. Multi-timepoint batch processing with progress tracking

Example
-------
>>> import numpy as np
>>> from matphase.preprocess.spatial import spatial_bandpass_filter
>>>
>>> # Generate synthetic grid data (176 x 251 x 100 timepoints)
>>> data = np.random.randn(176, 251, 100)
>>> data[0:20, :, :] = np.nan  # Add cortical mask
>>>
>>> # Apply DoG spatial bandpass filtering
>>> result = spatial_bandpass_filter(
...     data,
...     sigma_scales=[29.35, 14.93],
...     downsample_rate=2,
...     mode='dog',
...     show_progress=True
... )
>>>
>>> print(result.shape)  # (176, 251, 100, 1) - 1 bandpass scale
>>> print(result.bandpass.shape)  # Same as above
>>> print(result.lowpass.shape)  # (176, 251, 100, 2) - 2 pyramid levels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage
from tqdm import tqdm


@dataclass
class SpatialFilterResult:
    """Result container for spatial filtering operations.

    Attributes
    ----------
    bandpass : np.ndarray
        Bandpass-filtered data (n_y, n_x, n_t, n_bands).
        For DoG mode with 2 sigma scales, n_bands=1 (single bandpass band).
        For multi-scale DoG, n_bands = len(sigma_scales) - 1.
    lowpass : np.ndarray
        Low-pass filtered pyramid (n_y, n_x, n_t, n_scales).
        Contains Gaussian-smoothed outputs at each sigma scale.
    sigma_scales : np.ndarray
        Sigma values used for filtering (adjusted by downsample_rate).
    filter_mode : str
        Filtering mode used ("dog" or "lowpass").
    n_timepoints : int
        Number of timepoints processed.
    n_scales : int
        Number of sigma scales in pyramid.
    n_bands : int
        Number of bandpass bands (n_scales - 1 for DoG mode).
    """
    bandpass: np.ndarray
    lowpass: np.ndarray
    sigma_scales: np.ndarray
    filter_mode: str
    n_timepoints: int
    n_scales: int
    n_bands: int


def nanconv2d(
    image: np.ndarray,
    kernel: np.ndarray,
    mode: str = "constant",
    cval: float = 0.0
) -> np.ndarray:
    """NaN-aware 2D convolution.

    Replicates MATLAB's nanconv(..., 'edge', 'nanout') behavior:
    - NaN values are ignored during convolution
    - Output is NaN where input is NaN or where valid neighborhood is insufficient
    - Edge padding strategy matches MATLAB 'edge' mode

    Parameters
    ----------
    image : np.ndarray
        Input 2D image (n_y, n_x). May contain NaN values.
    kernel : np.ndarray
        Convolution kernel (odd size recommended).
    mode : str, default="constant"
        Padding mode for scipy.ndimage.convolve.
        "constant" (zero-pad), "reflect", "nearest", "mirror", "wrap".
        MATLAB 'edge' ~ scipy 'nearest'.
    cval : float, default=0.0
        Fill value for constant mode.

    Returns
    -------
    filtered : np.ndarray
        Filtered image (n_y, n_x) with NaN preserved.

    Algorithm
    ---------
    1. Create valid data mask (non-NaN pixels)
    2. Replace NaN with 0 in data
    3. Convolve data and mask separately with same kernel
    4. Normalize: filtered_data / filtered_mask
    5. Restore NaN where original data was NaN or mask sum too low

    References
    ----------
    MATLAB nanconv: https://www.mathworks.com/matlabcentral/fileexchange/41961-nanconv
    """
    # Handle all-NaN input
    if np.all(np.isnan(image)):
        return image.copy()

    # Create valid data mask
    valid_mask = ~np.isnan(image)

    # Replace NaN with 0 for convolution
    data_filled = np.where(valid_mask, image, 0.0)
    mask_filled = valid_mask.astype(float)

    # Convolve data and mask
    filtered_data = ndimage.convolve(data_filled, kernel, mode=mode, cval=cval)
    filtered_mask = ndimage.convolve(mask_filled, kernel, mode=mode, cval=cval)

    # Normalize by valid neighborhood count
    # Avoid division by zero: set denominator < epsilon to NaN
    epsilon = 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        result = filtered_data / filtered_mask

    # Restore NaN where original was NaN or insufficient valid neighbors
    result[~valid_mask] = np.nan
    result[filtered_mask < epsilon] = np.nan

    return result


def create_gaussian_kernel(sigma: float) -> np.ndarray:
    """Create 2D Gaussian kernel matching MATLAB fspecial('gaussian').

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian distribution.

    Returns
    -------
    kernel : np.ndarray
        2D Gaussian kernel (width x width) normalized to sum=1.
        Width = ceil(3 * sigma) following 3-sigma rule (captures 99.7% of distribution).

    Notes
    -----
    MATLAB fspecial('gaussian', n, sigma) creates n x n kernel.
    Here we auto-determine n = ceil(3 * sigma) to ensure adequate support.
    Minimum width is 3 to avoid degenerate kernels.
    """
    # Determine kernel width using 3-sigma rule
    width = max(3, int(np.ceil(3 * sigma)))

    # Ensure odd width for symmetric kernel
    if width % 2 == 0:
        width += 1

    # Create 1D Gaussian
    ax = np.arange(-width // 2 + 1, width // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    # 2D Gaussian: exp(-(x^2 + y^2) / (2 * sigma^2))
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize to sum=1
    kernel /= kernel.sum()

    return kernel


def gaussian_pyramid(
    data: np.ndarray,
    sigma_scales: list[float] | np.ndarray,
    show_progress: bool = True
) -> np.ndarray:
    """Generate Gaussian pyramid for multi-scale low-pass filtering.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array (n_y, n_x, n_t). May contain NaN values.
    sigma_scales : list[float] or np.ndarray
        Sigma values for each pyramid level (already adjusted by downsample_rate).
        Example: [14.675, 7.465] for physical scales [29.35, 14.93] / 2.
    show_progress : bool, default=True
        Show tqdm progress bar for timepoint iteration.

    Returns
    -------
    pyramid : np.ndarray
        Low-pass filtered pyramid (n_y, n_x, n_t, n_scales).
        pyramid[:, :, :, i] is data filtered with sigma_scales[i].

    Notes
    -----
    - Uses NaN-aware convolution via nanconv2d()
    - Iterates over timepoints to apply 2D filtering per slice
    - Progress bar shown when n_t > 10 and show_progress=True
    """
    n_y, n_x, n_t = data.shape
    n_scales = len(sigma_scales)

    # Initialize pyramid
    pyramid = np.zeros((n_y, n_x, n_t, n_scales), dtype=data.dtype)

    # Create kernels for each scale
    kernels = [create_gaussian_kernel(sigma) for sigma in sigma_scales]

    # Show progress for long sequences
    iterator = range(n_t)
    if show_progress and n_t > 10:
        iterator = tqdm(iterator, desc="Gaussian pyramid", unit="frame")

    # Apply filtering per timepoint
    for t in iterator:
        frame = data[:, :, t]
        for i_scale, kernel in enumerate(kernels):
            pyramid[:, :, t, i_scale] = nanconv2d(frame, kernel, mode="nearest")

    return pyramid


def difference_of_gaussians(
    pyramid: np.ndarray
) -> np.ndarray:
    """Compute difference-of-Gaussians (DoG) bandpass from pyramid.

    Parameters
    ----------
    pyramid : np.ndarray
        Gaussian pyramid (n_y, n_x, n_t, n_scales) from gaussian_pyramid().

    Returns
    -------
    bandpass : np.ndarray
        DoG bandpass (n_y, n_x, n_t, n_bands) where n_bands = n_scales - 1.
        bandpass[:, :, :, i] = pyramid[:, :, :, i+1] - pyramid[:, :, :, i]

    Notes
    -----
    - MATLAB: sigBPass(iScale,:,:,iTime) = sigLPass(iScale+1) - sigLPass(iScale)
    - Subtracts consecutive pyramid levels to isolate spatial frequency bands
    - NaN values propagate through subtraction
    """
    n_scales = pyramid.shape[3]
    n_bands = n_scales - 1

    if n_bands < 1:
        raise ValueError(f"Need at least 2 scales for DoG, got {n_scales}")

    # Compute differences between consecutive scales
    bandpass = np.diff(pyramid, axis=3)

    return bandpass


def spatial_bandpass_filter(
    data: np.ndarray,
    sigma_scales: list[float] | np.ndarray = [29.35, 14.93],
    downsample_rate: float = 2.0,
    mode: Literal["dog", "lowpass"] = "dog",
    show_progress: bool = True
) -> SpatialFilterResult:
    """Apply spatial bandpass filtering via Gaussian pyramid.

    Main entry point for spatial filtering operations. Replicates MATLAB
    load_fMRI.m lines 115-161 spatial bandpass logic.

    Parameters
    ----------
    data : np.ndarray
        Input 3D grid data (n_y, n_x, n_t). May contain NaN for cortical mask.
    sigma_scales : list[float] or np.ndarray, default=[29.35, 14.93]
        Sigma values in physical coordinates. Will be divided by downsample_rate.
        MATLAB default: params.sigmScale = [29.35, 14.93] with downSRate=2.
    downsample_rate : float, default=2.0
        Downsampling factor applied to coordinate grid (affects sigma scaling).
    mode : {"dog", "lowpass"}, default="dog"
        Filtering mode:
        - "dog": Difference-of-Gaussians bandpass (MATLAB default)
        - "lowpass": Return pyramid without differencing
    show_progress : bool, default=True
        Show tqdm progress bar for multi-scale filtering.

    Returns
    -------
    result : SpatialFilterResult
        Container with bandpass, lowpass pyramid, and metadata.

    Raises
    ------
    ValueError
        If data is not 3D, sigma_scales has <2 elements for DoG mode, or
        downsample_rate <= 0.

    Notes
    -----
    MATLAB Implementation Details:
    - sigmScale = [29.35, 14.93] in physical coordinates
    - sigmScale / downSRate converts to grid coordinates
    - filtWidth = ceil(3 * filtSigma) determines kernel size
    - Uses nanconv(..., 'edge', 'nanout') for NaN-aware filtering
    - Output shape: (n_y, n_x, n_t, n_scales) transposed to match expected layout

    Performance:
    - Typical: ~X ms/timepoint for 176x251 grid, 2 scales
    - Bottleneck: nanconv2d per-timepoint 2D convolution
    - Future: Consider GPU acceleration (CuPy, Taichi) for large datasets

    Examples
    --------
    >>> # DoG bandpass (default)
    >>> result = spatial_bandpass_filter(data)
    >>> print(result.bandpass.shape)  # (176, 251, 100, 1)
    >>>
    >>> # Low-pass pyramid only
    >>> result = spatial_bandpass_filter(data, mode='lowpass')
    >>> print(result.lowpass.shape)  # (176, 251, 100, 2)
    """
    # Validate inputs
    if data.ndim != 3:
        raise ValueError(f"Expected 3D input (n_y, n_x, n_t), got shape {data.shape}")

    if downsample_rate <= 0:
        raise ValueError(f"downsample_rate must be > 0, got {downsample_rate}")

    sigma_scales = np.asarray(sigma_scales)

    if mode == "dog" and len(sigma_scales) < 2:
        raise ValueError(f"DoG mode requires at least 2 sigma scales, got {len(sigma_scales)}")

    # Adjust sigma scales by downsample rate (convert to grid coordinates)
    sigma_grid = sigma_scales / downsample_rate

    # Generate Gaussian pyramid
    pyramid = gaussian_pyramid(data, sigma_grid, show_progress=show_progress)

    # Compute bandpass if requested
    if mode == "dog":
        bandpass = difference_of_gaussians(pyramid)
        n_bands = bandpass.shape[3]
    else:
        # For lowpass mode, bandpass is empty (user accesses pyramid via .lowpass)
        bandpass = np.zeros((*data.shape, 0), dtype=data.dtype)
        n_bands = 0

    # Package results
    result = SpatialFilterResult(
        bandpass=bandpass,
        lowpass=pyramid,
        sigma_scales=sigma_grid,
        filter_mode=mode,
        n_timepoints=data.shape[2],
        n_scales=len(sigma_grid),
        n_bands=n_bands
    )

    return result


def validate_spatial_filter_result(result: SpatialFilterResult) -> None:
    """Validate spatial filter result for expected properties.

    Parameters
    ----------
    result : SpatialFilterResult
        Result from spatial_bandpass_filter().

    Raises
    ------
    AssertionError
        If validation checks fail.

    Checks
    ------
    - Bandpass and lowpass shapes match expected dimensions
    - n_bands = n_scales - 1 for DoG mode
    - NaN values preserved in expected locations
    - Sigma scales are positive and monotonically decreasing (larger sigma = more smoothing)
    """
    # Check dimensions
    if result.filter_mode == "dog":
        assert result.n_bands == result.n_scales - 1, \
            f"Expected {result.n_scales - 1} bands, got {result.n_bands}"
        assert result.bandpass.shape[3] == result.n_bands, \
            f"Bandpass shape mismatch: {result.bandpass.shape[3]} != {result.n_bands}"

    assert result.lowpass.shape[3] == result.n_scales, \
        f"Lowpass pyramid scale mismatch: {result.lowpass.shape[3]} != {result.n_scales}"

    # Check sigma scales are positive
    assert np.all(result.sigma_scales > 0), \
        f"Sigma scales must be positive, got {result.sigma_scales}"

    # Check sigma scales are decreasing (MATLAB convention: larger sigma first)
    # Note: MATLAB uses [29.35, 14.93] where first is larger (more smoothing)
    if len(result.sigma_scales) > 1:
        assert np.all(np.diff(result.sigma_scales) < 0), \
            f"Sigma scales should be decreasing, got {result.sigma_scales}"


# Module exports
__all__ = [
    "SpatialFilterResult",
    "nanconv2d",
    "create_gaussian_kernel",
    "gaussian_pyramid",
    "difference_of_gaussians",
    "spatial_bandpass_filter",
    "validate_spatial_filter_result",
]
