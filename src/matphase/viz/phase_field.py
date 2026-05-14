"""
Phase field visualization utilities.

Functions for visualizing gradients, curl fields, and vector fields
from phase extraction results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Union
from tqdm import tqdm

from matphase.detect.phase_field import PhaseFieldResult


def plot_phase_field(
    result: PhaseFieldResult,
    timepoint: int = 0,
    figsize: Tuple[int, int] = (15, 10),
    cmap_grad: str = "viridis",
    cmap_curl: str = "RdBu_r",
    show_vectors: bool = True,
    vector_spacing: int = 5,
    title_prefix: str = "",
) -> plt.Figure:
    """
    Create comprehensive phase field visualization.

    Parameters:
        result: PhaseFieldResult containing gradients and curl
        timepoint: Time index to visualize (if 3D)
        figsize: Figure size
        cmap_grad: Colormap for gradient magnitude
        cmap_curl: Colormap for curl (diverging recommended)
        show_vectors: Whether to overlay vector field
        vector_spacing: Spacing between vector arrows
        title_prefix: Optional prefix for plot titles

    Returns:
        Matplotlib figure object
    """
    # Extract data for specified timepoint
    if result.gradient_x.ndim == 3:
        gx = result.gradient_x[:, :, timepoint]
        gy = result.gradient_y[:, :, timepoint]
        curl = result.curl[:, :, timepoint] if result.curl is not None else None
        mag = result.magnitude[:, :, timepoint]
        vnx = result.normalized_x[:, :, timepoint]
        vny = result.normalized_y[:, :, timepoint]
    else:
        gx = result.gradient_x
        gy = result.gradient_y
        curl = result.curl
        mag = result.magnitude
        vnx = result.normalized_x
        vny = result.normalized_y

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    title_suffix = f" (t={timepoint})" if result.gradient_x.ndim == 3 else ""

    # 1. Gradient X
    ax = axes[0, 0]
    im = ax.imshow(gx, cmap=cmap_grad, origin='lower')
    ax.set_title(f'{title_prefix}Gradient X{title_suffix}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='dPhase/dx')

    # 2. Gradient Y
    ax = axes[0, 1]
    im = ax.imshow(gy, cmap=cmap_grad, origin='lower')
    ax.set_title(f'{title_prefix}Gradient Y{title_suffix}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='dPhase/dy')

    # 3. Gradient Magnitude with optional vector overlay
    ax = axes[1, 0]
    im = ax.imshow(mag, cmap=cmap_grad, origin='lower')
    ax.set_title(f'{title_prefix}Gradient Magnitude{title_suffix}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='|Gradient|')

    if show_vectors and vnx is not None and vny is not None:
        # Overlay normalized vector field
        height, width = mag.shape
        y_coords, x_coords = np.mgrid[0:height:vector_spacing, 0:width:vector_spacing]

        # Sample vectors
        u = vnx[::vector_spacing, ::vector_spacing]
        v = vny[::vector_spacing, ::vector_spacing]

        # Mask invalid vectors
        valid = ~(np.isnan(u) | np.isnan(v))

        ax.quiver(
            x_coords[valid], y_coords[valid],
            u[valid], v[valid],
            color='white', alpha=0.6, scale=30, width=0.003,
            headwidth=4, headlength=5
        )

    # 4. Curl field
    ax = axes[1, 1]
    if curl is not None:
        # Use symmetric color scale around zero
        vmax = np.nanmax(np.abs(curl))
        im = ax.imshow(curl, cmap=cmap_curl, origin='lower', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{title_prefix}Curl (Vorticity){title_suffix}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Curl')
    else:
        ax.text(0.5, 0.5, 'Curl not computed', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{title_prefix}Curl (Vorticity){title_suffix}')
        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_curl_field(
    curl: np.ndarray,
    timepoint: int = 0,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Curl Field",
    colorbar_label: str = "Curl",
) -> plt.Figure:
    """
    Create focused curl field visualization.

    Parameters:
        curl: Curl array (2D or 3D)
        timepoint: Time index if 3D
        figsize: Figure size
        cmap: Colormap (diverging recommended)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        title: Plot title
        colorbar_label: Label for colorbar

    Returns:
        Matplotlib figure object
    """
    # Extract timepoint if 3D
    if curl.ndim == 3:
        curl_slice = curl[:, :, timepoint]
        title = f"{title} (t={timepoint})"
    else:
        curl_slice = curl

    # Auto-compute symmetric range if not provided
    if vmin is None or vmax is None:
        vmax_auto = np.nanmax(np.abs(curl_slice))
        vmin = vmin or -vmax_auto
        vmax = vmax or vmax_auto

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(curl_slice, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label=colorbar_label)

    plt.tight_layout()
    return fig


def plot_gradient_field(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    timepoint: int = 0,
    figsize: Tuple[int, int] = (12, 5),
    cmap: str = "viridis",
    vector_spacing: int = 5,
    title_prefix: str = "Gradient Field",
) -> plt.Figure:
    """
    Create gradient field visualization with vector overlay.

    Parameters:
        gradient_x: X component of gradient
        gradient_y: Y component of gradient
        timepoint: Time index if 3D
        figsize: Figure size
        cmap: Colormap for magnitude
        vector_spacing: Spacing between vector arrows
        title_prefix: Prefix for plot titles

    Returns:
        Matplotlib figure object
    """
    # Extract timepoint if 3D
    if gradient_x.ndim == 3:
        gx = gradient_x[:, :, timepoint]
        gy = gradient_y[:, :, timepoint]
        title_suffix = f" (t={timepoint})"
    else:
        gx = gradient_x
        gy = gradient_y
        title_suffix = ""

    # Compute magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # Normalize for vectors
    norm_factor = np.sqrt(gx**2 + gy**2)
    norm_factor[norm_factor == 0] = 1.0
    vnx = gx / norm_factor
    vny = gy / norm_factor

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Magnitude map
    ax = axes[0]
    im = ax.imshow(magnitude, cmap=cmap, origin='lower')
    ax.set_title(f'{title_prefix} Magnitude{title_suffix}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='|Gradient|')

    # 2. Vector field
    ax = axes[1]
    ax.imshow(magnitude, cmap=cmap, origin='lower', alpha=0.3)

    height, width = magnitude.shape
    y_coords, x_coords = np.mgrid[0:height:vector_spacing, 0:width:vector_spacing]

    u = vnx[::vector_spacing, ::vector_spacing]
    v = vny[::vector_spacing, ::vector_spacing]

    valid = ~(np.isnan(u) | np.isnan(v))

    ax.quiver(
        x_coords[valid], y_coords[valid],
        u[valid], v[valid],
        magnitude[::vector_spacing, ::vector_spacing][valid],
        cmap=cmap, scale=20, width=0.004,
        headwidth=4, headlength=5, alpha=0.8
    )
    ax.set_title(f'{title_prefix} Vectors{title_suffix}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()
    return fig


def save_phase_field_snapshot(
    result: PhaseFieldResult,
    output_path: Union[str, Path],
    timepoint: int = 0,
    **plot_kwargs,
) -> None:
    """
    Save phase field visualization to file.

    Parameters:
        result: PhaseFieldResult to visualize
        output_path: Output file path (PNG recommended)
        timepoint: Time index to visualize
        **plot_kwargs: Additional arguments passed to plot_phase_field
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_phase_field(result, timepoint=timepoint, **plot_kwargs)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_phase_field_batch(
    result: PhaseFieldResult,
    output_dir: Union[str, Path],
    prefix: str = "phase_field",
    timepoints: Optional[list] = None,
    show_progress: bool = True,
    **plot_kwargs,
) -> list:
    """
    Save multiple phase field snapshots with progress bar.

    Parameters:
        result: PhaseFieldResult to visualize
        output_dir: Output directory
        prefix: Filename prefix
        timepoints: List of timepoints to export (None = all)
        show_progress: Show tqdm progress bar
        **plot_kwargs: Additional arguments passed to plot_phase_field

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine timepoints
    if result.gradient_x.ndim == 3:
        n_times = result.gradient_x.shape[2]
        if timepoints is None:
            timepoints = list(range(n_times))
    else:
        timepoints = [0]

    saved_paths = []

    # Use tqdm for progress tracking
    iterator = tqdm(timepoints, desc="Saving phase field frames") if show_progress else timepoints

    for t in iterator:
        output_path = output_dir / f"{prefix}_t{t:04d}.png"
        save_phase_field_snapshot(result, output_path, timepoint=t, **plot_kwargs)
        saved_paths.append(output_path)

    return saved_paths
