"""
Spiral detection visualization utilities.

Functions for visualizing detected spirals, trajectories, and pattern metadata.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Union, List
from tqdm import tqdm

from matphase.detect.spirals import SpiralDetectionResult, SpiralPattern


def plot_spiral_overlays(
    curl: np.ndarray,
    result: SpiralDetectionResult,
    timepoint: int = 0,
    figsize: Tuple[int, int] = (15, 5),
    cmap_curl: str = "RdBu_r",
    cmap_labels: str = "tab20",
    curl_vmin: Optional[float] = None,
    curl_vmax: Optional[float] = None,
    show_centroids: bool = True,
    title_prefix: str = "",
) -> plt.Figure:
    """
    Create spiral detection overlay visualization.

    Parameters:
        curl: Curl field array (2D or 3D)
        result: SpiralDetectionResult containing detected patterns
        timepoint: Time index to visualize
        figsize: Figure size
        cmap_curl: Colormap for curl field
        cmap_labels: Colormap for labeled regions
        curl_vmin: Minimum curl value for colormap
        curl_vmax: Maximum curl value for colormap
        show_centroids: Whether to show pattern centroids
        title_prefix: Prefix for plot titles

    Returns:
        Matplotlib figure object
    """
    # Extract timepoint slice
    if curl.ndim == 3:
        curl_slice = curl[:, :, timepoint]
        labeled_slice = result.labeled_volume[:, :, timepoint]
        title_suffix = f" (t={timepoint})"
    else:
        curl_slice = curl
        labeled_slice = result.labeled_volume
        title_suffix = ""

    # Auto-compute symmetric curl range if not provided
    if curl_vmin is None or curl_vmax is None:
        vmax_auto = np.nanmax(np.abs(curl_slice))
        curl_vmin = curl_vmin or -vmax_auto
        curl_vmax = curl_vmax or vmax_auto

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Curl field
    ax = axes[0]
    im = ax.imshow(curl_slice, cmap=cmap_curl, origin='lower', vmin=curl_vmin, vmax=curl_vmax)
    ax.set_title(f'{title_prefix}Curl Field{title_suffix}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Curl')

    # 2. Labeled patterns
    ax = axes[1]
    # Show curl as background, labels as overlay
    ax.imshow(curl_slice, cmap='gray', origin='lower', alpha=0.3)
    im = ax.imshow(
        np.ma.masked_where(labeled_slice == 0, labeled_slice),
        cmap=cmap_labels, origin='lower', alpha=0.7, interpolation='nearest'
    )
    ax.set_title(f'{title_prefix}Detected Patterns{title_suffix}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Pattern ID')

    # 3. Overlay with centroids
    ax = axes[2]
    ax.imshow(curl_slice, cmap=cmap_curl, origin='lower', vmin=curl_vmin, vmax=curl_vmax, alpha=0.6)
    ax.imshow(
        np.ma.masked_where(labeled_slice == 0, labeled_slice),
        cmap=cmap_labels, origin='lower', alpha=0.5, interpolation='nearest'
    )

    if show_centroids:
        # Find patterns active at this timepoint
        for pattern in result.patterns:
            # Check if pattern exists at this timepoint
            if timepoint >= pattern.start_time and timepoint < pattern.start_time + pattern.duration:
                t_idx = timepoint - pattern.start_time
                if t_idx < len(pattern.centroids):
                    cy, cx = pattern.centroids[t_idx]
                    if not (np.isnan(cx) or np.isnan(cy)):
                        ax.plot(cx, cy, 'w*', markersize=15, markeredgewidth=2, markeredgecolor='k')
                        ax.text(cx + 2, cy + 2, f'P{pattern.pattern_id}',
                               color='white', fontsize=10, weight='bold',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    ax.set_title(f'{title_prefix}Spiral Overlay{title_suffix}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()
    return fig


def plot_spiral_trajectories(
    result: SpiralDetectionResult,
    spatial_shape: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "viridis",
    show_legend: bool = True,
    max_patterns_legend: int = 10,
    title: str = "Spiral Trajectories",
) -> plt.Figure:
    """
    Plot centroid trajectories for all detected patterns.

    Parameters:
        result: SpiralDetectionResult containing patterns
        spatial_shape: Optional (height, width) for background image
        figsize: Figure size
        cmap: Colormap for trajectory colors
        show_legend: Whether to show legend
        max_patterns_legend: Maximum patterns in legend (avoid clutter)
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Set spatial bounds
    if spatial_shape is not None:
        height, width = spatial_shape
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')

    # Use colormap for different patterns
    n_patterns = len(result.patterns)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, max(n_patterns, 1)))

    for i, pattern in enumerate(result.patterns):
        centroids = np.array(pattern.centroids)  # Shape: (duration, 2) as (y, x)

        # Filter out NaN centroids
        valid_mask = ~(np.isnan(centroids[:, 0]) | np.isnan(centroids[:, 1]))
        valid_centroids = centroids[valid_mask]

        if len(valid_centroids) == 0:
            continue

        # Extract x, y (note: centroids are (y, x))
        y_coords = valid_centroids[:, 0]
        x_coords = valid_centroids[:, 1]

        # Plot trajectory
        color = colors[i % len(colors)]
        label = f'Pattern {pattern.pattern_id} (n={pattern.duration})' if i < max_patterns_legend else None

        ax.plot(x_coords, y_coords, '-o', color=color, label=label,
               alpha=0.7, linewidth=2, markersize=6)

        # Mark start and end
        ax.plot(x_coords[0], y_coords[0], 's', color=color, markersize=10,
               markeredgewidth=2, markeredgecolor='k')
        ax.plot(x_coords[-1], y_coords[-1], '^', color=color, markersize=10,
               markeredgewidth=2, markeredgecolor='k')

    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)

    if show_legend and n_patterns > 0 and n_patterns <= max_patterns_legend:
        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    return fig


def plot_pattern_statistics(
    result: SpiralDetectionResult,
    figsize: Tuple[int, int] = (15, 10),
    title_prefix: str = "",
) -> plt.Figure:
    """
    Create statistical summary plots for detected patterns.

    Parameters:
        result: SpiralDetectionResult containing patterns
        figsize: Figure size
        title_prefix: Prefix for plot titles

    Returns:
        Matplotlib figure object
    """
    if len(result.patterns) == 0:
        # Create empty placeholder figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No patterns detected', ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig

    # Extract statistics
    durations = [p.duration for p in result.patterns]
    sizes_mean = [np.nanmean(p.instantaneous_sizes) for p in result.patterns]
    powers_mean = [np.nanmean(p.instantaneous_powers) for p in result.patterns]
    widths_mean = [np.nanmean(p.instantaneous_widths) for p in result.patterns]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. Duration histogram
    ax = axes[0, 0]
    ax.hist(durations, bins=min(20, len(durations)), edgecolor='black', alpha=0.7)
    ax.set_title(f'{title_prefix}Pattern Duration')
    ax.set_xlabel('Duration (frames)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    # 2. Size histogram
    ax = axes[0, 1]
    ax.hist(sizes_mean, bins=min(20, len(sizes_mean)), edgecolor='black', alpha=0.7, color='orange')
    ax.set_title(f'{title_prefix}Mean Pattern Size')
    ax.set_xlabel('Mean Size (voxels)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    # 3. Power histogram
    ax = axes[0, 2]
    ax.hist(powers_mean, bins=min(20, len(powers_mean)), edgecolor='black', alpha=0.7, color='green')
    ax.set_title(f'{title_prefix}Mean Pattern Power')
    ax.set_xlabel('Mean Power (curl magnitude)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    # 4. Duration vs Size scatter
    ax = axes[1, 0]
    ax.scatter(durations, sizes_mean, alpha=0.6, s=50)
    ax.set_title(f'{title_prefix}Duration vs Size')
    ax.set_xlabel('Duration (frames)')
    ax.set_ylabel('Mean Size (voxels)')
    ax.grid(True, alpha=0.3)

    # 5. Duration vs Power scatter
    ax = axes[1, 1]
    ax.scatter(durations, powers_mean, alpha=0.6, s=50, color='green')
    ax.set_title(f'{title_prefix}Duration vs Power')
    ax.set_xlabel('Duration (frames)')
    ax.set_ylabel('Mean Power')
    ax.grid(True, alpha=0.3)

    # 6. Pattern width distribution
    ax = axes[1, 2]
    ax.hist(widths_mean, bins=min(20, len(widths_mean)), edgecolor='black', alpha=0.7, color='purple')
    ax.set_title(f'{title_prefix}Mean Pattern Width')
    ax.set_xlabel('Mean Width (pixels)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_detection_frames(
    curl: np.ndarray,
    result: SpiralDetectionResult,
    output_dir: Union[str, Path],
    prefix: str = "detection",
    timepoints: Optional[List[int]] = None,
    show_progress: bool = True,
    **plot_kwargs,
) -> List[Path]:
    """
    Save detection overlay frames with progress bar.

    Parameters:
        curl: Curl field array
        result: SpiralDetectionResult
        output_dir: Output directory
        prefix: Filename prefix
        timepoints: List of timepoints to export (None = all)
        show_progress: Show tqdm progress bar
        **plot_kwargs: Additional arguments passed to plot_spiral_overlays

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine timepoints
    if curl.ndim == 3:
        n_times = curl.shape[2]
        if timepoints is None:
            timepoints = list(range(n_times))
    else:
        timepoints = [0]

    saved_paths = []

    # Use tqdm for progress tracking (only show if >10 timepoints)
    iterator = timepoints
    if show_progress and len(timepoints) > 10:
        iterator = tqdm(timepoints, desc="Saving detection frames")

    for t in iterator:
        output_path = output_dir / f"{prefix}_t{t:04d}.png"
        fig = plot_spiral_overlays(curl, result, timepoint=t, **plot_kwargs)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def plot_pattern_details(
    pattern: SpiralPattern,
    curl: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 10),
    cmap_curl: str = "RdBu_r",
) -> plt.Figure:
    """
    Create detailed visualization for a single pattern.

    Parameters:
        pattern: SpiralPattern to visualize
        curl: Optional curl field for spatial context
        figsize: Figure size
        cmap_curl: Colormap for curl background

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Temporal evolution of size
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(pattern.instantaneous_sizes, 'o-', linewidth=2)
    ax.set_title(f'Pattern {pattern.pattern_id}: Size Evolution')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Size (voxels)')
    ax.grid(True, alpha=0.3)

    # 2. Temporal evolution of power
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(pattern.instantaneous_powers, 'o-', linewidth=2, color='orange')
    ax.set_title(f'Pattern {pattern.pattern_id}: Power Evolution')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Power (curl magnitude)')
    ax.grid(True, alpha=0.3)

    # 3. Temporal evolution of width
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(pattern.instantaneous_widths, 'o-', linewidth=2, color='green')
    ax.set_title(f'Pattern {pattern.pattern_id}: Width Evolution')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Width (pixels)')
    ax.grid(True, alpha=0.3)

    # 4. Centroid trajectory
    ax = fig.add_subplot(gs[1, :])
    centroids = np.array(pattern.centroids)
    valid_mask = ~(np.isnan(centroids[:, 0]) | np.isnan(centroids[:, 1]))
    valid_centroids = centroids[valid_mask]

    if len(valid_centroids) > 0:
        y_coords = valid_centroids[:, 0]
        x_coords = valid_centroids[:, 1]
        ax.plot(x_coords, y_coords, 'o-', linewidth=2, markersize=8)
        ax.plot(x_coords[0], y_coords[0], 's', markersize=12, label='Start')
        ax.plot(x_coords[-1], y_coords[-1], '^', markersize=12, label='End')
        ax.set_title(f'Pattern {pattern.pattern_id}: Centroid Trajectory')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # 5. Spatial extent (bounding box visualization)
    ax = fig.add_subplot(gs[2, :])
    if curl is not None and pattern.bounding_box is not None:
        # Extract bounding box region from curl
        bb = pattern.bounding_box
        t_start = pattern.start_time
        t_mid = t_start + pattern.duration // 2

        if curl.ndim == 3 and t_mid < curl.shape[2]:
            curl_slice = curl[bb[0]:bb[1], bb[2]:bb[3], t_mid]
            im = ax.imshow(curl_slice, cmap=cmap_curl, origin='lower')
            ax.set_title(f'Pattern {pattern.pattern_id}: Spatial Extent (t={t_mid})')
            ax.set_xlabel('X (relative)')
            ax.set_ylabel('Y (relative)')
            plt.colorbar(im, ax=ax, label='Curl')
    else:
        ax.text(0.5, 0.5, 'No spatial data available', ha='center', va='center')
        ax.axis('off')

    return fig
