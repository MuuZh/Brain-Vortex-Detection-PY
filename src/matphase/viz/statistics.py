"""
Statistical visualization utilities for detection results.

Functions for comparing real vs surrogate data, threshold curves, and summary statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union


def plot_detection_summary(
    real_stats: Dict[str, float],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Detection Summary Statistics",
) -> plt.Figure:
    """
    Create summary visualization of detection statistics.

    Parameters:
        real_stats: Dictionary of statistics (e.g., from get_pattern_statistics_summary)
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract statistics
    n_patterns = real_stats.get('n_patterns', 0)
    total_voxels = real_stats.get('total_pattern_voxels', 0)
    mean_duration = real_stats.get('mean_duration', 0)
    mean_size = real_stats.get('mean_size', 0)
    mean_power = real_stats.get('mean_power', 0)

    # 1. Pattern count
    ax = axes[0, 0]
    ax.bar(['Patterns'], [n_patterns], color='steelblue', edgecolor='black', width=0.4)
    ax.set_title('Number of Patterns')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Total coverage
    ax = axes[0, 1]
    ax.bar(['Coverage'], [total_voxels], color='orange', edgecolor='black', width=0.4)
    ax.set_title('Total Pattern Coverage')
    ax.set_ylabel('Voxels')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Mean duration
    ax = axes[1, 0]
    ax.bar(['Duration'], [mean_duration], color='green', edgecolor='black', width=0.4)
    ax.set_title('Mean Pattern Duration')
    ax.set_ylabel('Frames')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Mean characteristics
    ax = axes[1, 1]
    metrics = ['Size', 'Power']
    values = [mean_size, mean_power]
    x_pos = np.arange(len(metrics))
    ax.bar(x_pos, values, color=['purple', 'red'], edgecolor='black', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_title('Mean Pattern Characteristics')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_surrogate_comparison(
    real_values: np.ndarray,
    surrogate_values: List[np.ndarray],
    metric_name: str = "Curl Magnitude",
    threshold_percentile: Optional[float] = 95.0,
    figsize: Tuple[int, int] = (12, 5),
    title_prefix: str = "",
) -> plt.Figure:
    """
    Compare real data against surrogate distribution.

    Parameters:
        real_values: Real data values (flattened)
        surrogate_values: List of surrogate arrays (each flattened)
        metric_name: Name of the metric being compared
        threshold_percentile: Percentile threshold to highlight
        figsize: Figure size
        title_prefix: Prefix for plot titles

    Returns:
        Matplotlib figure object
    """
    # Flatten arrays and remove NaNs
    real_flat = real_values.flatten()
    real_flat = real_flat[~np.isnan(real_flat)]

    surrogate_flat_list = []
    for surr in surrogate_values:
        surr_flat = surr.flatten()
        surr_flat = surr_flat[~np.isnan(surr_flat)]
        surrogate_flat_list.append(surr_flat)

    # Concatenate all surrogates
    all_surrogates = np.concatenate(surrogate_flat_list) if surrogate_flat_list else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Histograms
    ax = axes[0]
    bins = np.linspace(
        min(real_flat.min(), all_surrogates.min() if len(all_surrogates) > 0 else 0),
        max(real_flat.max(), all_surrogates.max() if len(all_surrogates) > 0 else 1),
        50
    )

    if len(all_surrogates) > 0:
        ax.hist(all_surrogates, bins=bins, alpha=0.5, label='Surrogates',
               color='gray', edgecolor='black', density=True)

    ax.hist(real_flat, bins=bins, alpha=0.7, label='Real Data',
           color='steelblue', edgecolor='black', density=True)

    # Mark threshold if provided
    if threshold_percentile is not None and len(all_surrogates) > 0:
        threshold = np.percentile(all_surrogates, threshold_percentile)
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                  label=f'{threshold_percentile}th percentile')

    ax.set_title(f'{title_prefix}Distribution Comparison')
    ax.set_xlabel(metric_name)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative distribution
    ax = axes[1]

    if len(all_surrogates) > 0:
        surr_sorted = np.sort(all_surrogates)
        surr_cdf = np.arange(1, len(surr_sorted) + 1) / len(surr_sorted)
        ax.plot(surr_sorted, surr_cdf, label='Surrogates', color='gray', linewidth=2)

    real_sorted = np.sort(real_flat)
    real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    ax.plot(real_sorted, real_cdf, label='Real Data', color='steelblue', linewidth=2)

    # Mark threshold
    if threshold_percentile is not None and len(all_surrogates) > 0:
        ax.axhline(threshold_percentile / 100, color='red', linestyle='--',
                  linewidth=2, label=f'{threshold_percentile}th percentile')
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2)

    ax.set_title(f'{title_prefix}Cumulative Distribution')
    ax.set_xlabel(metric_name)
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_threshold_curves(
    thresholds: Dict[str, float],
    data_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Detection Thresholds",
) -> plt.Figure:
    """
    Visualize threshold values as horizontal lines.

    Parameters:
        thresholds: Dictionary mapping threshold names to values
        data_ranges: Optional dictionary of (min, max) ranges for each threshold
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    threshold_names = list(thresholds.keys())
    threshold_values = list(thresholds.values())

    y_positions = np.arange(len(threshold_names))

    # Plot threshold bars
    colors = plt.cm.Set3(np.linspace(0, 1, len(threshold_names)))

    for i, (name, value) in enumerate(zip(threshold_names, threshold_values)):
        # Plot threshold line
        ax.barh(y_positions[i], value, height=0.6, color=colors[i],
               edgecolor='black', alpha=0.7, label=name)

        # Add value label
        ax.text(value, y_positions[i], f'  {value:.3f}',
               va='center', fontweight='bold')

        # Optionally show data range
        if data_ranges and name in data_ranges:
            min_val, max_val = data_ranges[name]
            ax.plot([min_val, max_val], [y_positions[i], y_positions[i]],
                   'k-', alpha=0.3, linewidth=8, zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(threshold_names)
    ax.set_xlabel('Threshold Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_parameter_sweep(
    parameter_values: np.ndarray,
    detection_counts: np.ndarray,
    parameter_name: str = "Threshold",
    figsize: Tuple[int, int] = (10, 6),
    highlight_value: Optional[float] = None,
    title: str = "Parameter Sensitivity Analysis",
) -> plt.Figure:
    """
    Plot detection counts as a function of parameter values.

    Parameters:
        parameter_values: Array of parameter values tested
        detection_counts: Array of detection counts for each parameter
        parameter_name: Name of the parameter
        figsize: Figure size
        highlight_value: Optional parameter value to highlight
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(parameter_values, detection_counts, 'o-', linewidth=2, markersize=8,
           color='steelblue')

    if highlight_value is not None:
        # Find closest parameter value
        idx = np.argmin(np.abs(parameter_values - highlight_value))
        ax.plot(parameter_values[idx], detection_counts[idx], 'r*',
               markersize=20, markeredgewidth=2, markeredgecolor='black',
               label=f'Selected: {highlight_value:.3f}')
        ax.axvline(parameter_values[idx], color='red', linestyle='--',
                  alpha=0.5, linewidth=2)

    ax.set_xlabel(parameter_name)
    ax.set_ylabel('Number of Detections')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if highlight_value is not None:
        ax.legend()

    plt.tight_layout()
    return fig


def plot_confusion_matrix_style(
    real_detected: int,
    real_missed: int,
    surrogate_detected: int,
    surrogate_passed: int,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Detection Classification",
) -> plt.Figure:
    """
    Create confusion-matrix-style visualization for detection results.

    Parameters:
        real_detected: Number of real patterns detected
        real_missed: Number of real patterns missed
        surrogate_detected: Number of surrogate patterns incorrectly detected
        surrogate_passed: Number of surrogate patterns correctly rejected
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create 2x2 grid
    data = np.array([
        [real_detected, real_missed],
        [surrogate_detected, surrogate_passed]
    ])

    im = ax.imshow(data, cmap='Blues', alpha=0.6)

    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Detected', 'Rejected'])
    ax.set_yticklabels(['Real Data', 'Surrogates'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, data[i, j],
                          ha="center", va="center", color="black",
                          fontsize=20, fontweight='bold')

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    return fig


def plot_statistical_power(
    real_metric: float,
    surrogate_mean: float,
    surrogate_std: float,
    threshold: float,
    metric_name: str = "Detection Metric",
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Statistical Power Analysis",
) -> plt.Figure:
    """
    Visualize statistical power: real value vs surrogate distribution.

    Parameters:
        real_metric: Observed real data metric
        surrogate_mean: Mean of surrogate distribution
        surrogate_std: Standard deviation of surrogate distribution
        threshold: Detection threshold
        metric_name: Name of the metric
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Generate surrogate distribution curve
    x = np.linspace(surrogate_mean - 4 * surrogate_std,
                    surrogate_mean + 4 * surrogate_std, 200)
    y = (1 / (surrogate_std * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((x - surrogate_mean) / surrogate_std) ** 2)

    # Plot surrogate distribution
    ax.plot(x, y, 'gray', linewidth=2, label='Surrogate Distribution')
    ax.fill_between(x, 0, y, where=(x < threshold), color='gray', alpha=0.3,
                    label='Null Hypothesis')
    ax.fill_between(x, 0, y, where=(x >= threshold), color='red', alpha=0.3,
                    label='Rejection Region')

    # Mark threshold
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
              label=f'Threshold: {threshold:.3f}')

    # Mark real value
    ax.axvline(real_metric, color='blue', linestyle='-', linewidth=3,
              label=f'Real Data: {real_metric:.3f}')

    # Calculate z-score
    z_score = (real_metric - surrogate_mean) / surrogate_std if surrogate_std > 0 else 0
    ax.text(0.02, 0.98, f'Z-score: {z_score:.2f}',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(metric_name)
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
