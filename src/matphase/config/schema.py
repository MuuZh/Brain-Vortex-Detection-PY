"""
Configuration schema using Pydantic for type validation.

This module defines the configuration structure for MatPhase,
including preprocessing, detection, analysis, and compute settings.
"""

import math
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    """File paths configuration."""

    data_dir: Optional[Path] = Field(None, description="Input data directory")
    output_dir: Path = Field(
        Path("./output"), description="Output directory for results"
    )
    cifti_file: Optional[Path] = Field(
        None, description="Path to CIFTI file (absolute or relative to data_dir)"
    )

    # Surface geometry files - separate paths for left and right hemispheres
    surface_left: Optional[Path] = Field(
        None, description="Path to left hemisphere surface file (.surf.gii)"
    )
    surface_right: Optional[Path] = Field(
        None, description="Path to right hemisphere surface file (.surf.gii)"
    )

    # Legacy field for backward compatibility (deprecated)
    geometry_file: Optional[Path] = Field(
        None, description="[Deprecated] Use surface_left/surface_right instead"
    )

    # Parcellation template files
    parcellation_left: Optional[Path] = Field(
        None, description="Path to left hemisphere parcellation template (.npy)"
    )
    parcellation_right: Optional[Path] = Field(
        None, description="Path to right hemisphere parcellation template (.npy)"
    )


class PreprocessingConfig(BaseModel):
    """Preprocessing parameters."""

    hemisphere: Literal["left", "right", "both"] = Field(
        "left", description="Which hemisphere(s) to process"
    )
    sigma_scale: list[float] = Field(
        [29.35, 14.93], description="Gaussian smoothing scales in mm"
    )
    downsample_rate: int = Field(
        2, ge=1, description="Spatial downsampling factor"
    )
    interpolation_method: Literal["cubic", "linear", "nearest", "tri_linear"] = Field(
        "cubic",
        description="Interpolation kernel: griddata modes or triangulation-based 'tri_linear' (requires faces)",
    )
    interpolation_coordinate_system: Literal["physical", "positive"] = Field(
        "physical",
        description="Coordinate system for interpolation: physical (MATLAB parity) or shift to positive axis before interpolation.",
    )

    # Coordinate grids (per hemisphere)
    left_x_coord_min: float = Field(-250, description="Left hemisphere X minimum")
    left_x_coord_max: float = Field(250, description="Left hemisphere X maximum")
    left_y_coord_min: float = Field(-150, description="Left hemisphere Y minimum")
    left_y_coord_max: float = Field(200, description="Left hemisphere Y maximum")
    right_x_coord_min: float = Field(-270, description="Right hemisphere X minimum")
    right_x_coord_max: float = Field(230, description="Right hemisphere X maximum")
    right_y_coord_min: float = Field(-180, description="Right hemisphere Y minimum")
    right_y_coord_max: float = Field(170, description="Right hemisphere Y maximum")

    # Temporal settings
    temporal_sampling_rate: float = Field(
        1.3889, gt=0, description="Temporal sampling rate in Hz"
    )

    # Filtering
    filter_low_freq: float = Field(
        0.01, gt=0, description="Bandpass filter lower cutoff (Hz)"
    )
    filter_high_freq: float = Field(
        0.1, gt=0, description="Bandpass filter upper cutoff (Hz)"
    )
    filter_order: int = Field(
        4, ge=1, description="Butterworth filter order"
    )
    filter_method: Literal["sosfiltfilt", "filtfilt"] = Field(
        "sosfiltfilt",
        description="Filter application method: 'sosfiltfilt' (recommended) or 'filtfilt'"
    )
    filter_padtype: Literal["odd", "even", "constant"] = Field(
        "odd",
        description="Padding type for filtfilt: 'odd' (default), 'even', or 'constant'"
    )
    filter_precision: Literal["float32", "float64"] = Field(
        "float32",
        description="Numerical precision for filtering: 'float32' or 'float64'"
    )

    # Phase extraction method selection
    phase_extraction_method: Literal["hilbert", "generalized_phase"] = Field(
        "hilbert",
        description="Phase extraction method: 'hilbert' or 'generalized_phase'"
    )

    # Hilbert method options
    return_analytic_signal: bool = Field(
        True,
        description="Return complex analytic signal (Hilbert method)"
    )

    # Generalized Phase (GP) method options (future implementation)
    gp_filter_range: Optional[tuple[float, float]] = Field(
        None,
        description="GP method frequency range (Hz), None=use bandpass range"
    )
    gp_smoothing_window: Optional[int] = Field(
        None,
        description="GP phase smoothing window (samples), None=no smoothing"
    )
    gp_phase_correction_threshold: float = Field(
        0.1,
        description="GP negative frequency detection threshold"
    )
    gp_neg_freq_extension: int = Field(
        3,
        ge=1,
        description="GP negative frequency mask extension factor (>=1)"
    )
    gp_return_inst_freq: bool = Field(
        True,
        description="GP method: return instantaneous frequency"
    )
    gp_return_neg_freq_mask: bool = Field(
        False,
        description="GP method: return negative frequency correction markers"
    )

    # Common temporal options
    temporal_demean: bool = Field(
        True,
        description="Demean each channel before filtering"
    )
    show_temporal_progress: bool = Field(
        True,
        description="Show progress bar for temporal filtering"
    )

    # Spatial filtering options
    sigma_scales: list[float] = Field(
        [29.35, 14.93],
        description="Gaussian sigma scales in physical coordinates (adjusted by downsample_rate)"
    )
    spatial_filter_mode: Literal["dog", "lowpass"] = Field(
        "dog",
        description="Spatial filtering mode: 'dog' (difference-of-Gaussians) or 'lowpass'"
    )
    show_spatial_progress: bool = Field(
        True,
        description="Show progress bar for spatial filtering"
    )

    # Surrogate generation options
    n_surrogates: int = Field(
        1,
        ge=1,
        description="Number of surrogate realizations to generate"
    )
    surrogate_random_seed: Optional[int] = Field(
        None,
        description="Random seed for surrogate generation (None=non-reproducible)"
    )
    surrogate_phase_mode: Literal["replace", "add"] = Field(
        "replace",
        description="Phase randomization mode: 'replace' (standard) or 'add' (MATLAB)"
    )

    # Legacy smoothing options (deprecated, use spatial_filter_mode)
    use_surrogate_filtering: bool = Field(
        False, description="[DEPRECATED] Use surrogate-based spatial filtering"
    )
    use_gaussian_smoothing: bool = Field(
        True, description="[DEPRECATED] Use Gaussian spatial smoothing"
    )

    @field_validator("filter_high_freq")
    @classmethod
    def validate_freq_range(cls, v: float, info) -> float:
        """Ensure high frequency is greater than low frequency."""
        if "filter_low_freq" in info.data and v <= info.data["filter_low_freq"]:
            raise ValueError(
                "filter_high_freq must be greater than filter_low_freq"
            )
        return v

    # Backward-compatible aliases for legacy coordinate names
    @property
    def x_coord_min(self) -> float:
        return self.left_x_coord_min

    @property
    def x_coord_max(self) -> float:
        return self.left_x_coord_max

    @property
    def y_coord_min(self) -> float:
        return self.left_y_coord_min

    @property
    def y_coord_max(self) -> float:
        return self.left_y_coord_max


class DetectionConfig(BaseModel):
    """Pattern detection parameters."""

    min_pattern_duration: int = Field(
        1, ge=1, description="Minimum pattern duration in frames"
    )
    min_pattern_size: int = Field(
        3, ge=1, description="Minimum spatial extent in pixels/nodes"
    )
    connectivity: Literal[6, 18, 26] = Field(
        6, description="Connectivity for 3D pattern detection"
    )
    use_weighted_centroids: bool = Field(
        True, description="Use amplitude-weighted centroids"
    )

    # Threshold parameters for spiral detection
    curl_threshold: Optional[float] = Field(
        1.0,
        description="Minimum curl magnitude for spiral detection (MATLAB default: 1.0, None=auto from surrogates)"
    )
    expansion_threshold: Optional[float] = Field(
        1.0,
        description="Maximum expansion (divergence) for spiral detection (MATLAB default: 1.0)"
    )
    phase_coherence_threshold: Optional[float] = Field(
        None,
        description="Minimum phase gradient coherence (legacy). Set to None to disable gradient-based filtering."
    )
    phase_difference_threshold: Optional[float] = Field(
        0.5235987756,
        description="Maximum allowed |phase_raw - phase_spatial| difference (radians) when filtering voxels."
    )
    apply_phase_difference_mask: bool = Field(
        False,
        description="DEPRECATED: kept for backward compatibility; phase-difference mask is always applied only to compatibility filtering, not expansion."
    )
    rotation_mode: Literal["both", "ccw", "cw"] = Field(
        "both",
        description="Rotation selection: 'ccw' (positive curl), 'cw' (negative curl), or 'both'"
    )
    enable_spiral_expansion: bool = Field(
        True,
        description="Enable MATLAB-style spiral expansion using radius and angle-difference constraints."
    )
    angle_window_center: float = Field(
        float(math.pi / 2),
        description="Ideal angle difference (radians) between center-origin vectors and phase vectors (default: 90 degrees)."
    )
    angle_window_half_width: float = Field(
        0.7853981633974483,
        description="Half-width (radians) around the ideal angle that remains compatible (default: +/-45 degrees)."
    )
    expansion_radius_min: float = Field(
        2.0,
        ge=0.0,
        description="Minimum radius (in grid units) used when expanding spiral footprints."
    )
    expansion_radius_max: float = Field(
        100.0,
        ge=0.0,
        description="Maximum radius (in grid units) allowed for spiral expansion."
    )
    expansion_radius_step: float = Field(
        0.5,
        gt=0.0,
        description="Increment (in grid units) between successive radius tests during expansion."
    )
    center_patch_radius: int = Field(
        1,
        ge=0,
        description="Radius (pixels) of the always-on patch around a spiral center during expansion (default 1 -> 3x3)."
    )

    # Surrogate-based thresholding
    use_surrogate_thresholds: bool = Field(
        False,
        description="Compute thresholds from surrogate distribution (overrides fixed thresholds)"
    )
    surrogate_percentile: float = Field(
        95.0,
        ge=50.0,
        le=99.9,
        description="Percentile for surrogate threshold (95.0 = p<0.05, 99.0 = p<0.01)"
    )
    n_surrogates_threshold: int = Field(
        1,
        ge=0,
        description="Number of surrogates for threshold estimation"
    )

    # Threshold application mode
    use_absolute_curl: bool = Field(
        True,
        description="Use absolute value of curl (True=both rotations, False=signed only)"
    )
    threshold_fill_value: float = Field(
        0.0,
        description="Value for sub-threshold regions (0.0 or NaN)"
    )


class ContrastConfig(BaseModel):
    """Contrast analysis options."""

    design_file: Optional[Path] = Field(
        None,
        description="Optional CSV/TSV input describing task blocks",
    )
    baseline_condition: str = Field(
        "rest",
        description="Default denominator condition for quick contrasts",
    )
    minimum_frames: int = Field(
        1,
        ge=1,
        description="Minimum number of frames required per condition",
    )
    percent_change_epsilon: float = Field(
        1e-6,
        gt=0,
        description="Stabilizer used when computing percent-change maps",
    )


class ClassificationConfig(BaseModel):
    """Pattern classification options."""

    label_file: Optional[Path] = Field(
        None,
        description="Optional CSV linking pattern IDs to behavioral labels",
    )
    feature_columns: List[str] = Field(
        default_factory=lambda: [
            "duration",
            "mean_size",
            "mean_power",
            "radius_estimate",
            "mean_speed",
            "trajectory_angle",
        ],
        description="Default feature columns to feed into classifiers",
    )
    cv_strategy: Literal["stratified_kfold", "kfold", "leave_one_out"] = Field(
        "stratified_kfold",
        description="Cross-validation strategy",
    )
    cross_validation_folds: int = Field(
        5,
        ge=2,
        description="Number of folds for k-fold style validation",
    )
    random_state: Optional[int] = Field(
        42,
        description="Optional RNG seed for deterministic splits",
    )
    wrap_angles: bool = Field(
        True,
        description="Wrap *_angle features to [-pi, pi]",
    )
    min_class_samples: int = Field(
        2,
        ge=1,
        description="Minimum samples per class required before training",
    )


class ReportConfig(BaseModel):
    """Reporting output options for analysis pipelines."""

    output_root: Path = Field(
        Path("tests/analysis/output"),
        description="Root directory where analysis reports are stored",
    )
    prefix: str = Field(
        "report",
        description="Folder prefix for timestamped report runs",
    )
    formats: List[Literal["markdown", "html"]] = Field(
        default_factory=lambda: ["markdown", "html"],
        description="Report formats to emit for each subject",
    )


class AnalysisCLIConfig(BaseModel):
    """Default inputs for the analysis CLI subcommand."""

    subjects_manifest: Optional[Path] = Field(
        None,
        description="Optional default manifest with bundle metadata",
    )
    default_bundles: List[Path] = Field(
        default_factory=list,
        description="Bundle directories processed when no manifest or CLI bundles are provided",
    )


class AnalysisConfig(BaseModel):
    """Analysis parameters."""

    task_onset_buffer: int = Field(
        0, ge=0, description="Frames before task onset to include"
    )
    task_offset_buffer: int = Field(
        0, ge=0, description="Frames after task offset to include"
    )
    significance_level: float = Field(
        0.05, gt=0, lt=1, description="Alpha for statistical tests"
    )
    permutation_count: int = Field(
        1000, ge=1, description="Number of permutations for null distribution"
    )
    classifier_type: Literal["svm", "random_forest", "logistic", "nearest_centroid"] = Field(
        "nearest_centroid", description="Preferred classifier type"
    )
    cross_validation_folds: int = Field(
        5, ge=2, description="K-folds for cross-validation"
    )
    contrast: ContrastConfig = Field(
        default_factory=ContrastConfig, description="Task contrast settings"
    )
    classification: ClassificationConfig = Field(
        default_factory=ClassificationConfig, description="Pattern classification settings"
    )
    report: ReportConfig = Field(
        default_factory=ReportConfig,
        description="Reporting/output settings for analysis pipelines",
    )
    cli: AnalysisCLIConfig = Field(
        default_factory=AnalysisCLIConfig,
        description="Default CLI inputs for analysis subcommands",
    )


class ComputeConfig(BaseModel):
    """Computational settings."""

    use_gpu: bool = Field(False, description="Enable GPU acceleration")
    gpu_device: int = Field(0, ge=0, description="GPU device ID")
    n_jobs: int = Field(-1, description="Number of parallel jobs")
    chunk_size: Optional[int] = Field(
        None, ge=1, description="Process data in chunks"
    )


class OutputConfig(BaseModel):
    """Output settings."""

    save_preprocessed: bool = Field(True, description="Save preprocessed data")
    save_patterns: bool = Field(True, description="Save detected patterns")
    save_figures: bool = Field(True, description="Save figures")
    save_phase_cube: bool = Field(
        False, description="Persist Hilbert phase cube to bundle directory"
    )
    phase_cube_filename: str = Field(
        "phase_cube.npy", description="Filename for saved phase cube (.npy)"
    )
    figure_format: Literal["png", "pdf", "svg"] = Field(
        "png", description="Figure output format"
    )
    figure_dpi: int = Field(300, ge=72, description="DPI for raster outputs")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level"
    )
    log_file: Optional[Path] = Field(None, description="Log file path")

class _HemisphereView:
    """
    Compatibility view to expose hemisphere settings via config.hemisphere.

    This preserves legacy access patterns after moving hemisphere settings
    into preprocessing.
    """

    def __init__(self, preprocessing: PreprocessingConfig):
        self._preprocessing = preprocessing

    @property
    def side(self) -> Literal["left", "right", "both"]:
        return self._preprocessing.hemisphere

    @side.setter
    def side(self, value: Literal["left", "right", "both"]) -> None:
        self._preprocessing.hemisphere = value

    @property
    def left_x_coord_min(self) -> float:
        return self._preprocessing.left_x_coord_min

    @property
    def left_x_coord_max(self) -> float:
        return self._preprocessing.left_x_coord_max

    @property
    def left_y_coord_min(self) -> float:
        return self._preprocessing.left_y_coord_min

    @property
    def left_y_coord_max(self) -> float:
        return self._preprocessing.left_y_coord_max

    @property
    def right_x_coord_min(self) -> float:
        return self._preprocessing.right_x_coord_min

    @property
    def right_x_coord_max(self) -> float:
        return self._preprocessing.right_x_coord_max

    @property
    def right_y_coord_min(self) -> float:
        return self._preprocessing.right_y_coord_min

    @property
    def right_y_coord_max(self) -> float:
        return self._preprocessing.right_y_coord_max


class MatPhaseConfig(BaseModel):
    """Root configuration for MatPhase."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    model_config = {"frozen": False, "validate_assignment": True}

    @property
    def hemisphere(self) -> _HemisphereView:
        return _HemisphereView(self.preprocessing)
