#!/usr/bin/env python3
"""
Configuration and settings for memory profiler.

Defines thresholds and profiling behavior.

Author: Vivian Mello
"""

from dataclasses import dataclass
from typing import Optional, Set
from pathlib import Path


@dataclass
class ProfilerConfig:
    """
    Configuration for memory profiler behavior.
    
    Attributes:
        enabled: Whether profiling is active
        output_tsv: Path to save TSV report (None = auto-generate)
        output_dir: Directory for profiling outputs
        memory_threshold_mb: Warn if function exceeds this peak RAM
        duration_threshold_s: Warn if function exceeds this duration
        track_dataframe_shapes: Capture DataFrame dimensions
        track_system_info: Capture system RAM availability
        include_modules: Only profile functions from these modules (None = all)
        exclude_modules: Skip profiling these modules
        show_pretty_table: Print table to terminal after profiling
        show_top_n: Show top N consumers in summary (0 = disable)
    """
    enabled: bool = False
    output_tsv: Optional[Path] = None
    output_dir: Path = Path(".")
    memory_threshold_mb: float = 5000.0  # Warn if >5GB
    duration_threshold_s: float = 300.0   # Warn if >5min
    track_dataframe_shapes: bool = True
    track_system_info: bool = True
    include_modules: Optional[Set[str]] = None
    exclude_modules: Optional[Set[str]] = None
    show_pretty_table: bool = True
    show_top_n: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.memory_threshold_mb <= 0:
            raise ValueError("memory_threshold_mb must be positive")
        
        if self.duration_threshold_s <= 0:
            raise ValueError("duration_threshold_s must be positive")
        
        if self.exclude_modules is None:
            self.exclude_modules = set()
    
    def should_profile_module(self, module_name: str) -> bool:
        """
        Check if module should be profiled based on include/exclude rules.
        
        Args:
            module_name: Full module path (e.g., 'pgptracker.analysis.stratify')
        
        Returns:
            True if module should be profiled
        """
        # Check exclude list first
        if self.exclude_modules:
            for excluded in self.exclude_modules:
                if module_name.startswith(excluded):
                    return False
        
        # Check include list (if specified)
        if self.include_modules is not None:
            for included in self.include_modules:
                if module_name.startswith(included):
                    return True
            return False  # Not in include list
        
        # Default: profile everything not excluded
        return True
    
    def is_memory_warning(self, peak_mb: float) -> bool:
        """Check if peak memory exceeds warning threshold."""
        return peak_mb > self.memory_threshold_mb
    
    def is_duration_warning(self, duration_s: float) -> bool:
        """Check if duration exceeds warning threshold."""
        return duration_s > self.duration_threshold_s


# Global configuration instance
_config = ProfilerConfig()


def get_config() -> ProfilerConfig:
    """
    Get global profiler configuration.
    
    Returns:
        Current ProfilerConfig instance
    """
    return _config


def set_config(config: ProfilerConfig):
    """
    Set global profiler configuration.
    
    Args:
        config: New ProfilerConfig instance
    """
    global _config
    _config = config


def configure_profiler(
    enabled: bool = False,
    output_tsv: Optional[Path] = None,
    output_dir: Path = Path("."),
    memory_threshold_mb: float = 5000.0,
    duration_threshold_s: float = 300.0,
    track_dataframe_shapes: bool = True,
    track_system_info: bool = True,
    include_modules: Optional[Set[str]] = None,
    exclude_modules: Optional[Set[str]] = None,
    show_pretty_table: bool = True,
    show_top_n: int = 5
):
    """
    Configure profiler settings.
    
    Args:
        enabled: Enable profiling
        output_tsv: Path to TSV output file
        output_dir: Directory for outputs
        memory_threshold_mb: Warning threshold for peak RAM
        duration_threshold_s: Warning threshold for duration
        track_dataframe_shapes: Capture DataFrame dimensions
        track_system_info: Capture system RAM info
        include_modules: Only profile these modules (None = all)
        exclude_modules: Skip these modules
        show_pretty_table: Print table after profiling
        show_top_n: Show top N consumers (0 = disable)
    
    Example:
        >>> from pgptracker.utils.profile_config import configure_profiler
        >>> configure_profiler(
        ...     enabled=True,
        ...     output_dir=Path("results"),
        ...     include_modules={'pgptracker.analysis'},
        ...     memory_threshold_mb=2000.0
        ... )
    """
    config = ProfilerConfig(
        enabled=enabled,
        output_tsv=output_tsv,
        output_dir=output_dir,
        memory_threshold_mb=memory_threshold_mb,
        duration_threshold_s=duration_threshold_s,
        track_dataframe_shapes=track_dataframe_shapes,
        track_system_info=track_system_info,
        include_modules=include_modules,
        exclude_modules=exclude_modules or set(),
        show_pretty_table=show_pretty_table,
        show_top_n=show_top_n
    )
    set_config(config)


def load_config_from_dict(config_dict: dict) -> ProfilerConfig:
    """
    Load configuration from dictionary (e.g., from YAML).
    
    Args:
        config_dict: Dictionary with configuration keys
    
    Returns:
        ProfilerConfig instance
    
    Example:
        >>> config_dict = {
        ...     'enabled': True,
        ...     'output_dir': 'results'
        ... }
        >>> config = load_config_from_dict(config_dict)
    """
    # Convert string paths to Path objects
    if 'output_tsv' in config_dict and config_dict['output_tsv']:
        config_dict['output_tsv'] = Path(config_dict['output_tsv'])
    
    if 'output_dir' in config_dict:
        config_dict['output_dir'] = Path(config_dict['output_dir'])
    
    # Convert lists to sets
    if 'include_modules' in config_dict and config_dict['include_modules']:
        config_dict['include_modules'] = set(config_dict['include_modules'])
    
    if 'exclude_modules' in config_dict and config_dict['exclude_modules']:
        config_dict['exclude_modules'] = set(config_dict['exclude_modules'])
    
    # Remove granularity if it exists in the dict, as it's no longer used
    config_dict.pop('granularity', None)
        
    return ProfilerConfig(**config_dict)


# Preset configurations
PRESET_PRODUCTION = ProfilerConfig(
    enabled=True,
    memory_threshold_mb=5000.0,
    duration_threshold_s=300.0,
    track_dataframe_shapes=True,
    track_system_info=True,
    show_pretty_table=True,
    show_top_n=5
)

PRESET_DEBUG = ProfilerConfig(
    enabled=True,
    memory_threshold_mb=1000.0,  # Lower threshold for debugging
    duration_threshold_s=60.0,    # Lower threshold
    track_dataframe_shapes=True,
    track_system_info=True,
    show_pretty_table=True,
    show_top_n=10  # Show more consumers
)

PRESET_MINIMAL = ProfilerConfig(
    enabled=True,
    memory_threshold_mb=10000.0,  # High threshold (only extreme cases)
    duration_threshold_s=600.0,
    track_dataframe_shapes=False,  # Skip shapes for performance
    track_system_info=False,
    show_pretty_table=False,       # Only save TSV
    show_top_n=0                   # No summary
)


def use_preset(preset_name: str):
    """
    Apply a preset configuration.
    
    Args:
        preset_name: Name of preset ('production', 'debug', 'minimal')
    
    Raises:
        ValueError: If preset name is invalid
    
    Example:
        >>> from pgptracker.utils.profile_config import use_preset
        >>> use_preset('debug')
    """
    presets = {
        'production': PRESET_PRODUCTION,
        'debug': PRESET_DEBUG,
        'minimal': PRESET_MINIMAL
    }
    
    if preset_name not in presets:
        raise ValueError(
            f"Invalid preset: {preset_name}. "
            f"Available presets: {list(presets.keys())}"
        )
    
    set_config(presets[preset_name])