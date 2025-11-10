#!/usr/bin/env python3
"""
Unit tests for pgptracker.utils.profile_config.py

run with: pytest tests/unit/test_profile_config.py -v
"""

import pytest
from pathlib import Path
from dataclasses import is_dataclass

# Import all components to be tested
from pgptracker.utils.profile_config import (
    ProfilerConfig,
    get_config,
    set_config,
    configure_profiler,
    load_config_from_dict,
    use_preset,
    PRESET_DEBUG,
    PRESET_MINIMAL
)

# Fixture to ensure global config is reset after each test
@pytest.fixture(autouse=True)
def reset_global_config():
    """Ensures each test starts with a fresh, default config."""
    set_config(ProfilerConfig())
    yield
    set_config(ProfilerConfig())


class TestProfilerConfig:
    """Tests the ProfilerConfig dataclass itself."""

    def test_is_dataclass(self):
        """Check that it is a dataclass."""
        assert is_dataclass(ProfilerConfig)

    def test_post_init_validation(self):
        """Test that __post_init__ raises errors for invalid thresholds."""
        with pytest.raises(ValueError, match="memory_threshold_mb must be positive"):
            ProfilerConfig(memory_threshold_mb=0)
        
        with pytest.raises(ValueError, match="memory_threshold_mb must be positive"):
            ProfilerConfig(memory_threshold_mb=-100)
            
        with pytest.raises(ValueError, match="duration_threshold_s must be positive"):
            ProfilerConfig(duration_threshold_s=0)
            
        with pytest.raises(ValueError, match="duration_threshold_s must be positive"):
            ProfilerConfig(duration_threshold_s=-10)

    def test_post_init_exclude_modules_none(self):
        """Test that exclude_modules=None is converted to an empty set."""
        config = ProfilerConfig(exclude_modules=None)
        assert config.exclude_modules == set()
        
        config = ProfilerConfig(exclude_modules={"test"})
        assert config.exclude_modules == {"test"}

    @pytest.mark.parametrize("include, exclude, module, expected", [
        # Case 1: Default (all allowed)
        (None, None, "pgptracker.analysis", True),
        (None, None, "some.other.module", True),
        
        # Case 2: Exclude only
        (None, {"pgptracker.utils"}, "pgptracker.utils.profiler", False),
        (None, {"pgptracker.utils"}, "pgptracker.analysis", True),
        
        # Case 3: Include only
        ({"pgptracker.analysis"}, None, "pgptracker.analysis.stratify", True),
        ({"pgptracker.analysis"}, None, "pgptracker.utils", False),
        
        # Case 4: Exclude takes precedence over Include
        ({"pgptracker.analysis"}, {"pgptracker.analysis.stratify"}, "pgptracker.analysis.stratify", False),
        ({"pgptracker.analysis"}, {"pgptracker.utils"}, "pgptracker.analysis.stratify", True),
    ])
    def test_should_profile_module(self, include, exclude, module, expected):
        """Test the include/exclude logic for modules."""
        config = ProfilerConfig(
            include_modules=set(include) if include else None,
            exclude_modules=set(exclude) if exclude else None
        )
        assert config.should_profile_module(module) == expected

    def test_warning_checks(self):
        """Test the boolean logic of the warning threshold functions."""
        config = ProfilerConfig(memory_threshold_mb=100.0, duration_threshold_s=10.0)
        
        # Memory checks
        assert config.is_memory_warning(101.0) is True
        assert config.is_memory_warning(100.0) is False
        assert config.is_memory_warning(99.0) is False
        
        # Duration checks
        assert config.is_duration_warning(11.0) is True
        assert config.is_duration_warning(10.0) is False
        assert config.is_duration_warning(9.0) is False


class TestConfigManagement:
    """Tests the global state and helper functions."""

    def test_get_and_set_config(self):
        """Test that set_config correctly updates the global state retrieved by get_config."""
        assert get_config().enabled is False # Check default
        
        new_config = ProfilerConfig(enabled=True, show_top_n=99)
        set_config(new_config)
        
        retrieved_config = get_config()
        assert retrieved_config == new_config
        assert retrieved_config.enabled is True
        assert retrieved_config.show_top_n == 99

    def test_configure_profiler_helper(self):
        """Test the configure_profiler facade function."""
        configure_profiler(
            enabled=True,
            memory_threshold_mb=1234.0,
            exclude_modules={"test.module"}
        )
        
        config = get_config()
        assert config.enabled is True
        assert config.memory_threshold_mb == 1234.0
        assert config.exclude_modules == {"test.module"}
        assert config.track_dataframe_shapes is True # Check default

    def test_configure_profiler_exclude_none_logic(self):
        """Test the 'exclude_modules or set()' logic in configure_profiler."""
        configure_profiler(exclude_modules=None)
        config = get_config()
        assert config.exclude_modules == set()

    def test_use_preset(self):
        """Test applying presets."""
        use_preset('debug')
        config = get_config()
        assert config == PRESET_DEBUG
        assert config.memory_threshold_mb == 1000.0
        assert config.show_top_n == 10

        use_preset('minimal')
        config = get_config()
        assert config == PRESET_MINIMAL
        assert config.track_dataframe_shapes is False

    def test_use_invalid_preset_raises_error(self):
        """Test that an invalid preset name raises a ValueError."""
        with pytest.raises(ValueError, match="Invalid preset: bad_name"):
            use_preset('bad_name')

    def test_load_config_from_dict(self):
        """Test loading config from a dictionary, including type conversions."""
        config_dict = {
            'enabled': True,
            'output_dir': 'test/results',
            'output_tsv': 'test/report.tsv',
            'include_modules': ['pgptracker.analysis', 'pgptracker.utils'],
            'exclude_modules': ['pgptracker.tests'],
            'show_top_n': 7
        }
        
        config = load_config_from_dict(config_dict)
        
        assert config.enabled is True
        assert config.show_top_n == 7
        assert config.memory_threshold_mb == 5000.0 # Check default
        
        # Test type conversions
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path('test/results')
        
        assert isinstance(config.output_tsv, Path)
        assert config.output_tsv == Path('test/report.tsv')
        
        assert isinstance(config.include_modules, set)
        assert config.include_modules == {'pgptracker.analysis', 'pgptracker.utils'}
        
        assert isinstance(config.exclude_modules, set)
        assert config.exclude_modules == {'pgptracker.tests'}

    def test_load_config_from_dict_handles_granularity_pop(self):
        """
        Test that loading a dict with the old 'granularity' key 
        does not cause a crash.
        """
        config_dict = {
            'enabled': True,
            'granularity': 'line-by-line' # This key should be safely ignored
        }
        
        try:
            config = load_config_from_dict(config_dict)
            assert config.enabled is True
        except TypeError:
            pytest.fail("load_config_from_dict failed to pop 'granularity' key.")