#!/usr/bin/env python3
"""
Unit tests for the PGPTracker profiling utilities.

This test suite covers:
1.  ProfilerConfig: Module filtering (include/exclude) and validation.
2.  MemoryProfiler: Enable/disable state, profile capture, and summary generation.
3.  @profile_memory: Decorator logic, including skipping disabled/excluded functions.
4.  Critical Logic: Verifies that LazyFrames are NOT collected.
5.  ProfileReporter: TSV generation, pretty table printing, and warning logic.
run with: pytest tests/unit/test_profiler.py -v
"""

import pytest
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO
import sys
import contextlib

# Import all components to be tested
from pgptracker.utils.profile_config import (
    ProfilerConfig, 
    set_config, 
    get_config, 
    use_preset
)
from pgptracker.utils.profiler import (
    MemoryProfiler, 
    profile_memory, 
    get_profiling_summary, 
    FunctionProfile, 
    _extract_shapes
)
from pgptracker.utils.profile_reporter import (
    generate_tsv_report, 
    print_pretty_table, 
    print_top_consumers, 
    get_warnings
)

# --- Test Data and Mocks ---

# A dummy function to be decorated
@profile_memory
def dummy_function(df, lf):
    """A dummy function to profile."""
    pass

# Another dummy function in a specific "module"
@profile_memory
def specific_module_func():
    """A dummy function to test module filtering."""
    pass

# Set the __module__ attribute to simulate it being in a different file
specific_module_func.__module__ = "pgptracker.analysis.test"


# --- Fixtures ---

@pytest.fixture(autouse=True)
def setup_profiler():
    """
    Fixture to set up and tear down the profiler for each test.
    Ensures tests are isolated.
    """
    # Reset to default config and enable profiling
    set_config(ProfilerConfig(enabled=True))
    MemoryProfiler.enable()
    
    yield  # Run the test
    
    # Teardown: disable, clear, and reset config
    MemoryProfiler.disable()
    MemoryProfiler.clear_profiles()
    set_config(ProfilerConfig())


# --- Test Cases ---

class TestProfilerConfig:
    """Tests for profile_config.py"""

    def test_config_validation(self):
        """Test that invalid config values raise errors."""
        with pytest.raises(ValueError):
            ProfilerConfig(memory_threshold_mb=-10)
        with pytest.raises(ValueError):
            ProfilerConfig(duration_threshold_s=0)

    @pytest.mark.parametrize("include, exclude, module, expected", [
        (None, None, "pgptracker.analysis", True),  # Default: profile all
        (None, {"pgptracker.analysis"}, "pgptracker.analysis.stratify", False), # Exclude
        ({"pgptracker.analysis"}, None, "pgptracker.analysis.stratify", True), # Include
        ({"pgptracker.analysis"}, None, "pgptracker.utils", False), # Not in include list
        (None, {"pgptracker.utils"}, "pgptracker.analysis", True), # Not in exclude list
    ])
    def test_should_profile_module(self, include, exclude, module, expected):
        """Test the include/exclude logic for modules."""
        config = ProfilerConfig(
            include_modules=set(include) if include else None,
            exclude_modules=set(exclude) if exclude else set()
        )
        assert config.should_profile_module(module) == expected

    def test_use_preset(self):
        """Test applying a preset configuration."""
        set_config(ProfilerConfig()) # Reset
        use_preset('debug')
        config = get_config()
        assert config.memory_threshold_mb == 1000.0
        assert config.show_top_n == 10
        
        use_preset('minimal')
        config = get_config()
        assert config.track_dataframe_shapes is False
        assert config.show_pretty_table is False

    def test_use_invalid_preset(self):
        """Test that an invalid preset name raises an error."""
        with pytest.raises(ValueError, match="Invalid preset"):
            use_preset('nonexistent_preset')


class TestMemoryProfiler:
    """Tests for profiler.py"""

    @patch('tracemalloc.get_traced_memory')
    @patch('psutil.Process')
    def test_profile_memory_decorator(self, mock_psutil_process, mock_get_traced_memory):
        """Test that the @profile_memory decorator captures data correctly."""
        # Setup mocks
        # Mock 'before' (current, peak) and 'after' (current, peak) memory
        mock_get_traced_memory.side_effect = [
            (10 * 1024**2, 10 * 1024**2),  # Before call
            (50 * 1024**2, 60 * 1024**2)   # After call
        ]
        mock_psutil_process.return_value.memory_info.return_value.rss = 100 * 1024**2

        # Run the profiled function
        dummy_function(None, None)

        profiles = MemoryProfiler.get_profiles()
        assert len(profiles) == 1
        
        p = profiles[0]
        assert p.function_name == "dummy_function"
        assert p.peak_mb == 60.0  # Peak is the high-water mark (60)
        assert p.diff_from_baseline_mb == 40.0 # Diff is current_after - current_before
        assert p.process_total_mb == 100.0

    def test_profiler_disabled(self):
        """Test that no profiles are collected when the profiler is disabled."""
        MemoryProfiler.disable()
        dummy_function(None, None)
        assert len(MemoryProfiler.get_profiles()) == 0

    def test_profiler_respects_exclude_config(self):
        """Test that the decorator skips functions from excluded modules."""
        # Exclude the module we manually set for specific_module_func
        config = ProfilerConfig(enabled=True, exclude_modules={"tests.unit.test_profiler"})
        set_config(config)
        
        specific_module_func()
        
        assert len(MemoryProfiler.get_profiles()) == 0

    def test_profiler_respects_include_config(self):
        """Test that the decorator skips functions not in the include list."""
        config = ProfilerConfig(enabled=True, include_modules={"pgptracker.utils"})
        set_config(config)
        
        # This function is in 'pgptracker.analysis.test', so it should be skipped
        specific_module_func()
        
        assert len(MemoryProfiler.get_profiles()) == 0

    def test_critical_extract_shapes_no_collect(self):
        """
        CRITICAL: Ensure _extract_shapes NEVER calls .collect() on a LazyFrame.
        """
        mock_df = MagicMock(spec=pl.DataFrame)
        mock_df.height = 100
        mock_df.width = 10
        
        mock_lf = MagicMock(spec=pl.LazyFrame)
        
        # Test with (df, lf) and no kwargs
        result = _extract_shapes((mock_df, mock_lf), {})
        assert result == "100×10, LazyFrame"
        
        # Test with kwargs
        result = _extract_shapes((), {"key_df": mock_df, "key_lf": mock_lf})
        assert result == "100×10, LazyFrame"

        # The most important assertion: .collect() was never called
        mock_lf.collect.assert_not_called()

    @patch('tracemalloc.get_traced_memory', side_effect=[
        (10*1024**2, 10*1024**2), (20*1024**2, 20*1024**2), # Call 1
        (20*1024**2, 20*1024**2), (50*1024**2, 50*1024**2)  # Call 2
    ])
    @patch('psutil.Process')
    def test_get_profiling_summary(self, mock_psutil, mock_tracemalloc):
        """Test the summary generation."""
        dummy_function(None, None) # Call 1
        dummy_function(None, None) # Call 2

        summary = get_profiling_summary()
        
        assert summary['total_functions'] == 2
        # The max peak should be the high-water mark from the last call
        assert summary['max_peak_mb'] == 50.0

    def test_get_profiling_summary_no_data(self):
        """Test summary when no profiles are collected."""
        MemoryProfiler.clear_profiles()
        summary = get_profiling_summary()
        assert summary['total_functions'] == 0
        assert summary['max_peak_mb'] == 0.0


class TestProfileReporter:
    """Tests for profile_reporter.py"""

    @patch('tracemalloc.get_traced_memory', return_value=(10*1024**2, 20*1024**2))
    @patch('psutil.Process')
    def test_generate_tsv_report(self, mock_psutil, mock_tracemalloc, tmp_path):
        """Test that a TSV report is correctly generated."""
        # Create a profile entry
        dummy_function(None, None)
        
        output_file = tmp_path / "test_report.tsv"
        generate_tsv_report(output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        
        assert "function\tmodule\ttimestamp" in content # Header
        assert "dummy_function" in content # Data
        assert "avg_mb" not in content # Ensure old column is gone

    def test_generate_tsv_report_no_data(self, tmp_path):
        """Test that report generation fails if no profiles exist."""
        MemoryProfiler.clear_profiles()
        output_file = tmp_path / "test_report.tsv"
        
        with pytest.raises(ValueError, match="No profiling data available"):
            generate_tsv_report(output_file)

    @patch('tracemalloc.get_traced_memory', return_value=(10*1024**2, 20*1024**2))
    @patch('psutil.Process')
    def test_print_pretty_table(self, mock_psutil, mock_tracemalloc):
        """Test that the pretty table prints to stdout."""
        mock_psutil.return_value.memory_info.return_value.rss = 100 * 1024**2
        dummy_function(None, None)
        
        f = StringIO()
        with contextlib.redirect_stdout(f):
            print_pretty_table()
        
        output = f.getvalue()
        assert "MEMORY PROFILING RESULTS" in output
        assert "dummy_function" in output
        assert "Avg MB" not in output # Ensure old column is gone

    @patch('tracemalloc.get_traced_memory', return_value=(10*1024**2, 20*1024**2))
    @patch('psutil.Process')
    def test_print_top_consumers(self, mock_psutil, mock_tracemalloc):
        """Test that top consumers prints to stdout."""
        dummy_function(None, None)
        
        f = StringIO()
        with contextlib.redirect_stdout(f):
            print_top_consumers(n=5)
            
        output = f.getvalue()
        assert "Top 1 Memory Consumers" in output
        assert "dummy_function" in output

    def test_get_warnings(self):
        """Test the warning generation logic."""
        config = get_config()
        config.memory_threshold_mb = 10.0  # 10 MB
        config.duration_threshold_s = 1.0  # 1 Second
        set_config(config)
        
        # Create mock profiles
        p_ok = MagicMock(spec=FunctionProfile)
        p_ok.peak_mb = 5.0
        p_ok.duration_seconds = 0.5
        
        p_mem = MagicMock(spec=FunctionProfile)
        p_mem.peak_mb = 20.0
        p_mem.duration_seconds = 0.5
        
        p_time = MagicMock(spec=FunctionProfile)
        p_time.peak_mb = 5.0
        p_time.duration_seconds = 2.0
        
        p_both = MagicMock(spec=FunctionProfile)
        p_both.peak_mb = 20.0
        p_both.duration_seconds = 2.0
        
        profiles = [p_ok, p_mem, p_time, p_both]
        warnings = get_warnings(profiles)
        
        assert len(warnings) == 3
        assert warnings[0][0] == p_mem and warnings[0][1] == 'memory'
        assert warnings[1][0] == p_time and warnings[1][1] == 'duration'
        assert warnings[2][0] == p_both and warnings[2][1] == 'memory & duration'