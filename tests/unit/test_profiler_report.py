#!/usr/bin/env python3
"""
Unit tests for pgptracker.utils.profile_reporter.py

run with: pytest tests/unit/test_profiler_report.py -v
"""

import pytest
import contextlib
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

# Import all functions to be tested
from pgptracker.utils.profile_reporter import (
    generate_tsv_report,
    print_pretty_table,
    get_top_consumers,
    print_top_consumers,
    get_warnings,
    print_warnings
)

# Import dependencies that we need to mock
from pgptracker.utils.profiler import FunctionProfile
from pgptracker.utils.profile_config import ProfilerConfig


@pytest.fixture
def mock_profiles():
    """
    Provides a list of mock FunctionProfile objects for testing.
    The list is intentionally unsorted to test sorting logic.
    """
    # Profile 1: Low mem, High time
    p_time = MagicMock(spec=FunctionProfile)
    p_time.function_name = "func_high_time"
    p_time.module_name = "module.b"
    p_time.timestamp = "2025-11-09 10:01:00"
    p_time.duration_seconds = 50.0
    p_time.peak_mb = 10.0
    p_time.diff_from_baseline_mb = 5.0
    p_time.process_total_mb = 20.0
    p_time.system_available_mb = 400.0
    p_time.input_shapes = "LazyFrame"
    p_time.output_shapes = "LazyFrame"
    p_time.calls = 1

    # Profile 2: High mem, Low time
    p_mem = MagicMock(spec=FunctionProfile)
    p_mem.function_name = "func_high_mem"
    p_mem.module_name = "module.a"
    p_mem.timestamp = "2025-11-09 10:00:00"
    p_mem.duration_seconds = 0.5
    p_mem.peak_mb = 100.0
    p_mem.diff_from_baseline_mb = 90.0
    p_mem.process_total_mb = 110.0
    p_mem.system_available_mb = 500.0
    p_mem.input_shapes = "100x10"
    p_mem.output_shapes = "100x5"
    p_mem.calls = 1
    
    # Profile 3: OK
    p_ok = MagicMock(spec=FunctionProfile)
    p_ok.function_name = "func_ok"
    p_ok.module_name = "module.c"
    p_ok.timestamp = "2025-11-09 10:02:00"
    p_ok.duration_seconds = 1.0
    p_ok.peak_mb = 5.0
    p_ok.diff_from_baseline_mb = 1.0
    p_ok.process_total_mb = 10.0
    p_ok.system_available_mb = 600.0
    p_ok.input_shapes = ""
    p_ok.output_shapes = ""
    p_ok.calls = 1

    return [p_time, p_mem, p_ok]


@pytest.fixture
def mock_config():
    """Provides a mock config object with specific thresholds for testing warnings."""
    config = MagicMock(spec=ProfilerConfig)
    config.memory_threshold_mb = 50.0  # 50 MB threshold
    config.duration_threshold_s = 30.0 # 30 sec threshold
    
    # Define the behavior of the warning check methods
    config.is_memory_warning.side_effect = lambda mb: mb > 50.0
    config.is_duration_warning.side_effect = lambda s: s > 30.0
    return config


def test_generate_tsv_report_success(mocker, mock_profiles, tmp_path):
    """Test that a TSV report is correctly generated and sorted."""
    # Mock the dependency on MemoryProfiler
    mocker.patch('pgptracker.utils.profiler.MemoryProfiler.get_profiles', return_value=mock_profiles)
    
    output_file = tmp_path / "report.tsv"
    generate_tsv_report(output_file)

    assert output_file.exists()
    content = output_file.read_text().strip().split('\n')
    
    # Check header
    assert content[0].startswith("function\tmodule\ttimestamp\tduration_s\tpeak_mb")
    assert "avg_mb" not in content[0] # Verify old column is gone

    # Check data rows (should be 3 data rows + 1 header)
    assert len(content) == 4
    
    # Check sorting (p_mem=100MB, p_time=10MB, p_ok=5MB)
    assert "func_high_mem" in content[1] # 100MB
    assert "func_high_time" in content[2] # 10MB
    assert "func_ok" in content[3] # 5MB


def test_generate_tsv_report_no_data(mocker, tmp_path):
    """Test that report generation fails if no profiles exist."""
    mocker.patch('pgptracker.utils.profiler.MemoryProfiler.get_profiles', return_value=[])
    
    output_file = tmp_path / "report.tsv"
    with pytest.raises(ValueError, match="No profiling data available"):
        generate_tsv_report(output_file)


def test_print_pretty_table_success(mocker, mock_profiles, mock_config):
    """Test that the pretty table prints correct info to stdout."""
    # Mock all external dependencies
    mocker.patch('pgptracker.utils.profiler.MemoryProfiler.get_profiles', return_value=mock_profiles)
    mocker.patch('pgptracker.utils.profile_reporter.get_config', return_value=mock_config)
    
    # Mock get_profiling_summary to return predictable data
    mock_summary = {'total_functions': 3, 'total_duration': 51.5, 'max_peak_mb': 100.0}
    mocker.patch('pgptracker.utils.profile_reporter.get_profiling_summary', return_value=mock_summary)
    
    f = StringIO()
    with contextlib.redirect_stdout(f):
        print_pretty_table()
    output = f.getvalue()

    # Check for key elements
    assert "MEMORY PROFILING RESULTS" in output
    assert "func_high_mem" in output
    assert "func_ok" in output
    assert "Avg MB" not in output # Verify old column is gone
    
    # Check summary section
    assert "Maximum peak memory: 100.00 MB" in output
    
    # Check that warnings are printed (p_mem and p_time should trigger)
    assert "WARNING: Functions exceeding thresholds" in output
    assert "func_high_time" in output
    assert "Duration: 50.00s" in output
    assert "func_high_mem" in output
    assert "Memory: 100.00 MB" in output
    # Split output into main table and warning sections
    # The split string must exactly match the print statement in the reporter
    output_parts = output.split("\nWARNING: Functions exceeding thresholds")
    main_table = output_parts[0]
    warning_section = output_parts[1] if len(output_parts) > 1 else ""

    # Check that 'func_ok' is in the main table, but not in the warning section
    assert "func_ok" in main_table
    assert "func_ok" not in warning_section


def test_print_pretty_table_no_data(mocker):
    """Test the "no data" message for print_pretty_table."""
    mocker.patch('pgptracker.utils.profiler.MemoryProfiler.get_profiles', return_value=[])
    
    f = StringIO()
    with contextlib.redirect_stdout(f):
        print_pretty_table()
    output = f.getvalue()
    
    assert "[WARNING] No profiling data to display" in output
    assert "MEMORY PROFILING RESULTS" not in output


def test_get_top_consumers(mocker, mock_profiles):
    """Test that top consumers are correctly sorted and sliced."""
    mocker.patch('pgptracker.utils.profiler.MemoryProfiler.get_profiles', return_value=mock_profiles)
    
    # Get top 2
    top_2 = get_top_consumers(n=2)
    assert len(top_2) == 2
    assert top_2[0].function_name == "func_high_mem" # 100MB
    assert top_2[1].function_name == "func_high_time" # 10MB
    
    # Get top 1
    top_1 = get_top_consumers(n=1)
    assert len(top_1) == 1
    assert top_1[0].function_name == "func_high_mem"


def test_print_top_consumers_success(mocker, mock_profiles):
    """Test the stdout print function for top consumers."""
    mocker.patch('pgptracker.utils.profiler.MemoryProfiler.get_profiles', return_value=mock_profiles)
    
    f = StringIO()
    with contextlib.redirect_stdout(f):
        print_top_consumers(n=2)
    output = f.getvalue()
    
    assert "Top 2 Memory Consumers" in output
    assert "1. func_high_mem" in output
    assert "Peak: 100.00 MB" in output
    assert "2. func_high_time" in output
    assert "Peak: 10.00 MB" in output
    assert "func_ok" not in output # Should be sliced off


def test_print_top_consumers_no_data(mocker):
    """Test the "no data" message for print_top_consumers."""
    mocker.patch('pgptracker.utils.profiler.MemoryProfiler.get_profiles', return_value=[])
    
    f = StringIO()
    with contextlib.redirect_stdout(f):
        print_top_consumers(n=5)
    output = f.getvalue()
    
    assert "[WARNING] No profiling data available" in output


def test_get_warnings(mocker, mock_profiles, mock_config):
    """Test the warning logic based on mock config thresholds."""
    mocker.patch('pgptracker.utils.profile_reporter.get_config', return_value=mock_config)
    
    warnings = get_warnings(mock_profiles)
    
    # p_mem (100MB > 50MB)
    # p_time (50s > 30s)
    # p_ok (5MB < 50MB, 1s < 30s)
    assert len(warnings) == 2 
    
    warn_map = {w[0].function_name: w[1] for w in warnings}
    assert "func_high_time" in warn_map
    assert warn_map["func_high_time"] == "duration"
    
    assert "func_high_mem" in warn_map
    assert warn_map["func_high_mem"] == "memory"
    
    assert "func_ok" not in warn_map


def test_print_warnings_no_warnings(mocker, mock_config):
    """Test that nothing is printed if no profiles exceed thresholds."""
    mocker.patch('pgptracker.utils.profile_reporter.get_config', return_value=mock_config)
    
    # Create a profile that is OK
    p_ok = MagicMock(spec=FunctionProfile)
    p_ok.peak_mb = 5.0
    p_ok.duration_seconds = 0.5
    
    f = StringIO()
    with contextlib.redirect_stdout(f):
        print_warnings([p_ok])
    output = f.getvalue()
    
    assert output.strip() == "" # Should print nothing