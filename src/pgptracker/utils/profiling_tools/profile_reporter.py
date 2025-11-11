#!/usr/bin/env python3
"""
Reporter for memory profiling results.

Generates TSV files and pretty terminal tables from profiling data.

Author: Vivian Mello
"""

from pathlib import Path
from typing import List, Tuple, Any, Sequence
# Import get_profiling_summary, remove FunctionProfile (not directly used here)
from pgptracker.utils.profiling_tools.profiler import MemoryProfiler, get_profiling_summary, FunctionProfile
# Only import get_config, as warnings functions will be moved here
from pgptracker.utils.profiling_tools.profile_config import get_config


def generate_tsv_report(output_path: Path) -> Path:
    """
    Generate TSV report of all profiled functions.
    
    Sorted by peak memory (descending).
    
    Args:
        output_path: Path to save TSV file
    
    Returns:
        Path to created TSV file
    
    Raises:
        ValueError: If no profiling data available
    """
    profiles = MemoryProfiler.get_profiles()
    
    if not profiles:
        raise ValueError("No profiling data available. Did you call MemoryProfiler.enable()?")
    
    # Sort by peak memory descending
    sorted_profiles = sorted(profiles, key=lambda p: p.peak_mb, reverse=True)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write TSV
    with open(output_path, 'w') as f:
        # Header (removed avg_mb)
        f.write("function\tmodule\ttimestamp\tduration_s\tpeak_mb\t"
                "diff_baseline_mb\tprocess_total_mb\tsystem_available_mb\t"
                "input_shapes\toutput_shapes\tcalls\n")
        
        # Data rows (removed p.avg_mb)
        for p in sorted_profiles:
            f.write(f"{p.function_name}\t{p.module_name}\t{p.timestamp}\t"
                    f"{p.duration_seconds}\t{p.peak_mb}\t"
                    f"{p.diff_from_baseline_mb}\t{p.process_total_mb}\t"
                    f"{p.system_available_mb}\t{p.input_shapes}\t"
                    f"{p.output_shapes}\t{p.calls}\n")
    
    print(f"[INFO] TSV report saved to: {output_path}")
    return output_path


def print_pretty_table():
    """
    Print pretty table of profiling results to terminal.
    
    Sorted by peak memory (descending).
    Includes all columns from TSV.
    """
    profiles = MemoryProfiler.get_profiles()
    
    if not profiles:
        print("[WARNING] No profiling data to display")
        return
    
    # Sort by peak memory descending
    sorted_profiles = sorted(profiles, key=lambda p: p.peak_mb, reverse=True)
    
    # Calculate column widths
    max_func_len = max(len(p.function_name) for p in sorted_profiles)
    max_module_len = max(len(p.module_name) for p in sorted_profiles)
    max_input_len = max(len(p.input_shapes) for p in sorted_profiles) if any(p.input_shapes for p in sorted_profiles) else 0
    max_output_len = max(len(p.output_shapes) for p in sorted_profiles) if any(p.output_shapes for p in sorted_profiles) else 0
    
    # Adjust widths for readability
    func_width = max(max_func_len, 30)
    module_width = max(max_module_len, 25)
    input_width = max(max_input_len, 12)
    output_width = max(max_output_len, 12)
    
    # Print header (Removed separator lines and Avg MB)
    print("\nMEMORY PROFILING RESULTS (sorted by peak RAM ↓)")
    
    header = (
        f"{'Function':<{func_width}} "
        f"{'Module':<{module_width}} "
        f"{'Timestamp':<19} "
        f"{'Duration':>10} "
        f"{'Peak MB':>10} "
        f"{'Diff MB':>10} "
        f"{'Proc MB':>10} "
        f"{'Sys Avail':>10} "
        f"{'Input Shapes':<{input_width}} "
        f"{'Output Shapes':<{output_width}}"
    )
    print(header)
    
    # Print data rows (Removed Avg MB)
    for p in sorted_profiles:
        row = (
            f"{p.function_name:<{func_width}} "
            f"{p.module_name:<{module_width}} "
            f"{p.timestamp:<19} "
            f"{p.duration_seconds:>10.2f}s "
            f"{p.peak_mb:>10.2f} "
            f"{p.diff_from_baseline_mb:>10.2f} "
            f"{p.process_total_mb:>10.2f} "
            f"{p.system_available_mb:>10.2f} "
            f"{p.input_shapes:<{input_width}} "
            f"{p.output_shapes:<{output_width}}"
        )
        print(row)
    
    # Print summary statistics (Using get_profiling_summary - DRY)
    summary = get_profiling_summary()
    
    print(f"\nSUMMARY:")
    print(f"  Total functions profiled: {summary['total_functions']}")
    print(f"  Total execution time: {summary['total_duration']:.2f}s")
    print(f"  Maximum peak memory: {summary['max_peak_mb']:.2f} MB")
    print()
    
    # Print warnings if any functions exceeded thresholds
    print_warnings(sorted_profiles)


def get_top_consumers(n: int = 5) -> List:
    """
    Get top N memory-consuming functions.
    
    Args:
        n: Number of top consumers to return
    
    Returns:
        List of FunctionProfile objects sorted by peak memory
    """
    profiles = MemoryProfiler.get_profiles()
    sorted_profiles = sorted(profiles, key=lambda p: p.peak_mb, reverse=True)
    return sorted_profiles[:n]


def print_top_consumers(n: int = 5):
    """
    Print top N memory consumers to terminal.
    
    Args:
        n: Number of top consumers to display
    """
    top = get_top_consumers(n)
    
    if not top:
        print("[WARNING] No profiling data available")
        return
    
    print(f"\nTop {len(top)} Memory Consumers:")
    for i, p in enumerate(top, 1):
        print(f"{i}. {p.function_name}")
        print(f"   Peak: {p.peak_mb:.2f} MB | Duration: {p.duration_seconds:.2f}s")
        if p.input_shapes:
            print(f"   Input shapes: {p.input_shapes}")
    print()

# ---------------------------------------------------------------------
# Functions moved from profile_config.py for better Separation of Concerns
# ---------------------------------------------------------------------

def get_warnings(profiles: Sequence[Any]) -> List[Tuple[Any, str]]:
    """
    Get list of profiles that exceeded warning thresholds.
    
    Args:
        profiles: List of FunctionProfile objects
    
    Returns:
        List of (profile, warning_type) tuples
        warning_type: 'memory', 'duration', or 'both'
    """
    config = get_config()
    warnings = []
    
    for profile in profiles:
        warning_type = []
        
        if config.is_memory_warning(profile.peak_mb):
            warning_type.append('memory')
        
        if config.is_duration_warning(profile.duration_seconds):
            warning_type.append('duration')
        
        if warning_type:
            warnings.append((profile, ' & '.join(warning_type)))
    
    return warnings


def print_warnings(profiles: List[FunctionProfile]):
    """
    Print warnings for profiles exceeding thresholds.
    
    Args:
        profiles: List of FunctionProfile objects
    """
    config = get_config()
    warnings = get_warnings(profiles)
    
    if not warnings:
        return
    
    # Removed separator lines
    print("\nWARNING: Functions exceeding thresholds")
    
    for profile, warning_type in warnings:
        print(f"\n{profile.function_name} ({profile.module_name})")
        
        if 'memory' in warning_type:
            print(f"  ⚠ Memory: {profile.peak_mb:.2f} MB "
                  f"(threshold: {config.memory_threshold_mb:.2f} MB)")
        
        if 'duration' in warning_type:
            print(f"  ⚠ Duration: {profile.duration_seconds:.2f}s "
                  f"(threshold: {config.duration_threshold_s:.2f}s)")
    
    print() # Add a final newline