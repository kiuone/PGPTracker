#!/usr/bin/env python3
"""
Memory profiler for PGPTracker using tracemalloc.

Provides decorator to track RAM usage per function with minimal overhead.
Captures: peak RAM, duration, DataFrame shapes, timestamps.

Author: Vivian Mello
"""

import tracemalloc
import time
import psutil
import functools
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import polars as pl
from pgptracker.utils.profiling_tools.profile_config import get_config


@dataclass
class FunctionProfile:
    """Memory profile data for a single function call."""
    function_name: str
    module_name: str
    timestamp: str
    duration_seconds: float
    peak_mb: float  # High-water mark of the process
    diff_from_baseline_mb: float
    process_total_mb: float
    system_available_mb: float
    input_shapes: str = ""
    output_shapes: str = ""
    calls: int = 1


class MemoryProfiler:
    """
    Global memory profiler using tracemalloc.
    
    Singleton pattern - only one profiler instance active.
    Collects data from all @profile_memory decorated functions.
    """
    _instance = None
    _enabled = False
    _profiles: List[FunctionProfile] = []
    _baseline_memory: Optional[int] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def enable(cls):
        """Start memory profiling."""
        if not cls._enabled:
            tracemalloc.start()
            cls._enabled = True
            # Get baseline memory (current, peak)
            baseline_mem, _ = cls._get_current_memory() 
            cls._baseline_memory = baseline_mem
    
    @classmethod
    def disable(cls):
        """Stop memory profiling."""
        if cls._enabled:
            tracemalloc.stop()
            cls._enabled = False
            print("[INFO] Memory profiling disabled")
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if profiling is active."""
        return cls._enabled
    
    @classmethod
    def add_profile(cls, profile: FunctionProfile):
        """Register a function profile."""
        cls._profiles.append(profile)
    
    @classmethod
    def get_profiles(cls) -> List[FunctionProfile]:
        """Get all collected profiles."""
        return cls._profiles
    
    @classmethod
    def clear_profiles(cls):
        """Clear all collected profiles."""
        cls._profiles = []
    
    @classmethod
    def _get_current_memory(cls) -> tuple[int, int]:
        """Get current and peak memory usage in bytes (tracemalloc)."""
        if cls._enabled:
            return tracemalloc.get_traced_memory() 
        return (0, 0)
    
    @classmethod
    def _get_system_info(cls) -> tuple[float, float]:
        """
        Get system memory info.
        
        Returns:
            (process_total_mb, system_available_mb)
        """
        process = psutil.Process()
        process_mb = process.memory_info().rss / (1024 * 1024)
        system_available_mb = psutil.virtual_memory().available / (1024 * 1024)
        return process_mb, system_available_mb


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.
    
    Tracks:
    - Peak RAM (process high-water mark)
    - Duration
    - DataFrame shapes (input/output if applicable)
    - Timestamp
    - Process total RAM
    - System available RAM
    - Diff from baseline
    
    Only active if MemoryProfiler.enable() was called.
    Respects ProfilerConfig settings (include/exclude modules).
    
    Usage:
        @profile_memory
        def my_function(df: pl.DataFrame) -> pl.DataFrame:
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if profiling is enabled
        if not MemoryProfiler.is_enabled():
            return func(*args, **kwargs)
        
        # Check if module should be profiled (from config)
        config = get_config()
        if not config.should_profile_module(func.__module__):
            return func(*args, **kwargs)
        
        # Capture initial state
        start_time = time.time()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        (mem_before, peak_before) = MemoryProfiler._get_current_memory()
        process_mb_before, system_available_mb = MemoryProfiler._get_system_info()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Capture final state
        duration = time.time() - start_time
        (mem_after, peak_after) = MemoryProfiler._get_current_memory()
        
        # Calculate memory stats
        # peak_after is the high-water mark since tracemalloc.start()
        peak_bytes = peak_after
        peak_mb = peak_bytes / (1024 * 1024)
        diff_mb = (mem_after - mem_before) / (1024 * 1024)
        
        # Get final system info
        process_mb_after, _ = MemoryProfiler._get_system_info()
        
        # Extract DataFrame shapes if config allows
        input_shapes = ""
        output_shapes = ""
        if config.track_dataframe_shapes:
            input_shapes = _extract_shapes(args, kwargs)
            output_shapes = _extract_shapes((result,), {})
        
        # Create profile record
        profile = FunctionProfile(
            function_name=func.__name__,
            module_name=func.__module__,
            timestamp=timestamp,
            duration_seconds=round(duration, 2),
            peak_mb=round(peak_mb, 2),
            diff_from_baseline_mb=round(diff_mb, 2),
            process_total_mb=round(process_mb_after, 2),
            system_available_mb=round(system_available_mb, 2) if config.track_system_info else 0.0,
            input_shapes=input_shapes,
            output_shapes=output_shapes
        )
        
        MemoryProfiler.add_profile(profile)
        
        return result
    
    return wrapper


def _extract_shapes(args: tuple, kwargs: dict) -> str:
    """
    Extract DataFrame shapes from function arguments.
    
    Args:
        args: Positional arguments
        kwargs: Keyword arguments
    
    Returns:
        String representation of shapes (e.g., "12000x66, 500x50")
    """
    shapes = []

    if args is None: # Check 1: Prevent crash if args tuple itself is None
        args = ()
    
    # Check args
    for arg in args:
        if arg is None: # Check 2: Skip None values inside the tuple (eg. (None, None))
            continue
        if isinstance(arg, pl.DataFrame):
            shapes.append(f"{arg.height}×{arg.width}")
        elif isinstance(arg, pl.LazyFrame):
            shapes.append("LazyFrame") # Do NOT call .collect()
    
    # Check kwargs
    for value in kwargs.values():
        if value is None: # Check 3: Skip None values from kwargs
            continue
        if isinstance(value, pl.DataFrame):
            shapes.append(f"{value.height}×{value.width}")
        elif isinstance(value, pl.LazyFrame):
            shapes.append("LazyFrame") # Do NOT call .collect()
    
    return ", ".join(shapes) if shapes else ""


def get_profiling_summary() -> Dict[str, Any]:
    """
    Get summary statistics of all profiled functions.
    
    Returns:
        Dictionary with summary metrics:
        - total_functions: Number of functions profiled
        - total_duration: Sum of all durations
        - max_peak_mb: Highest peak memory (process high-water mark)
    """
    profiles = MemoryProfiler.get_profiles()
    
    if not profiles:
        return {
            'total_functions': 0,
            'total_duration': 0.0,
            'max_peak_mb': 0.0
        }
    
    return {
        'total_functions': len(profiles),
        'total_duration': sum(p.duration_seconds for p in profiles),
        'max_peak_mb': max(p.peak_mb for p in profiles)
    }