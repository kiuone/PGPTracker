"""
Conda environment manager for PGPTracker.

This module handles detection and execution of commands in the correct
conda environments (qiime2, picrust2, or pgptracker).
"""

import subprocess
import psutil
import os
import multiprocessing
from pathlib import Path
from typing import List, Optional, Dict
from functools import lru_cache

# Environment mapping
ENV_MAP = {
    "qiime": "qiime2-amplicon-2025.10",
    "picrust": "picrust2",
    "pgptracker": "pgptracker"
}

def detect_available_cores() -> int:
    """
    Detects number of available CPU cores.
    
    Returns:
        int: Number of CPU cores.
    """
    return multiprocessing.cpu_count()

# def detect_available_cores() -> int:
#     return psutil.cpu_count(logical=False) # 'False' para núcleos físicos

def detect_available_memory() -> float:
    """
    Detects available system memory in GB.
    
    Returns:
        float: Available memory in GB.
    """
    try:
        # Linux
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    return round(mem_kb / (1024 * 1024), 2)
    except FileNotFoundError:
        # Fallback for non-Linux systems
        return 0.0
    
    return 0.0

# def detect_available_memory() -> float:
#     """
#     Detects available system memory in GB (cross-platform).
#     """
#     mem = psutil.virtual_memory()
#     return round(mem.total / (1024 * 1024 * 1024), 2)


def check_conda_available() -> bool:
    """
    Checks if conda is available in the system.
    
    Returns:
        bool: True if conda is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["conda", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

@lru_cache
def check_environment_exists(env_name: str) -> bool:
    """
    Checks if a conda environment exists.
    
    Args:
        env_name: Name of conda environment.
        
    Returns:
        bool: True if environment exists, False otherwise.
    """
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            return env_name in result.stdout
        return False
        
    except FileNotFoundError:
        return False


def validate_environment(tool: str) -> str:
    """
    Validates that required environment exists for a tool.
    
    Args:
        tool: Tool name ('qiime', 'picrust', or 'pgptracker').
        
    Returns:
        str: Name of the conda environment.
        
    Raises:
        RuntimeError: If conda is not available or environment doesn't exist.
    """
    if not check_conda_available():
        raise RuntimeError(
            "Conda is not available. PGPTracker requires conda to manage environments.\n"
            "Please install Miniconda or Anaconda: https://docs.conda.io/en/latest/miniconda.html"
        )
    
    if tool not in ENV_MAP:
        raise ValueError(f"Unknown tool: {tool}. Valid options: {list(ENV_MAP.keys())}")
    
    env_name = ENV_MAP[tool]
    
    if not check_environment_exists(env_name):
        raise RuntimeError(
            f"Required conda environment '{env_name}' not found.\n"
            f"Please create the environment before running PGPTracker.\n"
            f"See installation instructions in README.md"
        )
    
    return env_name


def run_command(
    tool: str,
    cmd: List[str],
    cwd: Optional[Path] = None,
    capture_output: bool = False,
    check: bool = True
) -> subprocess.CompletedProcess:
    """
    Runs a command in the appropriate conda environment.
    
    Args:
        tool: Tool name ('qiime', 'picrust', or 'pgptracker').
        cmd: Command to run as list of strings.
        cwd: Working directory for command execution.
        capture_output: Whether to capture stdout/stderr.
        check: Whether to raise exception on non-zero exit code.
        
    Returns:
        CompletedProcess: Result of command execution.
        
    Raises:
        RuntimeError: If environment is invalid.
        subprocess.CalledProcessError: If command fails and check=True.
    """
    env_name = validate_environment(tool)
    
    # Build conda run command
    full_cmd = ["conda", "run", "-n", env_name, *cmd]
    
    # Execute command
    result = subprocess.run(
        full_cmd,
        cwd=cwd,
        capture_output=capture_output,
        text=True,
        check=False
    )
    
    if check and result.returncode != 0:
        error_msg = f"Command failed with exit code {result.returncode}:\n"
        error_msg += f"  Command: {' '.join(cmd)}\n"
        if capture_output and result.stderr:
            error_msg += f"  Error: {result.stderr[:500]}"
        raise subprocess.CalledProcessError(
            result.returncode,
            full_cmd,
            result.stdout,
            result.stderr
        )
    
    return result


def get_system_resources() -> Dict[str, any]:
    """
    Gets available system resources.
    
    Returns:
        dict: Dictionary with 'cores' and 'memory_gb' keys.
    """
    return {
        'cores': detect_available_cores(),
        'memory_gb': detect_available_memory()
    }


def print_system_info() -> None:
    """
    Prints system information including available resources.
    """
    resources = get_system_resources()
    
    print("System Resources:")
    print(f"  CPU Cores: {resources['cores']}")
    
    if resources['memory_gb'] > 0:
        print(f"  Memory: {resources['memory_gb']} GB")
    else:
        print("  Memory: Unable to detect")
    
    print()
    print("Checking conda environments...")
    
    for tool, env_name in ENV_MAP.items():
        exists = check_environment_exists(env_name)
        status = "✓" if exists else "✗"
        print(f"  {status} {env_name}")
    
    print()