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
from datetime import date

# Environment mapping
ENV_MAP = {
    "qiime": "qiime2-amplicon-2025.10",
    "Picrust2": "picrust2",
    "PGPTracker": "pgptracker"
}

def detect_available_cores() -> int:
    return psutil.cpu_count(logical=True) or 1

def detect_available_memory() -> float:
    """
    Detects available system memory in GB (cross-platform).
    """
    mem = psutil.virtual_memory()
    return round(mem.total / (1024 * 1024 * 1024), 2)


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

@lru_cache(typed=True)
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
            # Split output into lines and check for exact match
            environments = [line.split()[0] for line in result.stdout.splitlines() if line]
            return env_name in environments
        return False
        
    except FileNotFoundError:
        return False

def validate_environment(tool: str) -> str:
    """
    Validates that required environment exists for a tool.
    
    Args:
        tool: Tool name ('qiime2', 'picrust', or 'pgptracker').
        
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
        tool: Tool name ('qiime2', 'picrust', or 'pgptracker').
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
            error_msg += f"  Error: {result.stderr}"
        raise subprocess.CalledProcessError(
            result.returncode,
            full_cmd,
            result.stdout,
            result.stderr
        )
    
    return result

def get_output_dir(args_output: Optional[str]) -> Path:
    """
    Gets the validated output directory Path based on user input.
    Defaults to 'results/run_YYYY-MM-DD' if not provided.
    """
    if args_output:
        return Path(args_output)
    # Default name
    return Path(f"results/run_{date.today():%d-%m-%Y}")

def get_threads(args_threads: Optional[int]) -> int:
    """
    Gets the number of threads to use, auto-detecting if not specified.
    """
    return args_threads or detect_available_cores()

# def detect_gpu() -> tuple[bool, str]:
#     """
#     Detect GPU availability for TensorFlow/TensorLy.
    
#     Returns:
#         (has_gpu, backend_name)
#         Backends: 'cuda' (NVIDIA), 'mps' (Apple Metal), 'cpu'
    
#     Example:
#         has_gpu, backend = detect_gpu()
#         if has_gpu:
#             print(f"Using GPU backend: {backend}")
#     """
#     try:
#         import torch
#         if torch.cuda.is_available():
#             return (True, 'cuda')
#         elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#             return (True, 'mps')
#     except ImportError:
#         pass
    
#     try:
#         import tensorflow as tf
#         if tf.config.list_physical_devices('GPU'):
#             return (True, 'cuda')
#     except ImportError:
#         pass
    
#     return (False, 'cpu')