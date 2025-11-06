"""
PGPTracker Interactive Mode.

This module provides the guided 4-step prompt interface
for users. It collects all necessary arguments and then
calls the core pipeline logic.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Import the core pipeline logic
from pgptracker.pipeline import run_pipeline
# Import helpers for detection
from pgptracker.utils.env_manager import detect_available_cores, detect_available_memory

# --- Helper functions for clean input ---

def _ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    """Asks a yes/no question."""
    options = "(y/n)"
    if default is True:
        options = "(Y/n)"
    elif default is False:
        options = "(y/N)"

    while True:
        resp = input(f"  {prompt} {options}: ").lower().strip()
        if resp == 'y':
            return True
        if resp == 'n':
            return False
        if resp == '' and default is not None:
            return default
        print("  Please answer 'y' or 'n'.")

def _ask_path(prompt: str, must_exist: bool = True) -> str:
    """Asks for a file path and validates it."""
    while True:
        resp = input(f"  → {prompt}: ").strip()
        if not resp:
            print("  Path cannot be empty.")
            continue
        
        path = Path(resp)
        if must_exist and not path.exists():
            print(f"  [ERROR] File not found: {path}")
            print("  Please check the path and try again.")
            continue
        
        return str(path) # Return as string, matching argparse

def _ask_float(prompt: str, default: float) -> float:
    """Asks for a float, returning default on empty."""
    while True:
        resp = input(f"  {prompt} (default: {default}): ").strip()
        if not resp:
            return default
        try:
            return float(resp)
        except ValueError:
            print("  Please enter a valid number (e.g., 1.7 or 2.0).")

def _ask_int(prompt: str, default: int) -> int:
    """Asks for an integer, returning default on empty."""
    while True:
        resp = input(f"  {prompt} (default: {default}): ").strip()
        if not resp:
            return default
        try:
            return int(resp)
        except ValueError:
            print("  Please enter a valid integer (e.g., 8 or 1000).")

# --- Main Interactive Function ---

def run_interactive_mode() -> int:
    """
    Runs the guided 4-step prompt mode.
    
    Collects arguments and passes them to the core pipeline.
    
    Returns:
        int: Exit code (0 for success, 1 for failure/cancel).
    """
    print("Running PGPTracker in Interactive Mode...")
    
    # Create a simple object to hold arguments, mimicking argparse
    args = argparse.Namespace()
    
    try:
        # --- [1/4] Input Files ---
        print("\n[1/4] Input Files")
        print("────────────────────────────────────────")
        if _ask_yes_no("Do you have .qza files?", default=False):
            args.rep_seqs = _ask_path("rep_seqs.qza path")
            args.feature_table = _ask_path("feature_table.qza path")
        else:
            args.rep_seqs = _ask_path("rep_seqs.fna path")
            args.feature_table = _ask_path("feature_table.biom path")
        
        args.classifier_qza = input("  → Custom classifier.qza path (optional, press Enter to use default): ").strip()
        if not args.classifier_qza:
            args.classifier_qza = None # Use default

        # --- [2/4] Parameters ---
        print("\n[2/4] Parameters")
        print("────────────────────────────────────────")
        args.save_intermediates = _ask_yes_no("Save intermediate files? (for debugging)", default=False)
        args.output = input("  → Output directory path (default: results/run_YYYY-MM-DD): ").strip() or None
        args.max_nsti = _ask_float("Max NSTI threshold", default=1.7)
        args.chunk_size = _ask_int("PICRUSt2 chunk size", default=1000)
        args.stratified = _ask_yes_no("Run stratified analysis?", default=False)

        # --- [3/4] Computational Resources ---
        print("\n[3/4] Computational Resources")
        print("────────────────────────────────────────")
        detected_cores = detect_available_cores()
        detected_mem = detect_available_memory()
        print(f"  Detected {detected_cores} CPU cores.")
        print(f"  Detected {detected_mem} GB RAM.")
        args.threads = _ask_int(f"Threads to use", default=detected_cores)

        # --- [4/4] Confirmation ---
        print("\n[4/4] Confirmation")
        print("────────────────────────────────────────")
        if not _ask_yes_no("Start pipeline with these settings?", default=True):
            print("Pipeline cancelled by user.")
            return 1
            
        print("\nStarting pipeline...")
        
        # Set this flag to False so the pipeline doesn't loop
        args.interactive = False 
        
        # Call the core pipeline logic with the collected args
        return run_pipeline(args)

    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user (Ctrl+C).")
        return 1
    except Exception as e:
        print(f"\n[UNHANDLED ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        return 1