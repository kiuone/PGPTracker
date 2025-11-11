"""
PGPTracker Interactive Mode.

This module provides the guided prompt interface for users.
It supports all PGPTracker subcommands with intuitive prompts
for each argument.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple

# Import core functions
from pgptracker.stage1_processing.pipeline_st1 import run_pipeline
from pgptracker.cli.subcommands import (
    export_command,
    place_seqs_command,
    hsp_command,
    metagenome_command,
    classify_command,
    merge_command,
    unstratify_pgpt_command,
    stratify_pgpt_command
)
from pgptracker.utils.env_manager import detect_available_cores, detect_available_memory

# --- Helper Functions for Input ---

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

def _ask_path(prompt: str, must_exist: bool = True, optional: bool = False) -> Optional[str]:
    """Asks for a file path and validates it."""
    while True:
        resp = input(f"  → {prompt}: ").strip()
        
        if not resp:
            if optional:
                return None
            print("  Path cannot be empty.")
            continue
        
        path = Path(resp)
        if must_exist and not path.exists():
            print(f"  [ERROR] File not found: {path}")
            print("  Please check the path and try again.")
            continue
        
        return str(path)

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

def _ask_choice(prompt: str, choices: list, default: str) -> str:
    """Asks user to select from a list of options."""
    print(f"  {prompt}")
    for i, choice in enumerate(choices, 1):
        marker = " (default)" if choice == default else ""
        print(f"    [{i}] {choice}{marker}")
    
    while True:
        resp = input(f"  → Select [1-{len(choices)}] or press Enter for default: ").strip()
        if not resp:
            return default
        try:
            idx = int(resp) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"  Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("  Please enter a valid number.")

def _display_and_ask_resources(default_threads: Optional[int] = None) -> int:
    """Display detected resources and ask for threads to use."""
    detected_cores = detect_available_cores()
    detected_mem = detect_available_memory()
    print(f"  Detected {detected_cores} CPU cores.")
    print(f"  Detected {detected_mem} GB RAM.")
    
    if default_threads is None:
        default_threads = detected_cores
    
    return _ask_int("Threads to use", default=default_threads)

# --- Subcommand Prompt Functions ---

def _prompt_process() -> argparse.Namespace:
    """Prompts for 'process' command arguments."""
    print("\n=== PROCESS: Full Pipeline (ASVs → PGPTs) ===\n")
    args = argparse.Namespace()
    
    # Input Files
    print("[1/4] Input Files")
    print("─" * 50)
    if _ask_yes_no("Do you have .qza files?", default=False):
        args.rep_seqs = _ask_path("rep_seqs.qza path")
        args.feature_table = _ask_path("feature_table.qza path")
    else:
        args.rep_seqs = _ask_path("rep_seqs.fna path")
        args.feature_table = _ask_path("feature_table.biom path")
    
    args.classifier_qza = _ask_path("Custom classifier.qza path (optional, press Enter to use default)", must_exist=True, optional=True)

    # Parameters
    print("\n[2/4] Parameters")
    print("─" * 50)
    args.save_intermediates = _ask_yes_no("Save intermediate files? (for debugging)", default=False)
    args.output = _ask_path("Output directory path (press Enter for default: results/run_YYYY-MM-DD)", must_exist=False, optional=True)
    args.max_nsti = _ask_float("Max NSTI threshold", default=1.7)
    args.chunk_size = _ask_int("PICRUSt2 chunk size", default=1000)
    args.stratified = _ask_yes_no("Run stratified analysis?", default=False)

    # Resources
    print("\n[3/4] Computational Resources")
    print("─" * 50)
    args.threads = _display_and_ask_resources()

    # Confirmation
    print("\n[4/4] Confirmation")
    print("─" * 50)
    if not _ask_yes_no("Start pipeline with these settings?", default=True):
        print("Pipeline cancelled by user.")
        sys.exit(1)
    
    args.interactive = False
    return args

def _prompt_export() -> argparse.Namespace:
    """Prompts for 'export' command arguments."""
    print("\n=== EXPORT: Convert .qza to .fna/.biom ===\n")
    args = argparse.Namespace()
    
    args.rep_seqs = _ask_path("Path to representative sequences (.qza or .fna)")
    args.feature_table = _ask_path("Path to feature table (.qza or .biom)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    
    return args

def _prompt_place_seqs() -> argparse.Namespace:
    """Prompts for 'place_seqs' command arguments."""
    print("\n=== PLACE_SEQS: Build Phylogenetic Tree ===\n")
    args = argparse.Namespace()
    
    args.sequences_fna = _ask_path("Path to representative sequences (.fna file)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.threads = _display_and_ask_resources()
    
    return args

def _prompt_hsp() -> argparse.Namespace:
    """Prompts for 'hsp' command arguments."""
    print("\n=== HSP: Predict Gene Content ===\n")
    args = argparse.Namespace()
    
    args.tree = _ask_path("Path to phylogenetic tree (e.g., placed_seqs.tre)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.threads = _display_and_ask_resources()
    args.chunk_size = _ask_int("Gene families per chunk", default=1000)
    
    return args

def _prompt_metagenome() -> argparse.Namespace:
    """Prompts for 'metagenome' command arguments."""
    print("\n=== METAGENOME: Normalize Abundances ===\n")
    args = argparse.Namespace()
    
    args.table_biom = _ask_path("Path to exported feature table (.biom file)")
    args.marker_gz = _ask_path("Path to marker predictions (marker_nsti_predicted.tsv.gz)")
    args.ko_gz = _ask_path("Path to KO predictions (KO_predicted.tsv.gz)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.max_nsti = _ask_float("Maximum NSTI threshold", default=1.7)
    
    return args

def _prompt_classify() -> argparse.Namespace:
    """Prompts for 'classify' command arguments."""
    print("\n=== CLASSIFY: Taxonomy Classification ===\n")
    args = argparse.Namespace()
    
    args.rep_seqs = _ask_path("Path to representative sequences (.qza or .fna)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.threads = _display_and_ask_resources()
    args.classifier_qza = _ask_path("Custom classifier.qza path (optional, press Enter for default)", must_exist=True, optional=True)
    
    return args

def _prompt_merge() -> argparse.Namespace:
    """Prompts for 'merge' command arguments."""
    print("\n=== MERGE: Merge Taxonomy into Table ===\n")
    args = argparse.Namespace()
    
    args.seqtab_norm_gz = _ask_path("Path to normalized table (seqtab_norm.tsv.gz)")
    args.taxonomy_tsv = _ask_path("Path to classified taxonomy (taxonomy.tsv)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.save_intermediates = _ask_yes_no("Save intermediate .biom files?", default=False)
    
    return args

def _prompt_unstratify_pgpt() -> argparse.Namespace:
    """Prompts for 'unstratify_pgpt' command arguments."""
    print("\n=== UNSTRATIFY_PGPT: Generate Unstratified PGPT Table ===\n")
    args = argparse.Namespace()
    
    args.ko_predictions = _ask_path("Path to unstratified KO predictions (pred_metagenome_unstrat.tsv.gz)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.pgpt_level = _ask_choice(
        "PGPT hierarchical level:",
        choices=['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5'],
        default='Lv3'
    )
    
    return args

def _prompt_stratify() -> argparse.Namespace:
    """Prompts for 'stratify' command arguments."""
    print("\n=== STRATIFY: Stratified Analysis (Genus x PGPT x Sample) ===\n")
    args = argparse.Namespace()
    
    args.merged_table = _ask_path("Path to merged taxonomy table")
    args.ko_predictions = _ask_path("Path to KO predictions table")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    
    args.tax_level = _ask_choice(
        "Taxonomic level for stratification:",
        choices=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'],
        default='Genus'
    )
    
    args.pgpt_level = _ask_choice(
        "PGPT hierarchical level:",
        choices=['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5'],
        default='Lv3'
    )
    
    return args

# --- Main Interactive Function ---

def run_interactive_mode() -> int:
    """
    Runs the interactive mode with menu-driven subcommand selection.
    
    Returns:
        int: Exit code (0 for success, 1 for failure/cancel).
    """
    print("\n" + "=" * 60)
    print("  PGPTracker Interactive Mode")
    print("=" * 60)
    
    # Define subcommand menu with correct type hint
    SUBCOMMANDS: Dict[str, Tuple[str, str, Callable, Callable]] = {
        '1': ('process', 'Run full pipeline (ASVs → PGPTs)', _prompt_process, run_pipeline),
        '2': ('export', 'Export .qza files to .fna/.biom', _prompt_export, export_command),
        '3': ('place_seqs', 'Build phylogenetic tree', _prompt_place_seqs, place_seqs_command),
        '4': ('hsp', 'Predict gene content', _prompt_hsp, hsp_command),
        '5': ('metagenome', 'Normalize abundances', _prompt_metagenome, metagenome_command),
        '6': ('classify', 'Classify taxonomy', _prompt_classify, classify_command),
        '7': ('merge', 'Merge taxonomy into table', _prompt_merge, merge_command),
        '8': ('unstratify_pgpt', 'Generate unstratified PGPT table', _prompt_unstratify_pgpt, unstratify_pgpt_command),
        '9': ('stratify', 'Run stratified analysis', _prompt_stratify, stratify_pgpt_command),
    }
    
    try:
        # Display menu
        print("\nAvailable Commands:")
        print("─" * 60)
        for key, (cmd_name, description, _, _) in SUBCOMMANDS.items():
            print(f"  [{key}] {description}")
        print("  [q] Quit (or Ctrl + C to abort process in any moment)")
        print("─" * 60)
        
        # Get user choice
        while True:
            choice = input("\nSelect a command [1-9, q]: ").strip().lower()
            
            if choice == 'q':
                print("Exiting PGPTracker Interactive Mode.")
                return 0
            
            if choice in SUBCOMMANDS:
                cmd_name, description, prompt_func, handler_func = SUBCOMMANDS[choice]
                break
            
            print("  Invalid choice. Please select a number 1-9 or 'q' to quit.")
        
        # Collect arguments for selected subcommand
        args = prompt_func()
        
        # Execute the command
        print(f"\nStarting {cmd_name}...")
        return handler_func(args)

    except KeyboardInterrupt:
        print("\n\nCancelled by user (Ctrl+C).")
        return 1
    except Exception as e:
        print(f"\n[UNHANDLED ERROR] {e}", file=sys.stderr)
        return 1