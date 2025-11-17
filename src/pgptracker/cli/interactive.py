"""
PGPTracker Interactive Mode.

This module provides the guided prompt interface for users.
It supports all PGPTracker subcommands with intuitive prompts
for each argument, including the new Stage 2 Analysis pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, List, Any

# Import core functions
from pgptracker.stage1_processing.pipeline_st1 import run_pipeline
from pgptracker.stage2_analysis.pipeline_st2 import run_stage2_pipeline
from pgptracker.cli.subcommands import (
    export_command,
    place_seqs_command,
    hsp_command,
    metagenome_command,
    classify_command,
    merge_command,
    unstratify_pgpt_command,
    stratify_pgpt_command,
    clr_command,
    analysis_command
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
            print("  Please enter a valid number (e.g., 1.7 or 30.0).")

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

def _ask_choice(prompt: str, choices: List[str], default: str) -> str:
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

def _ask_string(prompt: str, default: Optional[str] = None) -> str:
    """Asks for a simple string input."""
    default_text = f" (default: {default})" if default else ""
    while True:
        resp = input(f"  → {prompt}{default_text}: ").strip()
        if not resp:
            if default:
                return default
            print("  Input cannot be empty.")
            continue
        return resp

def _display_and_ask_resources(default_threads: Optional[int] = None) -> int:
    """Display detected resources and ask for threads to use."""
    detected_cores = detect_available_cores()
    detected_mem = detect_available_memory()
    print(f"  Detected {detected_cores} CPU cores.")
    print(f"  Detected {detected_mem} GB RAM.")
    
    if default_threads is None:
        default_threads = detected_cores
    
    return _ask_int("Threads to use", default=default_threads)

# --- Subcommand Prompt Functions (Stage 1) ---

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
    
    if args.stratified:
        args.tax_level = _ask_choice(
            "Taxonomic level for stratification:",
            choices=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'],
            default='Genus'
        )

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
    print("\n=== EXPORT: Convert .qza to .fna/.biom ===\n")
    args = argparse.Namespace()
    args.rep_seqs = _ask_path("Path to representative sequences (.qza or .fna)")
    args.feature_table = _ask_path("Path to feature table (.qza or .biom)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    return args

def _prompt_place_seqs() -> argparse.Namespace:
    print("\n=== PLACE_SEQS: Build Phylogenetic Tree ===\n")
    args = argparse.Namespace()
    args.sequences_fna = _ask_path("Path to representative sequences (.fna file)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.threads = _display_and_ask_resources()
    return args

def _prompt_hsp() -> argparse.Namespace:
    print("\n=== HSP: Predict Gene Content ===\n")
    args = argparse.Namespace()
    args.tree = _ask_path("Path to phylogenetic tree (e.g., placed_seqs.tre)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.threads = _display_and_ask_resources()
    args.chunk_size = _ask_int("Gene families per chunk", default=1000)
    return args

def _prompt_metagenome() -> argparse.Namespace:
    print("\n=== METAGENOME: Normalize Abundances ===\n")
    args = argparse.Namespace()
    args.table_biom = _ask_path("Path to exported feature table (.biom file)")
    args.marker_gz = _ask_path("Path to marker predictions (marker_nsti_predicted.tsv.gz)")
    args.ko_gz = _ask_path("Path to KO predictions (KO_predicted.tsv.gz)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.max_nsti = _ask_float("Maximum NSTI threshold", default=1.7)
    return args

def _prompt_classify() -> argparse.Namespace:
    print("\n=== CLASSIFY: Taxonomy Classification ===\n")
    args = argparse.Namespace()
    args.rep_seqs = _ask_path("Path to representative sequences (.qza or .fna)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.threads = _display_and_ask_resources()
    args.classifier_qza = _ask_path("Custom classifier.qza path (optional, press Enter for default)", must_exist=True, optional=True)
    return args

def _prompt_merge() -> argparse.Namespace:
    print("\n=== MERGE: Merge Taxonomy into Table ===\n")
    args = argparse.Namespace()
    args.seqtab_norm_gz = _ask_path("Path to normalized table (seqtab_norm.tsv.gz)")
    args.taxonomy_tsv = _ask_path("Path to classified taxonomy (taxonomy.tsv)")
    args.output = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    args.save_intermediates = _ask_yes_no("Save intermediate .biom files?", default=False)
    return args

def _prompt_unstratify_pgpt() -> argparse.Namespace:
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

# --- Subcommand Prompt Functions (Stage 2) ---

def _prompt_clr() -> argparse.Namespace:
    """Prompts for 'clr' command (Manual Normalization)."""
    print("\n=== CLR: Manual CLR Normalization ===\n")
    args = argparse.Namespace()
    
    args.input_table = _ask_path("Path to input feature table (TSV)")
    args.output_dir = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    
    args.input_format = _ask_choice(
        "Format of input table:",
        choices=['wide', 'long'],
        default='wide'
    )
    
    if args.input_format == 'wide':
        args.orientation = _ask_choice(
            "Table Orientation:",
            choices=['D_N', 'N_D'],
            default='D_N'
        )
        args.feature_col_name = _ask_string("Feature ID column name", default="Lv3")
    else:
        # Defaults for long format matching Stage 1
        args.orientation = 'D_N' 
        args.feature_col_name = "Lv3"
        
    return args

def _prompt_analysis() -> argparse.Namespace:
    """Prompts for 'analysis' command (Stage 2 Pipeline)."""
    print("\n=== ANALYSIS: Stage 2 (Statistics & Machine Learning) ===\n")
    args = argparse.Namespace()
    
    # 1. Inputs
    print("[1/4] Input Data")
    print("─" * 50)
    args.input_table = _ask_path("Path to Feature Table (e.g., unstratified_pgpt.tsv or stratified.tsv)")
    args.metadata = _ask_path("Path to Metadata file (TSV)")
    args.output_dir = _ask_path("Output directory (press Enter for default)", must_exist=False, optional=True)
    
    # 2. Table Config
    print("\n[2/4] Table Configuration")
    print("─" * 50)
    args.input_format = _ask_choice(
        "Input Table Format:",
        choices=['wide', 'long'],
        default='wide'
    )
    
    if args.input_format == 'wide':
        args.orientation = _ask_choice(
            "Orientation (D_N = Features as Rows, N_D = Samples as Rows):",
            choices=['D_N', 'N_D'],
            default='D_N'
        )
        args.feature_col_name = _ask_string(
            "Name of Feature ID column (e.g. 'Lv3', 'FeatureID'):", 
            default="Lv3"
        )
    else:
        # Long format handled automatically by defaults in pipeline
        args.orientation = 'D_N' 
        args.feature_col_name = "Lv3" # Placeholder, mostly unused for long path
        
    # 3. Experimental Design
    print("\n[3/4] Experimental Design")
    print("─" * 50)
    args.group_col = _ask_string(
        "Metadata column for GROUPING (e.g., 'Treatment', 'Site') used in Stats/Plots"
    )
    
    # 4. Machine Learning
    print("\n[4/4] Machine Learning Options")
    print("─" * 50)
    args.run_stats = True # Default True
    args.run_ml = True    # Default True
    
    if _ask_yes_no("Run Machine Learning (RF/Lasso)?", default=True):
        args.run_ml = True
        args.ml_type = _ask_choice(
            "ML Task Type:",
            choices=['classification', 'regression'],
            default='classification'
        )
        
        target_default = args.group_col
        args.target_col = _ask_string(
            f"Metadata column to PREDICT (Target) [Default: {target_default}]", 
            default=target_default
        )
    else:
        args.run_ml = False
        args.ml_type = 'classification' # Dummy
        args.target_col = None

    # Advanced hidden defaults
    args.tsne_perplexity = 30.0
    args.verbose = True # Interactive users usually want to see progress
    
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
    
    # Define subcommand menu
    SUBCOMMANDS: Dict[str, Tuple[str, str, Callable, Callable]] = {
        # Stage 1
        '1': ('process', 'Run Stage 1 Pipeline (ASVs → PGPTs)', _prompt_process, run_pipeline),
        '2': ('export', 'Export .qza to .fna/.biom', _prompt_export, export_command),
        '3': ('place_seqs', 'Build phylogenetic tree', _prompt_place_seqs, place_seqs_command),
        '4': ('hsp', 'Predict gene content', _prompt_hsp, hsp_command),
        '5': ('metagenome', 'Normalize abundances', _prompt_metagenome, metagenome_command),
        '6': ('classify', 'Classify taxonomy', _prompt_classify, classify_command),
        '7': ('merge', 'Merge taxonomy into table', _prompt_merge, merge_command),
        '8': ('unstratify_pgpt', 'Generate unstratified PGPT table', _prompt_unstratify_pgpt, unstratify_pgpt_command),
        '9': ('stratify', 'Run stratified analysis', _prompt_stratify, stratify_pgpt_command),
        # Stage 2
        '10': ('analysis', 'Run Stage 2 Pipeline (Stats & ML)', _prompt_analysis, analysis_command),
        '11': ('clr', 'Manual CLR Normalization', _prompt_clr, clr_command),
    }
    
    try:
        # Display menu
        print("\nAvailable Commands:")
        print("─" * 60)
        print(" [Stage 1: Processing]")
        for key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
             cmd_name, description, _, _ = SUBCOMMANDS[key]
             print(f"  [{key}] {description}")
             
        print("\n [Stage 2: Analysis]")
        for key in ['10', '11']:
             cmd_name, description, _, _ = SUBCOMMANDS[key]
             print(f"  [{key}] {description}")
             
        print("─" * 60)
        print("  [q] Quit")
        print("─" * 60)
        
        # Get user choice
        while True:
            choice = input("\nSelect a command [1-11, q]: ").strip().lower()
            
            if choice == 'q':
                print("Exiting PGPTracker Interactive Mode.")
                return 0
            
            if choice in SUBCOMMANDS:
                cmd_name, description, prompt_func, handler_func = SUBCOMMANDS[choice]
                break
            
            print("  Invalid choice. Please select a number 1-11 or 'q' to quit.")
        
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