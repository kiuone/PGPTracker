"""
PGPTracker CLI - Main entry point for command-line interface.

This module provides the main CLI structure using argparse to process
ASV sequences and generate PGPT (Plant Growth-Promoting Trait) predictions.

It supports four modes:
1. 'process': Runs the full pipeline end-to-end.
2. Individual steps ('export', 'place_seqs', etc.): Runs only that step.
3. 'setup': Sets up required Conda environments.
4. 'interactive': Guided prompts for input.

Author: Vivian Mello
Advisor: Prof. Marco Antônio Bacellar
Institution: UFPR Palotina - Bioprocess and Biotechnology Engineering
"""

import argparse
import sys
import importlib.resources
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

#imports from other scripts
from pgptracker.stage1_processing import pipeline_st1
from pgptracker.cli import subcommands
from pgptracker.cli.interactive import run_interactive_mode
from pgptracker.utils.profiling_tools.profiler import MemoryProfiler
from pgptracker.cli.subcommands import parent_parser
from pgptracker.utils.profiling_tools.profile_config import use_preset, get_config
from pgptracker.utils.profiling_tools.profile_reporter import generate_tsv_report, print_pretty_table
from pgptracker.utils.env_manager import check_environment_exists, ENV_MAP

def setup_command(args: argparse.Namespace) -> int:
    """
    Executes the Conda environment creation or update.
    Ensures all environments match the required .yml files.
    """
    print("PGPTracker - Environment Setup")
    
    # 1. Get the path to the packaged .yml files
    try:
        env_files_path = importlib.resources.files("pgptracker") / "environments"
    except Exception as e:
        print(f"Critical Error: Could not find the 'environments' folder. {e}")
        return 1
    
    # 2. Map environment names to their .yml files
    env_to_file_map = {
        ENV_MAP["qiime"]: "qiime2-amplicon-2025.10.yml",
        ENV_MAP["Picrust2"]: "picrust2.yml",
        # ENV_MAP["PGPTracker"]: "pgptracker.yml",
    }

    all_success = True
    if args.force:
        print("Checking and syncing environments (--force enabled)...")
    else:
        print("Checking environments (run with --force to update)...")

    for env_name, yml_filename in env_to_file_map.items():

        yml_path = env_files_path / yml_filename
        # Verify if the .yml file exists
        if not yml_path.is_file():
            print(f"\n[ERROR] Environment file not found: {yml_path}")
            print(f"       Cannot create or update environment '{env_name}'.")
            all_success = False
            continue 

        cmd = []
        action_text = ""

        env_exists = check_environment_exists(env_name)
        
        if env_exists and not args.force:
            print(f"[INFO] Environment '{env_name}' already exists. Skipping.")
            continue
        elif env_exists and args.force:
            print(f"[INFO] Environment '{env_name}' exists. Forcing update...")
            with importlib.resources.as_file(yml_path) as yml_real_path:
                cmd = ["conda", "env", "update", "--name", env_name, "-f", str(yml_path), "--prune"]
            action_text = "updated"
        else: # env_exists is False
            print(f"[INFO] Environment '{env_name}' not found. Attempting creation...")
            with importlib.resources.as_file(yml_path) as yml_real_path:
                cmd = ["conda", "env", "create", "--name", env_name, "-f", str(yml_path)]
            action_text = "created"
            
        print("       This may take several minutes...")

        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Environment '{env_name}' {action_text} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Failed to {action_text} '{env_name}'. Conda exited with code {e.returncode}.", file=sys.stderr)
            all_success = False
        except FileNotFoundError:
            print("[ERROR] 'conda' command not found. Is Conda installed and in your PATH?", file=sys.stderr)
            all_success = False
            break
            
    if all_success:
        print("Setup completed successfully!")
        return 0
    else:
        print("Setup failed for one or more environments.", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """
    Creates and configures the main argument parser.
    
    Returns:
        ArgumentParser: Configured parser with all subcommands and arguments.
    """
    parser = argparse.ArgumentParser(
        prog="pgptracker",
        description="PGPTracker: Integrate metagenomic data to correlate "
                    "microbial markers with plant biochemical traits",
        epilog="For more information, visit: https://github.com/kiuone/PGPTracker"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands:'process' (full pipeline), 'setup', or individual steps ('export', 'classify', etc.)"
    )
    
    # Process command (the main pipeline)
    process_parser = subparsers.add_parser(
        "process",
        parents=[parent_parser],
        help="Run the full pipeline (ASVs -> PGPTs)",
        description="Run the full PGPTracker pipeline from input sequences and table to final PGPT analysis."
    )
    _add_process_arguments(process_parser)
    process_parser.set_defaults(func=process_command)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Set up the required Conda environments (qiime2-amplicon-2025.10, picrust2)"
    )

    setup_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force update of existing environments to match .yml files"
        )
    
    setup_parser.set_defaults(func=setup_command)
    
    subcommands.register_export_command(subparsers)
    subcommands.register_place_seqs_command(subparsers)
    subcommands.register_hsp_command(subparsers)
    subcommands.register_metagenome_command(subparsers)
    subcommands.register_classify_command(subparsers)
    subcommands.register_merge_command(subparsers)
    subcommands.register_stratify_pgpt_command(subparsers)
    subcommands.register_unstratify_pgpt_command(subparsers)
    # STAGE 2 SUBCOMMANDS 
    subcommands.register_clr_command(subparsers)
    subcommands.register_analysis_command(subparsers)
    
    return parser

def _add_process_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Helper function to add all arguments for the 'process' command.
    
    Args:
        parser: ArgumentParser instance to add arguments to.
    """
    # Input files group
    input_group = parser.add_argument_group("input files")
    
    input_group.add_argument(
        "--rep-seqs",
        type=str,
        metavar="PATH",
        help="Path to representative sequences (.qza or .fna)"
    )

    input_group.add_argument(
         "--classifier-qza",
         type=str,
         metavar="PATH",
         default=None, # Optional
         help="Path to a custom QIIME2 classifier .qza file. "
                "If not provided, the default Greengenes (2024.09) classifier "
                "bundled with PGPTracker will be used."
    )
    
    input_group.add_argument(
        "--feature-table",
        type=str,
        metavar="PATH",
        help="Path to feature table (.qza or .biom)"
    )
    
    # Pipeline parameters group
    params_group = parser.add_argument_group("pipeline parameters")
    
    params_group.add_argument(
        "--max-nsti",
        type=float,
        default=1.7,
        metavar="FLOAT",
        help="Maximum NSTI threshold for filtering (default: 1.7)"
    )

    params_group.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        metavar="INT",
        help="Gene families per chunk for hsp.py (default: 1000)"
    )
    
    params_group.add_argument(
        "--stratified",
        action="store_true",
        help="Generate stratified output (Genus -> ASV -> KO -> PGPT)"
    )

    params_group.add_argument(
        "-l", "--tax-level",
        type=str,
        default="Genus",
        choices=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'],
        metavar="LEVEL",
        help="Taxonomic level for stratified analysis (default: Genus)"
    )

    params_group.add_argument(
        "--pgpt-level",
        type=str,
        default="Lv3",
        choices=['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5'],
        metavar="LEVEL",
        help="PGPT hierarchical level to use for analysis (default: Lv3)"
    )
    
    # Output options group
    output_group = parser.add_argument_group("output options")
    
    output_group.add_argument(
        "-o", "--output",
        type=str,
        default= None,
        metavar="PATH",
        help="Output directory (default: results/run_DD-MM-YYYY)"
    )
    
    # Computational resources group
    compute_group = parser.add_argument_group("computational resources")
    
    compute_group.add_argument(
        "-t", "--threads",
        type=int,
        default=None,
        metavar="INT",
        help="Number of threads (default: auto-detect)"
    )
    
    # Interactive mode
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode with guided prompts"
    )

def process_command(args: argparse.Namespace) -> int:
    """
    Handler for the 'process' command.
    Delegates to interactive mode or the core pipeline.
    """
    if args.interactive:
        return run_interactive_mode()
    
    # Check for required args in non-interactive mode
    if not args.rep_seqs or not args.feature_table:
        print("ERROR: --rep_seqs and --feature_table are required in non-interactive mode", file=sys.stderr)
        print("       (or, use the '-i' flag for interactive mode)", file=sys.stderr)
        return 1
    
    # Call the core pipeline logic
    print("\nPGPTracker - Full Process Pipeline (Stage 1)")
    print()
    
    exit_code = pipeline_st1.run_pipeline(args)
    
    return exit_code

def main() -> int:
    """
    The main entry point for the PGPTracker CLI application.

    This function orchestrates the main CLI execution flow:
    1.  Parses all command-line arguments using `argparse`.
    2.  Handles special cases (e.g., no arguments or the `-i` shortcut).
    3.  Checks for the `--profile` flag. If present, it enables the
        `MemoryProfiler` globally before any command is executed.
    4.  Delegates execution to the appropriate subcommand function
        (e.g., `process_command`, `setup_command`) registered in `args.func`.
    5.  After the subcommand completes (if it doesn't crash), checks again
        if profiling was enabled.
    6.  If profiling was enabled, it disables the profiler and triggers
        the generation of reports (both a .tsv file in the output
        directory and a summary table printed to the console).

    Returns:
        int: The exit code for the shell. `0` indicates success,
             while any non-zero value (typically `1`) indicates an error.
    """
    parser = create_parser()
    
    # Handle case where no command is given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1
    
    # Special case: 'pgptracker -i' → 'pgptracker process -i'
    if len(sys.argv) == 2 and sys.argv[1] == '-i':
        return run_interactive_mode()
    
    args = parser.parse_args()
    profile_name = getattr(args, 'profile', None)
    # Check if the --profile flag was used
    if profile_name:
        print(f"[INFO] Enabling profiler with preset: {args.profile}")
        use_preset(args.profile)
        MemoryProfiler.enable()

    exit_code = 1 # Default exit code on error

    # Dispatch to the correct function based on the subcommand
    if hasattr(args, 'func'):
        # Run the selected command (e.g., process_command, export_command)
        exit_code = args.func(args)
        
        # This code only runs if args.func() completes successfully.
        if profile_name and MemoryProfiler.is_enabled():
            MemoryProfiler.disable()
            
            config = get_config()
            
            # 1. Generate TSV
            # Determine output dir: use subcommand's dir, or default
            output_dir = Path("results") # Default
            if hasattr(args, 'output') and args.output:
                output_dir = Path(args.output)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / f"pgptracker_profile_{args.profile}_{datetime.now():%Y%m%d_%H%M%S}.tsv"

            generate_tsv_report(report_path)

            # 2. Print Pretty Table
            if config.show_pretty_table:
                print_pretty_table()
        
        return exit_code
    
    # Fallback if no command is given (should be caught by required=True)
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())

