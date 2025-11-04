"""
PGPTracker CLI - Main entry point for command-line interface.

This module provides the main CLI structure using argparse to process
ASV sequences and generate PGPT (Plant Growth-Promoting Trait) predictions.

Author: Vivian Mello
Advisor: Prof. Marco Antônio Bacellar
Institution: UFPR Palotina - Bioprocess and Biotechnology Engineering
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from pgptracker.utils.validators import validate_inputs
from pgptracker.qiime.export_module import export_qza_files
# Will be implemented in next artifacts
# from pgptracker.interactive import run_interactive_mode
# from pgptracker.picrust.phylo import build_phylogenetic_tree


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
        epilog="For more information, visit: https://github.com/yourusername/PGPTracker"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands"
    )
    
    # Process command (Stage 1: ASVs -> PGPTs)
    process_parser = subparsers.add_parser(
        "process",
        help="Process ASV sequences to generate PGPT predictions"
    )
    _add_process_arguments(process_parser)
    
    return parser


def _add_process_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Adds arguments for the 'process' command.
    
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
        "--stratified",
        action="store_true",
        help="Generate stratified output (Genus -> ASV -> KO -> PGPT)"
    )
    
    params_group.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save all intermediate files for debugging"
    )
    
    # Output options group
    output_group = parser.add_argument_group("output options")
    
    output_group.add_argument(
        "-o", "--output",
        type=str,
        default= None,
        metavar="PATH",
        help="Output directory (default: results/run_YYYY-MM-DD)"
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
    Executes the process command (Stage 1: ASVs -> PGPTs).
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    print("=" * 70)
    print("PGPTracker - Process Pipeline (Stage 1)")
    print("=" * 70)
    print()

    # Create output directory and creates if needed
    if args.output is None:
        date_str = datetime.date.today().isoformat()
        output_dir = Path(f"results/run_{date_str}")
    else:
        output_dir = Path(args.output)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create output directory: {e}")
        return 1 
    args.output = output_dir
    
    # Interactive mode
    if args.interactive:
        print("Starting interactive mode...")
        # Will be implemented in interactive.py
        # return run_interactive_mode()
        print("ERROR: Interactive mode not yet implemented")
        return 1
    
    # Non-interactive mode validation
    if not args.rep_seqs or not args.feature_table:
        print("ERROR: --rep-seqs and --feature-table are required in non-interactive mode")
        print("       Use --interactive for guided prompts")
        return 1
    
    # Validate input files
    print("Step 1/6: Validating input files...")
    try:
        inputs = validate_inputs(args.rep_seqs, args.feature_table, args.output)
        print(f"  -> Representative sequences: {args.rep_seqs}")
        print(f"  -> Feature table: {args.feature_table}")
        print(f"  -> Output directory: {args.output}") 
        print(f"  -> Detected formats: {inputs['seq_format']}, {inputs['table_format']}")
    except ValueError as e:
        print(f"\n{e}")
        return 1
    print()
    
    # Export .qza files if needed
    print("Step 2/6: Exporting files to standard formats...")
    try:
        exported = export_qza_files(inputs, inputs['output'])
        print()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\nERROR: Export failed: {e}")
        return 1
    
    # Build phylogenetic tree (PICRUSt2)
    print("Step 3/6: Building phylogenetic tree...")
    print("  -> Running PICRUSt2 place_seqs.py (Douglas et al., 2020)")
    # Will be implemented in picrust/phylo.py
    print()
    
    # Predict gene content (PICRUSt2)
    print("Step 4/6: Predicting gene content...")
    print("  -> Running PICRUSt2 hsp.py for marker genes (Douglas et al., 2020)")
    print("  -> Running PICRUSt2 hsp.py for KO predictions (Douglas et al., 2020)")
    # Will be implemented in picrust/predict.py
    print()
    
    # Normalize abundances (PICRUSt2)
    print("Step 5/6: Normalizing abundances...")
    print("  -> Running PICRUSt2 metagenome_pipeline.py (Douglas et al., 2020)")
    # Will be implemented in picrust/normalize.py
    print()
    
    # Generate PGPT tables
    print("Step 6/6: Generating PGPT tables...")
    print("  -> Mapping KOs to PGPTs using PLaBA database")
    # Will be implemented in analysis/pgpt.py
    print()
    
    print("=" * 70)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {args.output}")
    print("=" * 70)
    
    return 0


def main() -> int:
    """
    Main entry point for the CLI application.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "process":
        return process_command(args)
    
    # Should not reach here due to required=True
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())