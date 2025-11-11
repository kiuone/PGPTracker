
"""
PGPTracker Core Pipeline Logic.

This module contains the main 8-step pipeline logic,
which can be called by the CLI ('process') or the
interactive mode.
"""
import argparse
import sys
import importlib.resources
import subprocess
from pathlib import Path

# Local imports
from pgptracker.utils.validator import validate_inputs, ValidationError
from pgptracker.wrappers.qiime.export_module import export_qza_files
from pgptracker.utils.env_manager import (detect_available_cores, detect_available_memory,
    get_output_dir, get_threads)
from pgptracker.wrappers.picrust.place_seqs import build_phylogenetic_tree
from pgptracker.wrappers.picrust.hsp_prediction import predict_gene_content
from pgptracker.stage1_processing.gen_ko_abun import run_metagenome_pipeline
from pgptracker.wrappers.qiime.classify import classify_taxonomy
from pgptracker.stage1_processing.merge_tax_abun import merge_taxonomy_to_table
from pgptracker.stage1_processing.unstrat_pgpt import (load_pathways_db,
                                              generate_unstratified_pgpt)
from pgptracker.stage1_processing.strat_pgpt import generate_stratified_analysis
def run_pipeline(args: argparse.Namespace) -> int:
    """
    Executes the full process command (Stage 1: ASVs -> PGPTs).
    This function calls all pipeline steps sequentially.
    
    Args:
        args: Parsed command-line arguments (from CLI or interactive mode).
        
    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    # Note: These helpers get info from the 'args' object
    output_dir_str = get_output_dir(args.output)
    threads = get_threads(args.threads)
    RAM = detect_available_memory()
    print(f"Using {threads} threads for processing.")
    print(f"{RAM} of RAM available for processing.\n note: If the process get 'killed' it means you need more RAM.")

    # Validate input files
    print("\nStep 1/9: Validating input files...")
    try:
        inputs = validate_inputs(args.rep_seqs, args.feature_table, str(output_dir_str))
        print(f"  -> Representative sequences: {inputs['sequences']}")
        print(f"  -> Feature table: {inputs['table']}")
        print(f"  -> Output directory: {inputs['output']}")
        print(f"  -> Detected formats: {inputs['seq_format']}, {inputs['table_format']}")
    
    except ValidationError as e:
        print(f"\n[VALIDATION ERROR]\n{e}", file=sys.stderr)
        return 1
    print()
    
    # Export .qza files if needed
    print("\nStep 2/9: Exporting files to standard formats...")
    try:
        exported = export_qza_files(inputs, inputs['output'])
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[EXPORT ERROR] Export failed: {e}", file=sys.stderr)
        return 1
    
    # Define PICRUSt2 output directory
    picrust_dir = inputs['output'] / "picrust2_intermediates"
    
    # Build phylogenetic tree (PICRUSt2)
    print("\nStep 3/9: Building phylogenetic tree...")
    print(f" -> Using sequences: {exported['sequences']}") # Using the .fna exported
    print(" -> Running PICRUSt2 place_seqs.py (Douglas et al., 2020)")
    try:
        phylo_tree_path = build_phylogenetic_tree(
            sequences_path=exported['sequences'],
            output_dir=picrust_dir,
            threads=threads
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[PHYLO ERROR] Phylogenetic tree build failed: {e}", file=sys.stderr)
        return 1
    
    # Predict gene content (PICRUSt2)
    print("\nStep 4/9: Predicting gene content...")
    print("  -> Running PICRUSt2 hsp.py for marker genes (Douglas et al., 2020)")
    print("  -> Running PICRUSt2 hsp.py for KO predictions (Douglas et al., 2020)")
    try:
        # TODO: Add logic to pass chunk_size from args if needed
        predicted_paths = predict_gene_content(
            tree_path=phylo_tree_path,
            output_dir=picrust_dir,
            threads=threads,
            chunk_size=args.chunk_size
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[PREDICT ERROR] Gene prediction failed: {e}", file=sys.stderr)
        return 1
    
    # Normalize abundances (PICRUSt2)
    print("\nStep 5/9: Normalizing abundances and generating the unstratified table...")

    try:
        pipeline_outputs = run_metagenome_pipeline(
            table_path=exported['table'],
            marker_path=predicted_paths['marker'],
            ko_predicted_path=predicted_paths['ko'],
            output_dir=picrust_dir,
            max_nsti=args.max_nsti
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[PIPELINE ERROR] Metagenome pipeline failed: {e}", file=sys.stderr)
        return 1
    
    # Classify taxonomy (QIIME2)
    print("\nStep 6/9: Classifying taxonomy...")
    try:
        taxonomy_path=classify_taxonomy(
         rep_seqs_path=inputs['sequences'],    
         seq_format=inputs['seq_format'],     
         classifier_qza_path=Path(args.classifier_qza) if args.classifier_qza else None, 
         output_dir=inputs['output'], # Save in /output/taxonomy/
         threads=threads
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[TAXONOMY ERROR] Classification failed: {e}", file=sys.stderr)
        return 1
    print("     -> Taxonomic classification successful.")

    # Merge taxonomy into feature table (BIOM)
    print("\nStep 7/9: Merging taxonomy into feature table...")
    try:
        merged_table_path = merge_taxonomy_to_table(
            seqtab_norm_gz=pipeline_outputs['seqtab_norm'],
            taxonomy_tsv=taxonomy_path,
            output_dir=inputs['output'], # Save in /output/
            )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[MERGE ERROR] Table merging failed: {e}", file=sys.stderr)
        return 1
    
    # Generate Unstratified PGPT tables (PGPT x Sample)
    print("\nStep 8/9: Generating Unstratified PGPT tables (PGPT x Sample)...")
    try:
        unstratified_output = generate_unstratified_pgpt(
            unstrat_ko_path=pipeline_outputs['pred_metagenome_unstrat'],
            output_dir=inputs['output'], pgpt_level=args.pgpt_level)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[UNSTRATIFIED ERROR] Failed: {e}", file=sys.stderr)
        return 1
        
    stratified_output_name = None
    if args.stratified:
        print(f"\nStep 9/9: Generating Stratified PGPT tables ({args.tax_level} x PGPT x Sample)...")
        try:
            # All inputs are available from previous steps
            stratified_output = generate_stratified_analysis(
                merged_table_path=merged_table_path,
                ko_predicted_path=predicted_paths['ko'],
                output_dir=inputs['output'],
                taxonomic_level=args.tax_level,
                pgpt_level=args.pgpt_level,
                # batch_size=args.batch_size
            )
            stratified_output_name = stratified_output.name
        
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"\n[STRATIFIED ERROR] Failed: {e}", file=sys.stderr)
            print("  -> Continuing pipeline (unstratified analysis completed).")
  
    print("\nProcess pipeline completed successfully!")
    print(f"Results saved to:")
    print(f"  -> Unstratified PGPTs in: {unstratified_output.name}")
    print(f"  -> Merged Table in: {merged_table_path.name}")
    print(f"  -> KO Predictions in: {predicted_paths['ko'].name}")
    
    if stratified_output_name:
        print(f"  -> Stratified PGPTs: {stratified_output_name}")
    elif args.stratified:
        print("  -> Stratified PGPTs: FAILED (see error above)")
    else:
        print("\nNote: you can run 'pgptracker stratify' to generate stratified analysis.")

    return 0