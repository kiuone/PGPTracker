
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
from pgptracker.utils.validator import validate_inputs, ValidationError
from pgptracker.qiime.export_module import export_qza_files
from pgptracker.utils.env_manager import (
    detect_available_cores, detect_available_memory,
    get_output_dir, get_threads
)
from pgptracker.picrust.place_seqs import build_phylogenetic_tree
from pgptracker.picrust.hsp_prediction import predict_gene_content
from pgptracker.picrust.metagenome_p2 import run_metagenome_pipeline
from pgptracker.qiime.classify import classify_taxonomy
from pgptracker.utils.merge import merge_taxonomy_to_table

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
    print(f"Setting Max NSTI to: {args.max_nsti}")

    # Validate input files
    print("\nStep 1/8: Validating input files...")
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
    print("\nStep 2/8: Exporting files to standard formats...")
    try:
        exported = export_qza_files(inputs, inputs['output'])
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[EXPORT ERROR] Export failed: {e}", file=sys.stderr)
        return 1
    
    # Define PICRUSt2 output directory
    picrust_dir = inputs['output'] / "picrust2_intermediates"
    
    # Build phylogenetic tree (PICRUSt2)
    print("\nStep 3/8: Building phylogenetic tree...")
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
    print("\nStep 4/8: Predicting gene content...")
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
    print("\nStep 5/8: Normalizing abundances...")
    print(f" -> Using table: {exported['table']}") # Using the .biom exported
    print(" -> Running PICRUSt2 metagenome_pipeline.py (Douglas et al., 2020)")
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
    print("\nStep 6/8: Classifying taxonomy...")
    classifier_path_obj = None
    if args.classifier_qza:
        # 1. User gave a custom classifier
        print(f"\n-> Using custom classifier from: {args.classifier_qza}")
        classifier_path_obj = Path(args.classifier_qza)
        # Note: classify_taxonomy will handle the .exists() check
    else:
        # 2. User didn't provide a classifier, use default bundled
        print("\n-> Using default bundled Greengenes (2024.09) classifier.")
        try:
            classifier_path_obj = importlib.resources.files("pgptracker") / "databases" / "2024.09.taxonomy.asv.tsv.qza"
            # We pass this 'Traversable' object directly
        except FileNotFoundError:
            print("[ERROR] Default classifier (2024.09.taxonomy.asv.tsv.qza) not found!", file=sys.stderr)
            print("    This file should be bundled with PGPTracker.", file=sys.stderr)
            print("    Ensure 'databases/*.qza' is in setup.py's package_data.", file=sys.stderr)
            return 1
        print(f" -> Classifier object: {classifier_path_obj}")
        
    try:
        taxonomy_path = classify_taxonomy(
         rep_seqs_path=inputs['sequences'],    
         seq_format=inputs['seq_format'],     
         classifier_qza_path=Path(args.classifier_qza) if args.classifier_qza else None, 
         output_dir=inputs['output'], # Save in /output/taxonomy/
         threads=threads
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[TAXONOMY ERROR] Classification failed: {e}", file=sys.stderr)
        return 1

    # Merge taxonomy into feature table (BIOM)
    print("\nStep 7/8: Merging taxonomy into feature table...")
    try:
        merged_table_path = merge_taxonomy_to_table(
            seqtab_norm_gz=pipeline_outputs['seqtab_norm'],
            taxonomy_tsv=taxonomy_path,
            output_dir=inputs['output'], # Save in /output/
            save_intermediates=args.save_intermediates
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[MERGE ERROR] Table merging failed: {e}", file=sys.stderr)
        return 1
    
    # Generate PGPT tables
    print("\nStep 8/8: Generating PGPT tables...")
    print("  -> Mapping KOs to PGPTs using PLaBA database")
    # Will be implemented in analysis/pgpt.py
    # For now, just print success message
    print("Pipeline completed successfully!")
    print(f"Results saved to: {inputs['output']}")

    return 0   