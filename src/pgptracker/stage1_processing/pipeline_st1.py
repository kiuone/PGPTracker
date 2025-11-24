"""
PGPTracker Core Pipeline Logic.

This module contains the main pipeline logic (Stage 1: ASVs -> PGPTs),
orchestrating the wrappers for PICRUSt2 and QIIME2.
"""
import argparse
import sys
import subprocess
from pathlib import Path
from pgptracker.utils.validator import validate_inputs, ValidationError
from pgptracker.wrappers.qiime.export_module import export_qza_files
from pgptracker.utils.env_manager import detect_available_memory, get_output_dir, get_threads
from pgptracker.wrappers.picrust.place_seqs import build_phylogenetic_tree
from pgptracker.wrappers.picrust.hsp_prediction import predict_gene_content
from pgptracker.stage1_processing.gen_ko_abun import run_metagenome_pipeline
from pgptracker.wrappers.qiime.classify import classify_taxonomy
from pgptracker.stage1_processing.merge_tax_abun import merge_taxonomy_to_table
from pgptracker.stage1_processing.unstrat_pgpt import generate_unstratified_pgpt
from pgptracker.stage1_processing.strat_pgpt import generate_stratified_analysis

def run_pipeline(args: argparse.Namespace) -> int:
    """
    Executes the full process command (Stage 1: ASVs -> PGPTs).
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    # 1. Setup Resources
    output_dir = get_output_dir(args.output)
    threads = get_threads(args.threads)
    ram_gb = detect_available_memory()
    
    print(f"Using {threads} threads for processing.")
    print(f"{ram_gb} GB of RAM available.")

    # 2. Validate Inputs
    print("\nStep 1/9: Validating input files...")
    try:
        inputs = validate_inputs(args.rep_seqs, args.feature_table, str(output_dir))
        print(f"  -> Sequences: {inputs['sequences']}")
        print(f"  -> Table: {inputs['table']}")
    except ValidationError as e:
        print(f"\n[VALIDATION ERROR]\n{e}", file=sys.stderr)
        return 1
    
    # 3. Export QIIME2 artifacts (if needed)
    print("\nStep 2/9: Exporting files to standard formats...")
    try:
        exported = export_qza_files(inputs, inputs['output'])
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[EXPORT ERROR] {e}", file=sys.stderr)
        return 1
    
    # Define intermediate directory
    picrust_dir = inputs['output'] / "picrust2_intermediates"
    
    # 4. Phylogenetic Placement
    print("\nStep 3/9: Building phylogenetic tree...")
    try:
        phylo_tree_path = build_phylogenetic_tree(
            sequences_path=exported['sequences'],
            output_dir=picrust_dir,
            threads=threads
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[PHYLO ERROR] Tree build failed: {e}", file=sys.stderr)
        return 1
    
    # 5. Gene Content Prediction (HSP)
    print("\nStep 4/9: Predicting gene content...")
    try:
        predicted_paths = predict_gene_content(
            tree_path=phylo_tree_path,
            output_dir=picrust_dir,
            threads=threads,
            chunk_size=args.chunk_size
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[PREDICT ERROR] Gene prediction failed: {e}", file=sys.stderr)
        return 1
    
    # 6. Metagenome Pipeline (Normalization)
    print("\nStep 5/9: Normalizing abundances...")
    try:
        pipeline_outputs = run_metagenome_pipeline(
            table_path=exported['table'],
            marker_path=predicted_paths['marker'],
            ko_predicted_path=predicted_paths['ko'],
            output_dir=picrust_dir,
            max_nsti=args.max_nsti
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[PIPELINE ERROR] Normalization failed: {e}", file=sys.stderr)
        return 1
    
    # 7. Taxonomy Classification
    print("\nStep 6/9: Classifying taxonomy...")
    try:
        taxonomy_path = classify_taxonomy(
            rep_seqs_path=inputs['sequences'],    
            seq_format=inputs['seq_format'],     
            classifier_qza_path=Path(args.classifier_qza) if args.classifier_qza else None, 
            output_dir=inputs['output'],
            threads=threads
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[TAXONOMY ERROR] Classification failed: {e}", file=sys.stderr)
        return 1

    # 8. Merge Taxonomy
    print("\nStep 7/9: Merging taxonomy into feature table...")
    try:
        merged_table_path = merge_taxonomy_to_table(
            seqtab_norm_gz=pipeline_outputs['seqtab_norm'],
            taxonomy_tsv=taxonomy_path,
            output_dir=inputs['output']
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[MERGE ERROR] Table merging failed: {e}", file=sys.stderr)
        return 1
    
    # 9. Unstratified Analysis
    print("\nStep 8/9: Generating Unstratified PGPT tables...")
    try:
        unstratified_output = generate_unstratified_pgpt(
            unstrat_ko_path=pipeline_outputs['pred_metagenome_unstrat'],
            output_dir=inputs['output'], 
            pgpt_level=args.pgpt_level
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[UNSTRATIFIED ERROR] Failed: {e}", file=sys.stderr)
        return 1
        
    # 10. Stratified Analysis (Optional)
    stratified_output_name = None
    if args.stratified:
        print(f"\nStep 9/9: Generating Stratified PGPT tables ({args.tax_level})...")
        try:
            stratified_output = generate_stratified_analysis(
                merged_table_path=merged_table_path,
                ko_predicted_path=predicted_paths['ko'],
                output_dir=inputs['output'],
                taxonomic_level=args.tax_level,
                pgpt_level=args.pgpt_level
            )
            stratified_output_name = stratified_output.name
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"\n[STRATIFIED ERROR] Failed: {e}", file=sys.stderr)
            print("  -> Continuing (unstratified completed).")
  
    # Summary
    print("\nProcess pipeline completed successfully!")
    print(f"Results saved to: {inputs['output']}")
    print(f"  -> Unstratified: {unstratified_output.name}")
    print(f"  -> Merged Table: {merged_table_path.name}")
    if stratified_output_name:
        print(f"  -> Stratified:   {stratified_output_name}")

    return 0