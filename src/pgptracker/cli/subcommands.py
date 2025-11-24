"""
Subcommand definitions for PGPTracker CLI.

Handles individual pipeline steps and Stage 2 analysis.
"""
import argparse
import sys
from pathlib import Path
import polars as pl

# Local imports
from pgptracker.utils.validator import validate_output_file as _validate_output
from pgptracker.utils.env_manager import get_output_dir, get_threads
# Wrappers
from pgptracker.wrappers.qiime.export_module import export_qza_files
from pgptracker.wrappers.picrust.place_seqs import build_phylogenetic_tree
from pgptracker.wrappers.picrust.hsp_prediction import predict_gene_content
from pgptracker.wrappers.qiime.classify import classify_taxonomy
# Stage 1 Processing
from pgptracker.stage1_processing.gen_ko_abun import run_metagenome_pipeline
from pgptracker.stage1_processing.merge_tax_abun import merge_taxonomy_to_table
from pgptracker.stage1_processing.unstrat_pgpt import generate_unstratified_pgpt
from pgptracker.stage1_processing.strat_pgpt import generate_stratified_analysis
# Stage 2 Analysis
from pgptracker.stage2_analysis.clr_normalize import apply_clr
from pgptracker.stage2_analysis import pipeline_st2

# --- Handlers ---

def export_command(args: argparse.Namespace) -> int:
    try:
        output_dir = get_output_dir(args.output)
        if not args.rep_seqs or not args.feature_table:
            print("ERROR: --rep-seqs and --feature-table required.", file=sys.stderr)
            return 1
        
        inputs = {
            'sequences': Path(args.rep_seqs),
            'table': Path(args.feature_table),
            'seq_format': 'qza' if args.rep_seqs.endswith('.qza') else 'fasta',
            'table_format': 'qza' if args.feature_table.endswith('.qza') else 'biom',
            'output': output_dir
        }
        export_qza_files(inputs, output_dir)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}", file=sys.stderr)
        return 1

def place_seqs_command(args: argparse.Namespace) -> int:
    try:
        output_dir = get_output_dir(args.output)
        threads = get_threads(args.threads)
        seq_path = Path(args.sequences_fna)
        _validate_output(seq_path, "place_seqs", "sequences")

        build_phylogenetic_tree(
            sequences_path=seq_path,
            output_dir=output_dir,
            threads=threads
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Tree build failed: {e}", file=sys.stderr)
        return 1

def hsp_command(args: argparse.Namespace) -> int:
    try:
        output_dir = get_output_dir(args.output)
        tree_path = Path(args.tree)
        threads = get_threads(args.threads)
        _validate_output(tree_path, "hsp", "tree")

        predict_gene_content(
            tree_path=tree_path,
            output_dir=output_dir,
            threads=threads,
            chunk_size=args.chunk_size
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Gene prediction failed: {e}", file=sys.stderr)
        return 1

def metagenome_command(args: argparse.Namespace) -> int:
    try:
        output_dir = get_output_dir(args.output)
        paths = {
            'table': Path(args.table_biom),
            'marker': Path(args.marker_gz),
            'ko': Path(args.ko_gz)
        }
        for name, p in paths.items():
            _validate_output(p, "metagenome", name)
        
        run_metagenome_pipeline(
            table_path=paths['table'],
            marker_path=paths['marker'],
            ko_predicted_path=paths['ko'],
            output_dir=output_dir,
            max_nsti=args.max_nsti
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Normalization failed: {e}", file=sys.stderr)
        return 1

def classify_command(args: argparse.Namespace) -> int:
    try:
        output_dir = get_output_dir(args.output)
        threads = get_threads(args.threads)
        seq_path = Path(args.rep_seqs)
        
        seq_format = 'qza' if seq_path.suffix == '.qza' else 'fasta'
        
        classify_taxonomy(
            rep_seqs_path=seq_path,
            seq_format=seq_format,
            classifier_qza_path=Path(args.classifier_qza) if args.classifier_qza else None,
            output_dir=output_dir,
            threads=threads
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Classification failed: {e}", file=sys.stderr)
        return 1

def merge_command(args: argparse.Namespace) -> int:
    try:
        merge_taxonomy_to_table(
            seqtab_norm_gz=Path(args.seqtab_norm_gz),
            taxonomy_tsv=Path(args.taxonomy_tsv),
            output_dir=get_output_dir(args.output)
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Merge failed: {e}", file=sys.stderr)
        return 1

def unstratify_pgpt_command(args: argparse.Namespace) -> int:
    try:
        generate_unstratified_pgpt(
            unstrat_ko_path=Path(args.ko_predictions),
            output_dir=get_output_dir(args.output),
            pgpt_level=args.pgpt_level
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Unstratified analysis failed: {e}", file=sys.stderr)
        return 1

def stratify_pgpt_command(args: argparse.Namespace) -> int:
    try:
        generate_stratified_analysis(
            merged_table_path=Path(args.merged_table),
            ko_predicted_path=Path(args.ko_predictions),
            output_dir=get_output_dir(args.output),
            taxonomic_level=args.tax_level,
            pgpt_level=args.pgpt_level
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Stratified analysis failed: {e}", file=sys.stderr)
        return 1

# --- Stage 2 Handlers ---

def clr_command(args: argparse.Namespace) -> int:
    try:
        output_dir = get_output_dir(args.output)
        df = pl.read_csv(args.input, separator='\t', has_header=True)
        
        outputs = apply_clr(
            df,
            format=args.format,
            sample_col=args.sample_col,
            value_col=args.value_col
        )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        for k, v in outputs.items():
            v.write_csv(output_dir / f"{k}.tsv", separator='\t')
            
        return 0
    except Exception as e:
        print(f"\n[ERROR] CLR failed: {e}", file=sys.stderr)
        return 1

def analysis_command(args: argparse.Namespace) -> int:
    try:
        if args.run_ml and not args.target_col:
            args.target_col = args.group_col
        pipeline_st2.run_stage2_pipeline(args)
        return 0
    except Exception as e:
        print(f"[ERROR] Analysis pipeline failed: {e}")
        return 1

# --- Registration ---

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--profile', choices=['production', 'debug', 'minimal'], nargs='?', const='production', default=None)

def register_export_command(subparsers):
    p = subparsers.add_parser("export", parents=[parent_parser], help="Export QZA to FNA/BIOM")
    p.add_argument("--rep-seqs", required=True)
    p.add_argument("--feature-table", required=True)
    p.add_argument("-o", "--output")
    p.set_defaults(func=export_command)

def register_place_seqs_command(subparsers):
    p = subparsers.add_parser("place_seqs", parents=[parent_parser], help="Build phylogenetic tree")
    p.add_argument("--sequences-fna", required=True, help="Input FASTA")
    p.add_argument("-o", "--output")
    p.add_argument("-t", "--threads", type=int)
    p.set_defaults(func=place_seqs_command)

def register_hsp_command(subparsers):
    p = subparsers.add_parser("hsp", parents=[parent_parser], help="Predict gene content")
    p.add_argument("--tree", required=True, help="Input tree")
    p.add_argument("-o", "--output")
    p.add_argument("-t", "--threads", type=int)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.set_defaults(func=hsp_command)

def register_metagenome_command(subparsers):
    p = subparsers.add_parser("metagenome", parents=[parent_parser], help="Normalize abundances")
    p.add_argument("--table-biom", required=True)
    p.add_argument("--marker-gz", required=True)
    p.add_argument("--ko-gz", required=True)
    p.add_argument("-o", "--output")
    p.add_argument("--max-nsti", type=float, default=1.7)
    p.set_defaults(func=metagenome_command)

def register_classify_command(subparsers):
    p = subparsers.add_parser("classify", parents=[parent_parser], help="Classify taxonomy")
    p.add_argument("--rep-seqs", required=True)
    p.add_argument("-o", "--output")
    p.add_argument("-t", "--threads", type=int)
    p.add_argument("--classifier-qza")
    p.set_defaults(func=classify_command)

def register_merge_command(subparsers):
    p = subparsers.add_parser("merge", parents=[parent_parser], help="Merge taxonomy")
    p.add_argument("--seqtab-norm-gz", required=True)
    p.add_argument("--taxonomy-tsv", required=True)
    p.add_argument("-o", "--output")
    p.set_defaults(func=merge_command)

def register_unstratify_pgpt_command(subparsers):
    p = subparsers.add_parser("unstratify_pgpt", parents=[parent_parser], help="Generate unstratified PGPTs")
    p.add_argument("-k", "--ko-predictions", required=True)
    p.add_argument("-o", "--output")
    p.add_argument("--pgpt-level", default="Lv3", choices=['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5'])
    p.set_defaults(func=unstratify_pgpt_command)

def register_stratify_pgpt_command(subparsers):
    p = subparsers.add_parser("stratify", parents=[parent_parser], help="Generate stratified PGPTs")
    p.add_argument("-i", "--merged-table", required=True)
    p.add_argument("-k", "--ko-predictions", required=True)
    p.add_argument("-o", "--output")
    p.add_argument("-l", "--tax-level", default="Genus", choices=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
    p.add_argument("--pgpt-level", default="Lv3")
    p.set_defaults(func=stratify_pgpt_command)

def register_clr_command(subparsers):
    p = subparsers.add_parser("clr", parents=[parent_parser], help="Apply CLR transformation")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--format", required=True, choices=['wide', 'long'])
    p.add_argument("--sample-col", default="Sample")
    p.add_argument("--value-col", default="Total_PGPT_Abundance")
    p.set_defaults(func=clr_command)

def register_analysis_command(subparsers):
    p = subparsers.add_parser("analysis", parents=[parent_parser], help="Stage 2 Analysis")
    p.add_argument("-i", "--input-table", required=True)
    p.add_argument("-m", "--metadata", required=True)
    p.add_argument("-o", "--output-dir", default="results_stage2")
    p.add_argument("--group-col", required=True)
    p.add_argument("--target-col", default=None)
    p.add_argument("--orientation", default="D_N", choices=["D_N", "N_D"])
    p.add_argument("--feature-col-name", default="Lv3")
    p.add_argument("--input-format", default="wide")
    p.add_argument("--metadata-id-col", default="#SampleID")
    p.add_argument("--no-stats", dest="run_stats", action="store_false", default=True)
    p.add_argument("--no-ml", dest="run_ml", action="store_false", default=True)
    p.add_argument("--ml-type", default="classification", choices=["classification", "regression"])
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    
    # Re-added arguments:
    p.add_argument("--plot-formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg", "html"], 
                   help="List of formats to export plots (default: png pdf)")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging")
    
    p.set_defaults(func=analysis_command)