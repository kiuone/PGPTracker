"""
Subcommand definitions for PGPTracker CLI.

This module defines the functions (handlers) and argument parser setups
(registration) for each individual pipeline step, allowing them to be
run independently.
"""
import argparse
import sys
import importlib.resources
import subprocess
from pathlib import Path

# Local imports
from pgptracker.utils.validator import ValidationError
from pgptracker.wrappers.qiime.export_module import export_qza_files
from pgptracker.wrappers.picrust.place_seqs import build_phylogenetic_tree
from pgptracker.wrappers.picrust.hsp_prediction import predict_gene_content
from pgptracker.stage1_processing.gen_ko_abun import run_metagenome_pipeline
from pgptracker.wrappers.qiime.classify import classify_taxonomy
from pgptracker.stage1_processing.merge_tax_abun import merge_taxonomy_to_table
from pgptracker.utils.validator import validate_output_file as _validate_output
from pgptracker.utils.env_manager import get_output_dir, get_threads
from pgptracker.stage1_processing.unstrat_pgpt import generate_unstratified_pgpt
from pgptracker.stage1_processing.strat_pgpt import generate_stratified_analysis

# Handler Functions (logic for each subcommand)
def export_command(args: argparse.Namespace) -> int:
    """Handler for the 'export' subcommand."""
    try:
        output_dir = get_output_dir(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not args.rep_seqs or not args.feature_table:
            print("ERROR: --rep-seqs and --feature-table are required.", file=sys.stderr)
            return 1
        
        seq_path = Path(args.rep_seqs)
        table_path = Path(args.feature_table)
        
        # We need to guess the format for the export_qza_files function
        seq_format = 'qza' if seq_path.suffix == '.qza' else 'fasta'
        table_format = 'qza' if table_path.suffix == '.qza' else 'biom'

        inputs = {
            'sequences': seq_path,
            'table': table_path,
            'seq_format': seq_format,
            'table_format': table_format,
            'output': output_dir  # Pass Path object
        }
        
        exported_paths = export_qza_files(inputs, output_dir)
        print("\nExport successful:")
        print(f"  -> Sequences: {exported_paths['sequences']}")
        print(f"  -> Table: {exported_paths['table']}")
        return 0

    except (ValidationError, RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n[ERROR] Export failed: {e}", file=sys.stderr)
        return 1

def place_seqs_command(args: argparse.Namespace) -> int:
    """Handler for the 'place_seqs' subcommand."""
    try:
        output_dir = get_output_dir(args.output)
        threads = get_threads(args.threads)
        seq_path = Path(args.sequences_fna)
        
        _validate_output(seq_path, "place_seqs", "representative sequences")

        tree_path = build_phylogenetic_tree(
            sequences_path=seq_path,
            output_dir=output_dir,
            threads=threads
        )
        print(f"\nPhylogenetic tree build successful:")
        print(f"  -> Output tree: {tree_path}")
        return 0
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n[ERROR] Tree build failed: {e}", file=sys.stderr)
        return 1

def hsp_command(args: argparse.Namespace) -> int:
    """Handler for the 'hsp' subcommand."""
    try:
        output_dir = get_output_dir(args.output)
        threads = get_threads(args.threads)
        tree_path = Path(args.tree)

        _validate_output(tree_path, "hsp", "phylogenetic tree")
        
        print(f"  -> Threads: {threads}")
        print(f"  -> Chunk size: {args.chunk_size}")
        
        predicted_paths = predict_gene_content(
            tree_path=tree_path,
            output_dir=output_dir,
            threads=threads,
            chunk_size=args.chunk_size
        )
        print(f"\nGene prediction successful:")
        print(f"  -> Marker file: {predicted_paths['marker']}")
        print(f"  -> KO file: {predicted_paths['ko']}")
        return 0
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n[ERROR] Gene prediction failed: {e}", file=sys.stderr)
        return 1

def metagenome_command(args: argparse.Namespace) -> int:
    """Handler for the 'metagenome' subcommand."""
    try:
        output_dir = get_output_dir(args.output)
        table_path = Path(args.table_biom)
        marker_path = Path(args.marker_gz)
        ko_path = Path(args.ko_gz)

        _validate_output(table_path, "metagenome", "feature table")
        _validate_output(marker_path, "metagenome", "marker predictions")
        _validate_output(ko_path, "metagenome", "KO predictions")
        
        pipeline_outputs = run_metagenome_pipeline(
            table_path=table_path,
            marker_path=marker_path,
            ko_predicted_path=ko_path,
            output_dir=output_dir,
            max_nsti=args.max_nsti
        )
        print(f"\nNormalization successful:")
        print(f"  -> Normalized table: {pipeline_outputs['seqtab_norm']}")
        print(f"  -> Unstratified KOs: {pipeline_outputs['pred_metagenome_unstrat']}")
        return 0
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n[ERROR] Normalization failed: {e}", file=sys.stderr)
        return 1

def classify_command(args: argparse.Namespace) -> int:
    """Handler for the 'classify' subcommand."""
    try:
        output_dir = get_output_dir(args.output)
        threads = get_threads(args.threads)
        seq_path = Path(args.rep_seqs)

        _validate_output(seq_path, "classify", "representative sequences")

        seq_format = 'qza' if seq_path.suffix == '.qza' else 'fasta'
        seq_format = 'qza' if seq_path.suffix == '.qza' else 'fasta'
            
        tax_path = classify_taxonomy(
            rep_seqs_path=seq_path,
            seq_format=seq_format,
            classifier_qza_path=Path(args.classifier_qza) if args.classifier_qza else None,
            output_dir=output_dir,
            threads=threads
        )
        
        print(f"\nTaxonomy classification successful: {tax_path}")
        return 0
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n[ERROR] Classification failed: {e}", file=sys.stderr)
        return 1

def merge_command(args: argparse.Namespace) -> int:
    """Handler for the 'merge' subcommand."""
    try:
        output_dir = get_output_dir(args.output)
        seqtab_gz_path = Path(args.seqtab_norm_gz)
        tax_tsv_path = Path(args.taxonomy_tsv)

        _validate_output(seqtab_gz_path, "merge", "normalized sequence table")
        _validate_output(tax_tsv_path, "merge", "taxonomy table")

        merged_path = merge_taxonomy_to_table(
            seqtab_norm_gz=seqtab_gz_path,
            taxonomy_tsv=tax_tsv_path,
            output_dir=output_dir,
        )
        print(f"\nTable merge successful:")
        print(f"  -> Final merged table: {merged_path}")
        return 0
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n[ERROR] Table merge failed: {e}", file=sys.stderr)
        return 1
    
def unstratify_pgpt_command(args: argparse.Namespace) -> int:
    """Handler for the 'pgpt_unstratify' subcommand."""
    try:
        # 1. Validate input
        ko_path = Path(args.ko_predictions)
        _validate_output(ko_path, "pgpt_unstratify", "unstratified KO predictions")

        # 2. Get output directory
        output_dir = get_output_dir(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> Output directory: {output_dir}")

        # 3. Run the analysis
        # This function loads the bundled database internally
        generate_unstratified_pgpt(
            unstrat_ko_path=ko_path,
            output_dir=output_dir, 
            pgpt_level=args.pgpt_level)
        
        print("\nUnstratified PGPT analysis command completed successfully.")
        return 0

    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[UNSTRATIFY ERROR] Analysis failed: {e}", file=sys.stderr)
        return 1
    
def stratify_pgpt_command(args: argparse.Namespace) -> int:
    """Handler for the 'stratify' subcommand."""
    try:
        # 1. Validate input files
        merged_table = Path(args.merged_table)
        ko_predictions = Path(args.ko_predictions)
        
        _validate_output(merged_table, "stratify", "merged taxonomy table")
        _validate_output(ko_predictions, "stratify", "KO predictions table")

        # 2. Get output directory
        output_dir = get_output_dir(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 4. Run the stratified analysis
        generate_stratified_analysis(
            merged_table_path=merged_table,
            ko_predicted_path=ko_predictions,
            output_dir=output_dir,
            taxonomic_level=args.tax_level,
            pgpt_level=args.pgpt_level,
            # batch_size=args.batch_size
        )
        
        return 0
        
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[STRATIFY ERROR] Analysis failed: {e}", file=sys.stderr)
        return 1
    
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    '--profile',
    choices=['production', 'debug', 'minimal'],
    nargs='?',
    const='production',
    default=None,
    help='Enable memory profiling (default preset if flag is used: production)'
)

# Registration Functions (Argument Parsers)
def register_export_command(subparsers: argparse._SubParsersAction):
    """Registers the 'export' subcommand."""
    export_parser = subparsers.add_parser(
        "export",
        parents=[parent_parser],
        help="Step 2: Export QIIME2 .qza files to .fna/.biom",
        description="Step 2: Export QIIME2 .qza files to standard .fna and .biom formats."
    )
    export_parser.add_argument("--rep-seqs", type=str, required=True, metavar="PATH", help="Path to representative sequences (.qza or .fna)")
    export_parser.add_argument("--feature-table", type=str, required=True, metavar="PATH", help="Path to feature table (.qza or .biom)")
    export_parser.add_argument("-o", "--output", type=str, metavar="PATH", help="Output directory (default: results/run_DD-MM-YYYY)")
    export_parser.set_defaults(func=export_command)

def register_place_seqs_command(subparsers: argparse._SubParsersAction):
    """Registers the 'place_seqs' subcommand."""
    place_seqs_parser = subparsers.add_parser(
        "place_seqs",
        parents=[parent_parser],
        help="Step 3: Build phylogenetic tree (PICRUSt2 place_seqs.py)",
        description="Step 3: Build phylogenetic tree by placing sequences into reference tree using PICRUSt2 place_seqs.py."
    )
    place_seqs_parser.add_argument("--sequences-fna", type=str, required=True, metavar="PATH", help="Path to representative sequences (.fna file)")
    place_seqs_parser.add_argument("-o", "--output", type=str, metavar="PATH", help="Output directory (default: results/run_DD-MM-YYYY)")
    place_seqs_parser.add_argument("-t", "--threads", type=int, metavar="INT", help="Number of threads (default: auto-detect)")
    place_seqs_parser.set_defaults(func=place_seqs_command)

def register_hsp_command(subparsers: argparse._SubParsersAction):
    """Registers the 'hsp' subcommand."""
    predict_parser = subparsers.add_parser(
        "hsp",
        parents=[parent_parser],
        help="Step 4: Predict gene content (PICRUSt2 hsp.py)",
        description="Step 4: Predict gene content (16S copy number and KOs) using PICRUSt2 hsp.py."
    )
    predict_parser.add_argument("--tree", type=str, required=True, metavar="PATH", help="Path to phylogenetic tree (e.g., placed_seqs.tre)")
    predict_parser.add_argument("-o", "--output", type=str, metavar="PATH", help="Output directory (default: results/run_DD-MM-YYYY)")
    predict_parser.add_argument("-t", "--threads", type=int, metavar="INT", help="Number of threads (default: auto-detect)")
    predict_parser.add_argument("--chunk-size", type=int, default=1000, metavar="INT", help="Gene families per chunk for hsp.py (default: 1000)")
    predict_parser.set_defaults(func=hsp_command)

def register_metagenome_command(subparsers: argparse._SubParsersAction):
    """Registers the 'metagenome' subcommand."""
    metagenome_parser = subparsers.add_parser(
        "metagenome",
        parents=[parent_parser],
        help="Step 5: Normalize abundances (PICRUSt2 metagenome_pipeline.py)",
        description="Step 5: Normalize abundances by 16S copy number and create KO abundance table using PICRUSt2 metagenome_pipeline.py."
    )
    metagenome_parser.add_argument("--table-biom", type=str, required=True, metavar="PATH", help="Path to exported feature table (.biom file)")
    metagenome_parser.add_argument("--marker-gz", type=str, required=True, metavar="PATH", help="Path to marker predictions (marker_nsti_predicted.tsv.gz)")
    metagenome_parser.add_argument("--ko-gz", type=str, required=True, metavar="PATH", help="Path to KO predictions (KO_predicted.tsv.gz)")
    metagenome_parser.add_argument("-o", "--output", type=str, metavar="PATH", help="Output directory (default: results/run_DD-MM-YYYY)")
    metagenome_parser.add_argument("--max-nsti", type=float, default=1.7, metavar="FLOAT", help="Maximum NSTI threshold (default: 1.7)")
    metagenome_parser.set_defaults(func=metagenome_command)

def register_classify_command(subparsers: argparse._SubParsersAction):
    """Registers the 'classify' subcommand."""
    classify_parser = subparsers.add_parser(
        "classify",
        parents=[parent_parser],
        help="Step 6: Classify taxonomy (QIIME2 classify-sklearn)",
        description="Step 6: Classify taxonomy using QIIME2 feature-classifier classify-sklearn."
    )
    classify_parser.add_argument("--rep-seqs", type=str, required=True, metavar="PATH", help="Path to representative sequences (.qza or .fna)")
    classify_parser.add_argument("-o", "--output", type=str, metavar="PATH", help="Output directory (default: results/run_DD-MM-YYYY)")
    classify_parser.add_argument("-t", "--threads", type=int, metavar="INT", help="Number of threads (default: auto-detect)")
    classify_parser.add_argument("--classifier-qza", type=str, metavar="PATH", help="Path to a custom QIIME2 classifier .qza file (default: Greengenes 2024.09)")
    classify_parser.set_defaults(func=classify_command)

def register_merge_command(subparsers: argparse._SubParsersAction):
    """Registers the 'merge' subcommand."""
    merge_parser = subparsers.add_parser(
        "merge",
        parents=[parent_parser],
        help="Step 7: Merge taxonomy into normalized feature table",
        description="Step 7: Merge the QIIME2 taxonomy file into the PICRUSt2 normalized BIOM table."
    )
    merge_parser.add_argument("--seqtab-norm-gz", type=str, required=True, metavar="PATH", help="Path to normalized table (seqtab_norm.tsv.gz)")
    merge_parser.add_argument("--taxonomy-tsv", type=str, required=True, metavar="PATH", help="Path to classified taxonomy (taxonomy.tsv)")
    merge_parser.add_argument("-o", "--output", type=str, metavar="PATH", help="Output directory (default: results/run_DD-MM-YYYY)")
    merge_parser.set_defaults(func=merge_command)

def register_unstratify_pgpt_command(subparsers: argparse._SubParsersAction):
    """Registers the 'pgpt_unstratify' subcommand."""
    unstrat_pgpt_parser = subparsers.add_parser(
        "unstratify_pgpt",
        parents=[parent_parser],
        help="Step 8: Generate unstratified PGPT table (PGPT x Sample)",
        description="Run only the unstratified analysis. "
                    "Takes PICRUSt2 KO predictions as input.")
    # Required Input
    unstrat_pgpt_parser.add_argument("-k", "--ko-predictions", type=str, required=True,
        metavar="PATH", help="Path to unstratified KO predictions (e.g., 'pred_metagenome_unstrat.tsv.gz')")
    
    # Optional Output
    unstrat_pgpt_parser.add_argument("-o", "--output",type=str,metavar="PATH", help="Output directory (default: results/run_DD-MM-YYYY)")
    unstrat_pgpt_parser.set_defaults(func=unstratify_pgpt_command)
    unstrat_pgpt_parser.add_argument("--pgpt-level", type=str, default="Lv3",
        choices=['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5'],
        metavar="LEVEL",
        help="PGPT hierarchical level to use for analysis (default: %(default)s)")

def register_stratify_pgpt_command(subparsers: argparse._SubParsersAction):
    """Registers the 'stratify' subcommand."""
    
    stratify_parser = subparsers.add_parser(
        "stratify",
        parents=[parent_parser],
        help="Step 9: Run stratified analysis (e.g., Genus x PGPT x Sample)",
        description="Run the stratified analysis on outputs from 'pgptracker process'. "
                    "This answers: 'Which taxon contributes to which PGPT?'")
    
    # Input files
    input_group = stratify_parser.add_argument_group("input files (outputs from 'pgptracker process')")
    input_group.add_argument("-i", "--merged-table", type=str, required=True, metavar="PATH",
        help="Path to the merged table (e.g., 'norm_wt_feature_table.tsv')")
    
    input_group.add_argument("-k", "--ko-predictions", type=str, required=True, metavar="PATH",
        help="Path to KO predictions (e.g., 'KO_predicted.tsv.gz')")
    
    # Parameters
    params_group = stratify_parser.add_argument_group("analysis parameters")
    params_group.add_argument("-l", "--tax-level", type=str, default="Genus",
        choices=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'],
        help="Taxonomic level to stratify by (default: %(default)s)")
    
    # params_group.add_argument("-b", "--batch-size", type=int,
    #     default=500, metavar="INT",
    #     help="Number of taxa to process per batch (default: %(default)s)")
    
    stratify_parser.add_argument("--pgpt-level", type=str, default="Lv3",
        choices=['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5'],
        metavar="LEVEL",
        help="PGPT hierarchical level to use for analysis (default: %(default)s)")
    
    # Output
    output_group = stratify_parser.add_argument_group("output options")
    output_group.add_argument("-o", "--output", type=str, metavar="PATH",
        help="Output directory (default: results/run_DD-MM-YYYY)")
    
    stratify_parser.set_defaults(func=stratify_pgpt_command)