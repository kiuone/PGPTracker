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
import polars as pl

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
from pgptracker.stage2_analysis.clr_normalize import apply_clr
from pgptracker.stage2_analysis import pipeline_st2

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

# BEGGINGING OF STAGE 2 SUBCOMMANDS

def clr_command(args: argparse.Namespace) -> int:
    """Handler for the 'clr' subcommand (Stage 2)."""
    try:
        # 1. Get paths
        input_path = Path(args.input)
        _validate_output(input_path, "clr", "input abundance table")

        output_dir = get_output_dir(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Running CLR Transformation...")
        print(f"  -> Input table: {input_path}")
        print(f"  -> Format: {args.format}")
        print(f"  -> Output directory: {output_dir}")

        # 2. Load dataframe
        df = pl.read_csv(input_path, separator='\t', has_header=True)

        # 3. Call the CLR function
        clr_outputs = apply_clr(
            df,
            format=args.format,
            sample_col=args.sample_col,
            value_col=args.value_col
        )

        # 4. Save all resulting dataframes
        print("\nCLR transformation successful. Saving outputs:")
        for key, df_out in clr_outputs.items():
            output_file = output_dir / f"{key}.tsv"
            df_out.write_csv(output_file, separator='\t')
            print(f"  -> Saved: {output_file.name}")
        
        return 0

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[CLR ERROR] CLR transformation failed: {e}", file=sys.stderr)
        return 1
    
def analysis_command(args: argparse.Namespace) -> int:
    """
    Handler for the 'analysis' subcommand (Stage 2).
    """
    if not hasattr(args, 'verbose'):
        args.verbose = False

    # Fallback: if target_col is not set, use group_col
    if args.run_ml and not args.target_col:
        args.target_col = args.group_col
    try:
        pipeline_st2.run_stage2_pipeline(args)
        return 0
    except Exception as e:
        print(f"[ERROR] Analysis pipeline failed: {e}")
        return 1
    
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    '--profile',
    choices=['production', 'debug', 'minimal'],
    nargs='?',
    const='production',
    default=False,
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

# BEGGINGING OF STAGE 2 REGISTRATION

def register_clr_command(subparsers: argparse._SubParsersAction):
    """Registers the 'clr' subcommand (Stage 2)."""
    
    clr_parser = subparsers.add_parser(
        "clr",
        parents=[parent_parser],
        help="Step 10 (Stage 2): Apply CLR transformation to an abundance table",
        description="Applies Centered Log-Ratio (CLR) transformation to a wide "
                    "or long format abundance table. Handles zeros via "
                    "multiplicative replacement.")
    
    # Input/Output
    io_group = clr_parser.add_argument_group("input/output files")
    io_group.add_argument("-i", "--input", type=str, required=True, metavar="PATH",
        help="Path to the input abundance table (must be .tsv)")
    io_group.add_argument("-o", "--output", type=str, required=True, metavar="PATH",
        help="Output directory to save the transformed table(s)")
    
    # Parameters
    params_group = clr_parser.add_argument_group("transformation parameters")
    params_group.add_argument("--format", type=str, required=True,
        choices=['wide', 'long'],
        help="Format of the input table ('wide' or 'long')")
    
    params_group.add_argument("--sample-col", type=str, default="Sample",
        metavar="NAME",
        help="Name of the sample column (for 'long' format) (default: %(default)s)")
    
    params_group.add_argument("--value-col", type=str, default="Total_PGPT_Abundance",
        metavar="NAME",
        help="Name of the abundance/value column (for 'long' format) (default: %(default)s)")
    
    clr_parser.set_defaults(func=clr_command)

def register_analysis_command(subparsers: argparse._SubParsersAction) -> None:
    """
    Registers the 'analysis' subcommand arguments.
    """
    parser = subparsers.add_parser(
        "analysis",
        parents=[parent_parser], # Inclui --profile
        help="Run Stage 2: Statistical Analysis & Machine Learning",
        description="Performs CLR normalization, Diversity Analysis, "
                    "Statistical Testing (KW/MWU), and Machine Learning (RF/Lasso)."
    )
    
    # Required Inputs
    req_group = parser.add_argument_group("required arguments")
    req_group.add_argument(
        "-i", "--input-table",
        required=True,
        metavar="PATH",
        help="Path to the Feature Table (usually unstratified/stratified TSV from Stage 1)."
    )
    req_group.add_argument(
        "-m", "--metadata",
        required=True,
        metavar="PATH",
        help="Path to the Metadata TSV file."
    )
    
    # Configuration
    conf_group = parser.add_argument_group("configuration")
    conf_group.add_argument(
        "-o", "--output-dir",
        default="results_stage2",
        metavar="PATH",
        help="Directory to save analysis results (default: results_stage2)"
    )
    conf_group.add_argument(
        "--group-col",
        required=True,
        metavar="COL",
        help="Metadata column to use for grouping (e.g., 'Treatment', 'Site'). "
             "Used for plotting and statistical tests."
    )
    
    # Normalization Config
    norm_group = parser.add_argument_group("normalization options")
    norm_group.add_argument(
        "--orientation",
        choices=['D_N', 'N_D'],
        default='D_N',
        help="Orientation of the input table. Stage 1 outputs are usually D_N (features x samples). "
             "(default: D_N)"
    )
    norm_group.add_argument(
        "--feature-col-name",
        default="Lv3",
        metavar="NAME",
        help="Name of the Feature ID column in the input table (e.g., 'Lv3', 'FeatureID'). "
             "(default: Lv3)"
    )
    
    # Analysis Modules (Flags to DISABLE)
    # By default, everything runs. User passes --no-X to disable.
    mod_group = parser.add_argument_group("analysis modules")
    
    mod_group.add_argument(
        "--no-stats",
        dest="run_stats",
        action="store_false",
        default=True,
        help="Skip Statistical Testing (Kruskal-Wallis / Mann-Whitney U)"
    )
    
    mod_group.add_argument(
        "--no-ml",
        dest="run_ml",
        action="store_false",
        default=True,
        help="Skip Machine Learning models (Random Forest / Lasso / Boruta)"
    )
    
    # ML Specifics
    ml_group = parser.add_argument_group("machine learning options")
    ml_group.add_argument(
        "--target-col",
        metavar="COL",
        default=None,
        help="Metadata column to predict. Defaults to --group-col if not set."
    )
    ml_group.add_argument(
        "--ml-type",
        choices=['classification', 'regression'],
        default='classification',
        help="Type of Machine Learning task (default: classification)"
    )
    
    # Advanced
    adv_group = parser.add_argument_group("advanced options")
    adv_group.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Perplexity parameter for t-SNE (default: 30.0). "
             "Should be less than the number of samples."
    )

    req_group.add_argument(
        "--metadata-id-col",
        default="#SampleID",  
        metavar="NAME",
        help="Name of the Sample ID column in the metadata file. "
             "Will be renamed to 'Sample' to match the feature table. (default: #SampleID)"
    )

    norm_group.add_argument(
        "--input-format",
        choices=['wide', 'long', 'stratified', 'unstratified'],
        default='wide',
        help="Format of the input table. Use 'long' for stratified outputs. (default: wide)"
        "Choices are: wide, long, stratified, unstratified."
    )

    adv_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable detailed logging info (default: only warnings/errors shown)."
    )

    adv_group.add_argument(
        "--plot-formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg", "html"],
        help="List of formats to export plots (e.g., --plot-formats png svg). Default: png pdf"
    )
    
    parser.set_defaults(func=analysis_command)