"""
Table merging module for PGPTracker.

Handles the complex chain of unzipping, BIOM conversion, and metadata merging.
Uses Polars for memory-efficient, streaming processing of large tables.
"""
import subprocess
import gzip
import shutil
import polars as pl
import sys
from pathlib import Path
from pgptracker.utils.env_manager import run_command
from pgptracker.utils.validator import ValidationError
from pgptracker.utils.validator import validate_output_file as _validate_output
from pgptracker.utils.validator import find_asv_column
from pgptracker.utils.profiling_tools.profiler import profile_memory

TAXONOMY_COLS = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

@profile_memory
def _process_taxonomy_polars(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    Lazily splits taxonomy column into levels and reorders columns using Polars.
    
Tasks:
    1. Identify ASV_ID (first col) and taxonomy (last col).
    2. Split 'taxonomy' into Kingdom, Phylum, etc.
    3. Clean prefixes (e.g., 'k__')
    4. Reorder columns to: OTU/ASV_ID | Tax | Samples...
    """
    # 1. Split taxonomy string 'k_Bacteria; p_Firmicutes; ...'
    df_lazy = df_lazy.with_columns(
        pl.col('taxonomy').str.split('; ').alias('tax_split'))
    
    # 2. Lazily create new columns for each level
    for i, level_name in enumerate(TAXONOMY_COLS):
        df_lazy = df_lazy.with_columns(
            pl.col('tax_split')
            .list.slice(i, 1)
            .list.first()
            .str.replace(r"^[dkpcofgs]__", "") # Clean prefix
            .alias(level_name))
    
    # Replace empty strings (like 'g__') with null
    for level_name in TAXONOMY_COLS:
        df_lazy = df_lazy.with_columns(
            pl.when(pl.col(level_name) == "")
            .then(None)
            .otherwise(pl.col(level_name))
            .alias(level_name))

    asv_col = find_asv_column(df_lazy)

    # 3. Reorder columns
    # Get sample columns (all columns not in the sets below)
    tax_and_helpers = set(TAXONOMY_COLS) | {asv_col, 'taxonomy', 'confidence', 'tax_split'}
    
    # We must 'collect_schema' to know all column names for reordering
    all_cols = df_lazy.collect_schema().names()
    
    sample_cols = [col for col in all_cols if col not in tax_and_helpers]
    
    # New order: OTU/ASV_ID, then taxonomy, then all sample columns
    new_order = [asv_col] + TAXONOMY_COLS + sample_cols
    
    # Select the final columns in the correct order
    df_lazy = df_lazy.select(new_order)
    
    return df_lazy


@profile_memory
def merge_taxonomy_to_table(
    seqtab_norm_gz: Path,
    taxonomy_tsv: Path,
    output_dir: Path,
) -> Path:
    """
    Merges PICRUSt2 abundances with QIIME2 taxonomy using Polars.

    It joins the two tables on their respective ID columns 
    and then uses the `_process_taxonomy_polars` helper
    to parse the semi-colon-delimited taxonomy string into distinct columns
    (e.g., 'Kingdom', 'Phylum', 'Genus', etc.).

    The entire operation is streamed, and the final, large, merged table is
    written directly to disk, avoiding high memory consumption.

    Args:
        seqtab_norm_gz (Path): Path to the PICRUSt2 normalized abundance table.
            (e.g., 'seqtab_norm.tsv.gz').
            Expected ID column: 'normalized'.
        taxonomy_tsv (Path): Path to the QIIME2 exported taxonomy file.
            (e.g., 'taxonomy.tsv').
        output_dir (Path): The directory to save the final merged file.

    Returns:
        Path: The path to the final merged and processed TSV file
            (e.g., 'norm_wt_feature_table.tsv').
    """
    
    final_processed_tsv = output_dir / "norm_wt_feature_table.tsv"

    # Step 1: Scan the normalized feature table (from PICRUSt2)
    # PICRUSt2 output starts with '#', but the first real line is the header
    df_norm = pl.scan_csv(
        seqtab_norm_gz,
        separator='\t',
        comment_prefix="#"
    ).rename({"normalized": "OTU/ASV_ID"}) # Rename the ID column
    
    # Step 2: Scan the taxonomy table (from QIIME2)
    # This file also starts with '#'
    df_tax = pl.scan_csv(
        taxonomy_tsv,
        separator='\t',
        comment_prefix=None
    ).rename({"#OTU/ASV_ID": "OTU/ASV_ID"}) # Rename the ID column
    
    # Step 3: Join the two lazyframes
    print(" \n -> Merging taxonomy and normalized table using Polars...")
    df_joined = df_norm.join(df_tax, on="OTU/ASV_ID", how="left")
    
    # Step 4: Process taxonomy strings into columns
    processed_lazy = _process_taxonomy_polars(df_joined)

    # Step 5: Collect (stream) and write final TSV
    processed_lazy.sort([TAXONOMY_COLS], nulls_last=True).collect(engine="streaming").write_csv(
        final_processed_tsv, separator='\t')
    
    _validate_output(final_processed_tsv, "Polars Merge", "final processed table")

    # Step 6: Print Snippets
    cols = processed_lazy.sort([TAXONOMY_COLS], nulls_last=True).collect_schema().names()[:9]
    snippet_df = pl.read_csv(
        final_processed_tsv,
        separator='\t',
        columns=cols, # Read only first 9 columns
        n_rows=3, # Read only 3 rows
    )
    print("\n--- Data head, first 9 columns e first 3 rows---")
    with pl.Config(set_fmt_str_lengths=20, tbl_width_chars=160,tbl_rows=3,
                    tbl_cols=9,tbl_hide_dataframe_shape=True,
                    tbl_hide_column_data_types=True):
        print(snippet_df)

    return final_processed_tsv