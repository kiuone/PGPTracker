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

def _process_taxonomy_polars(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    Lazily splits taxonomy column into levels and reorders columns using Polars.
    
Tasks:
    1. Identify ASV_ID (first col) and taxonomy (last col).
    2. Split 'taxonomy' into Kingdom, Phylum, etc.
    3. Clean prefixes (e.g., 'k__')
    4. Reorder columns to: OTU/ASV_ID | Tax | Samples...
    """
    # Define standard taxonomic levels
    tax_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    # 1. Split taxonomy string 'k_Bacteria; p_Firmicutes; ...'
    df_lazy = df_lazy.with_columns(
        pl.col('taxonomy').str.split('; ').alias('tax_split')
    )
    
    # 2. Lazily create new columns for each level
    for i, level_name in enumerate(tax_levels):
        df_lazy = df_lazy.with_columns(
            pl.col('tax_split')
            .list.slice(i, 1)
            .list.first()
            .str.replace(r"^[dkpcofgs]__", "") # Clean prefix
            .alias(level_name)
        )
    
    # Replace empty strings (like 'g__') with null
    for level_name in tax_levels:
        df_lazy = df_lazy.with_columns(
            pl.when(pl.col(level_name) == "")
            .then(None)
            .otherwise(pl.col(level_name))
            .alias(level_name)
        )

    # 3. Reorder columns
    # Get sample columns (all columns not in the sets below)
    tax_and_helpers = set(tax_levels) | {'OTU/ASV_ID', 'taxonomy', 'confidence', 'tax_split'}
    
    # We must 'collect_schema' to know all column names for reordering
    all_cols = df_lazy.collect_schema().names()
    
    sample_cols = [col for col in all_cols if col not in tax_and_helpers]
    
    # New order: OTU/ASV_ID, then taxonomy, then all sample columns
    new_order = ['OTU/ASV_ID'] + tax_levels + sample_cols
    
    # Select the final columns in the correct order
    df_lazy = df_lazy.select(new_order)
    
    return df_lazy

def merge_taxonomy_to_table(
    seqtab_norm_gz: Path,
    taxonomy_tsv: Path,
    output_dir: Path,
    save_intermediates: bool = False
) -> Path:
    """
    Runs the table merging and processing using a Polars-native lazy pipeline.
    This replaces the gunzip -> biom convert -> biom merge -> biom convert pipeline.
    """
    
    final_processed_tsv = output_dir / "norm_wt_feature_table.tsv"
    
    # The intermediate .biom files are no longer created,
    # so we can clean up the old work directory if it exists.
    work_dir = output_dir / "merge_work"
    if not save_intermediates and work_dir.exists():
        print("  -> Cleaning up old intermediate merge_work directory...")
        try:
            shutil.rmtree(work_dir)
        except OSError as e:
            print(f"  [Warning] Could not remove old work dir: {e}")

    try:
        # Step 1: Scan the normalized feature table (from PICRUSt2)
        # PICRUSt2 output starts with '#', but the first real line is the header
        df_norm = pl.scan_csv(
            seqtab_norm_gz,
            separator='\t',
            comment_prefix="#"
        ).rename({"normalized": "OTU/ASV_ID"}) # Rename the ID column
        
        # Step 2: Scan the taxonomy table (from QIIME2)
        # This file also starts with '#' (after our fix in classify.py)
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
        processed_lazy.sort(['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'], 
                            nulls_last=True).collect(engine="streaming").write_csv(
            final_processed_tsv, separator='\t'
        )
        
        _validate_output(final_processed_tsv, "Polars Merge", "final processed table")

        # Step 6: Print Snippets (same as before)
        try:
            cols = processed_lazy.sort(['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'], 
                                       nulls_last=True).collect_schema().names()[:9]
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
        except Exception as e:
            print(f"  [Warning] Could not print data snippets: {e}")

    except Exception as e:
        # Catch Polars errors
        print(f"  [ERROR] Polars merging pipeline failed: {e}", file=sys.stderr)
        raise RuntimeError("Polars merging pipeline failed.") from e

    return final_processed_tsv