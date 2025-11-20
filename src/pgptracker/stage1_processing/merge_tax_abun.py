"""
Table merging module for PGPTracker.

Handles the complex chain of unzipping, BIOM conversion, and metadata merging.
Uses Polars for memory-efficient, streaming processing of large tables.
"""
import subprocess
import polars as pl
import sys
import shutil
from pathlib import Path
from typing import List
from pgptracker.utils.env_manager import run_command
from pgptracker.utils.validator import ValidationError
from pgptracker.utils.validator import validate_output_file as _validate_output
from pgptracker.utils.validator import find_asv_column
from pgptracker.utils.profiling_tools.profiler import profile_memory

TAXONOMY_COLS: List[str] = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

@profile_memory
def _process_taxonomy_polars(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    Lazily splits the QIIME2 taxonomy string into hierarchical levels, cleans prefixes,
    and applies a taxonomic fallback (GTDB code prefixing) using Polars expressions.
    
    Tasks:
        1. Identify ASV_ID and 'taxonomy' column.
        2. Split 'taxonomy' into Kingdom, Phylum, etc.
        3. Clean prefixes (e.g., 'k__').
        4. Apply fallback: prefix numeric/code identifiers with the last known Latin taxon.
        5. Reorder columns to: OTU/ASV_ID | Tax | Samples...
    """
    # Identify ASV column name early, before complex column manipulations
    asv_col = find_asv_column(df_lazy)

    # 1. Split taxonomy string (e.g., 'k__Bacteria; p__Firmicutes;...')
    df_lazy = df_lazy.with_columns(
        pl.col('taxonomy').str.split('; ').alias('tax_split'))
    
    # Start with base column creation, cleaning prefixes
    temp_cols = []
    for i, level_name in enumerate(TAXONOMY_COLS):
        # Extract and clean prefix
        col_expr = (
            pl.col('tax_split')
            .list.slice(i, 1)
            .list.first()
            .str.replace(r"^[dkpcofgs]__", "") # Clean prefix (e.g., k__, p__)
            .alias(level_name)
        )
        temp_cols.append(col_expr)

    df_lazy = df_lazy.with_columns(temp_cols)

    # Convert empty strings to None after initial creation to handle missing data clearly
    df_lazy = df_lazy.with_columns([
        pl.when(pl.col(level) == "").then(None).otherwise(pl.col(level)).alias(level)
        for level in TAXONOMY_COLS
    ])

    # 2. Apply Fallback Logic: Prepend the last non-null, non-code taxon to the current code
    
    last_valid_taxon_expr = pl.lit(None, dtype=pl.Utf8)
    final_taxonomy_cols = []

    for level in TAXONOMY_COLS:
        # Check if the current level is an ID code (contains digits or hyphens)
        is_code = (pl.col(level).str.contains(r"\d|-")) & pl.col(level).is_not_null()
        
        # Calculate the new value for the current column (level)
        new_value = (pl.when(is_code & last_valid_taxon_expr.is_not_null())
            .then(last_valid_taxon_expr + "-" + pl.col(level))
            .otherwise(pl.col(level))
            .alias(level))

        # Update accumulator for the next iteration: If current level is not a code, it becomes the new accumulator
        last_valid_taxon_expr = (pl.when(~is_code & pl.col(level).is_not_null())
            .then(pl.col(level))
            .otherwise(last_valid_taxon_expr)) # Keep the previous valid taxon
        
        final_taxonomy_cols.append(new_value)

    # 3. Final Selection and Reordering
    
    # Define columns to keep/exclude based on the known ASV column
    # Exclude all taxonomy helper columns from the final selection
    exclude_cols = TAXONOMY_COLS + ['tax_split', 'taxonomy', 'confidence']
    
    # Select ASV column, sample columns (via exclude), and the new final taxonomy columns
    df_lazy = df_lazy.select([pl.exclude(exclude_cols), *final_taxonomy_cols])

    # Reorder columns: ASV ID, Final Taxonomy, Sample Columns
    
    # Get all columns remaining in the dataframe (these are ASV ID + Sample Columns + Final Taxonomy Columns)
    all_cols = df_lazy.collect_schema().names()
    
    # Identify sample columns by excluding ASV ID and final taxonomy columns
    tax_and_id_cols = set(TAXONOMY_COLS) | {asv_col}
    sample_cols = [col for col in all_cols if col not in tax_and_id_cols]
    
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

    It joins the two tables on their respective ID columns and then uses the 
    `_process_taxonomy_polars` helper to parse the semi-colon-delimited taxonomy 
    string into distinct columns (e.g., 'Kingdom', 'Phylum', 'Genus', etc.). 
    The entire operation is streamed, and the final, large, merged table is 
    written directly to disk, avoiding high memory consumption.

    Args:
        seqtab_norm_gz (Path): Path to the PICRUSt2 normalized abundance table 
            (e.g., 'seqtab_norm.tsv.gz').
        taxonomy_tsv (Path): Path to the QIIME2 exported taxonomy file 
            (e.g., 'taxonomy.tsv').
        output_dir (Path): The directory to save the final merged file.

    Returns:
        Path: The path to the final merged and processed TSV file 
            (e.g., 'norm_wt_feature_table.tsv').
    """
    
    final_processed_tsv = output_dir / "norm_wt_feature_table.tsv"

    # Step 1: Scan the normalized feature table (from PICRUSt2)
    df_norm = pl.scan_csv(
        seqtab_norm_gz,
        separator='\t',
        comment_prefix="#"
    ).rename({"normalized": "OTU/ASV_ID"}) # Rename the ID column
    
    # Step 2: Scan the taxonomy table (from QIIME2)
    # FIX: Removed comment_prefix="#". The header starts with #, so we must read it as a header, not a comment.
    df_tax = pl.scan_csv(
        taxonomy_tsv,
        separator='\t',
        has_header=True,
        skip_rows=0
    ).rename({"#OTU/ASV_ID": "OTU/ASV_ID"}) # Rename the ID column
    
    # Step 3: Join the two lazyframes
    print(" \n -> Merging taxonomy and normalized table using Polars...")
    df_joined = df_norm.join(df_tax, on="OTU/ASV_ID", how="left")
    
    # Step 4: Process taxonomy strings into columns (includes fallback logic)
    processed_lazy = _process_taxonomy_polars(df_joined)

    # Step 5: Collect (stream) and write final TSV
    # Sorting by taxonomy ensures consistency and easier downstream analysis/viewing
    processed_lazy.sort(TAXONOMY_COLS, nulls_last=True).collect(engine="streaming").write_csv(
        final_processed_tsv, separator='\t')
    
    _validate_output(final_processed_tsv, "Polars Merge", "final processed table")

    # Step 6: Print Snippets
    cols = processed_lazy.sort(TAXONOMY_COLS, nulls_last=True).collect_schema().names()[:9]
    snippet_df = pl.read_csv(
        final_processed_tsv,
        separator='\t',
        columns=cols, # Read only first 9 columns
        n_rows=3, # Read only 3 rows
    )
    print("\n--- Data head, first 9 columns and first 3 rows---")
    with pl.Config(set_fmt_str_lengths=20, tbl_width_chars=160,tbl_rows=3,
                    tbl_cols=9,tbl_hide_dataframe_shape=True,
                    tbl_hide_column_data_types=True):
        print(snippet_df)

    return final_processed_tsv