"""
Metagenome pipeline runner for PGPTracker.

This module wraps and optimizes PICRUSt2's metagenome_pipeline.py 
(Douglas et al., 2020) using Polars for performance. It handles
normalization by marker genes and generation of unstratified 
metagenome predictions.

File originally named metagenome_pipeline.py
"""

from pathlib import Path
from typing import Dict, List, Any
from pgptracker.utils.env_manager import run_command
import subprocess # Keep subprocess for CalledProcessError
from pgptracker.utils.validator import validate_output_file as _validate_output
import polars as pl
import gzip
import gc # For explicit memory management
# No longer importing find_asv_column, logic is in main function
from pgptracker.utils.profiling_tools.profiler import profile_memory

@profile_memory
def _filter_by_nsti_polars(
    lf: pl.LazyFrame, 
    max_nsti: float, 
    id_col: str
) -> pl.LazyFrame:
    """
    Filters a LazyFrame by max_nsti and drops PICRUSt2 metadata columns.
    
    Args:
        lf: The LazyFrame to filter (e.g., marker or function predictions).
        max_nsti: The maximum NSTI value to retain.
        id_col: The name of the sequence identifier column.

    Returns:
        A filtered LazyFrame.
    """
    # Get column names at this stage of the plan
    column_names = lf.collect_schema().names()
    
    if 'metadata_NSTI' in column_names:
        lf = lf.filter(pl.col('metadata_NSTI') <= max_nsti)
        
    meta_cols = [
        c for c in column_names
        if c.startswith('metadata_') or c == 'closest_reference_genome'
    ]
    if meta_cols:
        lf = lf.drop(meta_cols)
        
    return lf

@profile_memory
def _normalize_by_marker_polars(
    lf_table: pl.LazyFrame, 
    lf_marker: pl.LazyFrame, 
    sample_cols: list[str], 
    id_col: str
) -> pl.LazyFrame:
    """
    Normalizes ASV abundances by marker gene copy numbers (Polars implementation).

    This replicates PICRUSt2's norm_by_marker_copies.
    
    Args:
        lf_table: LazyFrame of feature (ASV) abundances.
        lf_marker: LazyFrame of predicted marker gene counts.
        sample_cols: List of sample column names.
        id_col: The name of the sequence identifier column.

    Returns:
        A LazyFrame of normalized abundances, rounded to 2 decimals.
    """
    # Get marker column names at this stage of the plan
    marker_column_names = lf_marker.collect_schema().names()
    
    # Identify the marker count column (it's the only one besides id_col)
    marker_col = [c for c in marker_column_names if c != id_col][0]

    # Prepare marker df
    lf_marker = lf_marker.select(
        pl.col(id_col),
        # Division by zero is expected to produce 'inf' (infinity),
        # matching the behavior of Pandas/Numpy. Do not handle this
        # as a special case (e.g., converting 0 to 1), as that
        # would silently corrupt the normalization (Abund / 1).
        pl.col(marker_col).alias("marker_copies")
    )

    # Join and normalize all sample columns
    lf_norm = lf_table.join(lf_marker, on=id_col, how="inner")
    
    # Dynamically build expressions for all sample columns
    norm_expressions = [
        (pl.col(s) / pl.col("marker_copies")).round(2)
        for s in sample_cols
    ]
    
    lf_norm = lf_norm.with_columns(norm_expressions)
    
    return lf_norm.select([id_col] + sample_cols)

@profile_memory
def _unstrat_funcs_only_by_samples_polars(
    lf_func: pl.LazyFrame, 
    lf_norm: pl.LazyFrame, 
    ko_cols: list[str], 
    sample_cols: list[str],
    id_col: str
) -> pl.LazyFrame:
    """
    Generates the unstratified KO x Sample abundance table using Polars.

    This replicates PICRUSt2's unstrat_funcs_only_by_samples using
    database-style joins and aggregations instead of loops.
    
    Args:
        lf_func: LazyFrame of predicted KO copy numbers (ASV x KO).
        lf_norm: LazyFrame of normalized abundances (ASV x Sample).
        ko_cols: List of KO column names (e.g., 'ko:K00001').
        sample_cols: List of sample column names.
        id_col: The name of the sequence identifier column.

    Returns:
        A LazyFrame of the final unstratified (KO x Sample) table.
    """
    
    # 1. Unpivot KO predictions (ASV x KO -> ASV, KO, Copy_Num)
    # Filter copy_num > 0 early to reduce join size
    lf_func_long = lf_func.select(
        [id_col] + ko_cols
    ).unpivot(
        index=id_col,
        on=ko_cols,
        variable_name="function",
        value_name="copy_num"
    ).filter(pl.col('copy_num') > 0)

    # 2. Unpivot Normalized Abundances (ASV x Sample -> ASV, Sample, Abundance)
    # Filter abundance > 0 early to reduce join size
    lf_norm_long = lf_norm.select(
        [id_col] + sample_cols
    ).unpivot(
        index=id_col,
        on=sample_cols,
        variable_name="sample",
        value_name="abundance"
    ).filter(pl.col('abundance') > 0)

    # 3. Join
    # (ASV, Sample, Abundance) JOIN (ASV, KO, Copy_Num) on ASV_ID
    # This creates the full contribution table in long format
    lf_joined = lf_norm_long.join(lf_func_long, on=id_col, how="inner")

    # 4. Calculate Contribution
    # This is the core metagenome prediction calculation
    lf_contrib = lf_joined.with_columns(
        (pl.col("abundance") * pl.col("copy_num")).alias("unstrat_abun"))

    # 5. Aggregate (Sum by KO and Sample)
    lf_agg = lf_contrib.group_by(["function", "sample"]).agg(
        pl.col("unstrat_abun").sum())
    
    df_agg = lf_agg.collect(engine="streaming")

    # 6. Pivot to final (KO x Sample) table
    df_pivot = df_agg.pivot(
        values="unstrat_abun",
        index="function",
        on="sample",
        aggregate_function="sum"
    ).fill_null(0.0) # Replace missing values with 0

    # Return as LazyFrame to match the main pipeline expectation
    return df_pivot.lazy()

@profile_memory
def run_metagenome_pipeline(
    table_path: Path,
    marker_path: Path,
    ko_predicted_path: Path,
    output_dir: Path,
    max_nsti: float = 1.7
) -> Dict[str, Path]:
    """
    Normalizes abundances and generates unstratified metagenome predictions
    using Polars.
    
    This function replicates the core logic of PICRUSt2's
    metagenome_pipeline.py (norm_by_marker_copies and 
    unstrat_funcs_only_by_samples) for performance.

    Args:
        table_path: Path to feature table (.biom or .tsv)
        marker_path: Path to marker predictions (marker_nsti_predicted.tsv.gz)
        ko_predicted_path: Path to KO predictions (KO_predicted.tsv.gz)
        output_dir: Base directory for PICRUSt2 outputs
        max_nsti: Maximum NSTI threshold for filtering (default: 1.7)
        
    Returns:
        Dictionary with paths to output files:
            - 'seqtab_norm': Normalized feature table (seqtab_norm.tsv.gz)
            - 'pred_metagenome_unstrat': Unstratified KO abundances (pred_metagenome_unstrat.tsv.gz)
            
    Raises:
        FileNotFoundError: (via Polars) If any input file doesn't exist
        RuntimeError: If output validation fails or file is empty
        pl.exceptions.ColumnNotFoundError: If expected ASV_ID columns aren't found
    """
    # 1. Validate inputs
    # No p.exists() check needed (Violation of Rule 2).
    # Polars' scan_csv will raise FileNotFoundError if missing.
    
    # 2. Define output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    # Maintain PICRUSt2's original directory structure for consistency
    metagenome_out_dir = output_dir / "KO_metagenome_out" 
    metagenome_out_dir.mkdir(parents=True, exist_ok=True)
    
    seqtab_norm_path = metagenome_out_dir / "seqtab_norm.tsv.gz"
    pred_unstrat_path = metagenome_out_dir / "pred_metagenome_unstrat.tsv.gz"

    # --- BIOM Conversion Step ---
    if table_path.suffix == ".biom":
        table_tsv_path = table_path.with_suffix(".tsv")
        print("     -> Converting BIOM to TSV for Polars processing...")
        convert_cmd = [
            "biom", "convert",
            "-i", str(table_path),
            "-o", str(table_tsv_path),
            "--to-tsv"
        ]
        # Use 'qiime' env, which contains 'biom-format'
        run_command("qiime", convert_cmd, check=True)
        table_path_to_load = table_tsv_path
    else:
        table_path_to_load = table_path

    print("     -> Loading tables with Polars (Lazy)...")
    # 3. Load Data (Lazy)
    # Use 'try_parse_dates=False' for performance and to avoid schema errors
    lf_table = pl.scan_csv(
        table_path_to_load, separator='\t', skip_rows=1, comment_prefix=None, try_parse_dates=False
    )
    lf_marker = pl.scan_csv(
        marker_path, separator='\t', comment_prefix='#', try_parse_dates=False
    )
    lf_func = pl.scan_csv(
        ko_predicted_path, separator='\t', comment_prefix='#', try_parse_dates=False
    )

    # 4. Find ASV_ID columns
    # Collect schema (fast operation) to get column names
    table_schema = lf_table.collect_schema()
    marker_schema = lf_marker.collect_schema()
    func_schema = lf_func.collect_schema()

    ASV_ID_CANDIDATES: List[str] = ['ASV_ID', '#ASV_ID', 'feature-id', 
                                    '#OTU ID', 'sequence', 'normalized']
    
    id_cols = []
    # Create a list of tuples (schema, file_description) to loop over
    schema_info = [
        (table_schema, "table file"),
        (marker_schema, "marker file"),
        (func_schema, "function file")
    ]

    for schema, file_desc in schema_info:
        column_names = schema.names()
        asv_col = next((c for c in ASV_ID_CANDIDATES if c in column_names), None)

        # VALIDATION: Check if ASV_ID column was found
        if asv_col is None:
            raise pl.exceptions.ColumnNotFoundError(
                f"Could not find a valid ASV_ID column in {file_desc} ('{table_path_to_load}'). "
                f"Expected one of: {ASV_ID_CANDIDATES}"
            )
        id_cols.append(asv_col)

    # Unpack the results
    id_col_table, id_col_marker, id_col_func = id_cols

    # Standardize ASV_ID column name across all frames for joins
    lf_table = lf_table.rename({id_col_table: "ASV_ID"})
    lf_marker = lf_marker.rename({id_col_marker: "ASV_ID"})
    lf_func = lf_func.rename({id_col_func: "ASV_ID"})
    id_col = "ASV_ID"

    # 5. Filter by NSTI (replicates drop_tips_by_nsti)
    print(f"     -> Filtering ASVs by NSTI <= {max_nsti}...")
    lf_marker_filt = _filter_by_nsti_polars(lf_marker, max_nsti, id_col)
    lf_func_filt = _filter_by_nsti_polars(lf_func, max_nsti, id_col)

    # 6. Get Overlapping ASVs (replicates three_df_index_overlap_sort)
    # Collect unique ASV lists (small dataframes) into memory.
    # Joining small in-memory dataframes is faster than a complex
    # multi-join lazy query plan.
    asvs_marker_df = lf_marker_filt.select(id_col).unique().collect()
    asvs_func_df = lf_func_filt.select(id_col).unique().collect()
    asvs_table_df = lf_table.select(id_col).unique().collect()

    # Join the in-memory dataframes to find the common set
    common_asvs_df = asvs_table_df.join(
        asvs_marker_df, on=id_col, how="inner"
    ).join(
        asvs_func_df, on=id_col, how="inner"
    )
    
    # Free intermediate dfs (Rule 10)
    del asvs_marker_df, asvs_func_df, asvs_table_df
    
    common_asvs_lf = common_asvs_df.lazy()

    # Filter all tables to common ASVs using the lazy common set
    lf_table_common = lf_table.join(common_asvs_lf, on=id_col, how="inner")
    lf_marker_common = lf_marker_filt.join(common_asvs_lf, on=id_col, how="inner")
    lf_func_common = lf_func_filt.join(common_asvs_lf, on=id_col, how="inner")

    # Get column lists
    sample_cols = [c for c in table_schema.names() if c != id_col_table]
    # KOs are prefixed; this captures all function columns
    ko_cols = [c for c in func_schema.names() if c.startswith("ko:")]

    # --- 7. Normalization (replicates norm_by_marker_copies) ---
    lf_norm_plan = _normalize_by_marker_polars(lf_table_common, lf_marker_common, 
                                               sample_cols, id_col)
    
    # CRITICAL: Collect the normalized table ONCE to prevent re-computation.
    # This materializes the result of the normalization plan.
    df_norm = lf_norm_plan.collect(engine="streaming")
    
    # Free memory from large lazy dependencies (Rule 10)
    del lf_table_common, lf_marker_common, lf_norm_plan, common_asvs_lf
    gc.collect()

    # Save normalized table
    # Rename ID col for PICRUSt2 compatibility
    df_norm_to_save = df_norm.rename({id_col: "normalized"})
    
    with gzip.open(seqtab_norm_path, 'wb') as f:
        df_norm_to_save.write_csv(f, separator='\t')
    
    _validate_output(seqtab_norm_path, "Polars Normalization", "normalized sequence table")
    print(f"     -> Normalized table saved: {seqtab_norm_path.name}")

    # --- 8. Unstratified Prediction (replicates unstrat_funcs_only_by_samples) ---
    print("     -> Generating unstratified KO predictions...")
    
    # Pass the collected DataFrame (as lazy) to the next step
    lf_unstrat_plan = _unstrat_funcs_only_by_samples_polars(
        lf_func_common, df_norm.lazy(), ko_cols, sample_cols, id_col
    )

    # Collect the final unstratified table
    df_unstrat = lf_unstrat_plan.collect(engine="streaming")
    
    # Free remaining large dataframes (Rule 10)
    del lf_func_common, df_norm, lf_unstrat_plan
    gc.collect()

    # Save unstratified table
    with gzip.open(pred_unstrat_path, 'wb') as f:
        df_unstrat.write_csv(f, separator='\t')

    _validate_output(pred_unstrat_path, "Polars Unstratified", "unstratified metagenome predictions")
    print(f"     -> Unstratified predictions saved: {pred_unstrat_path.name}")
    
    # 9. Return paths
    return {
        'seqtab_norm': seqtab_norm_path,
        'pred_metagenome_unstrat': pred_unstrat_path
    }