"""
Metagenome pipeline runner for PGPTracker.

This module wraps PICRUSt2 metagenome_pipeline.py (Douglas et al., 2020)
to normalize sequence abundances and generate unstratified metagenome predictions.

File originally named normalize.py
"""

from pathlib import Path
from typing import Dict
from pgptracker.utils.env_manager import run_command
import subprocess # Keep subprocess for CalledProcessError
from pgptracker.utils.validator import validate_output_file as _validate_output
import polars as pl
import gzip
from pgptracker.utils.validator import find_asv_column
from pgptracker.utils.profiling_tools.profiler import profile_memory

@profile_memory
def _filter_by_nsti_polars(
    lf: pl.LazyFrame, 
    max_nsti: float, 
    id_col: str
) -> pl.LazyFrame:
    """
    Filters a LazyFrame by max_nsti and drops metadata columns.
    """
    if 'metadata_NSTI' in lf.columns:
        lf = lf.filter(pl.col('metadata_NSTI') <= max_nsti)
        
    # Drop all metadata columns
    meta_cols = [c for c in lf.columns if c.startswith('metadata_') or c == 'closest_reference_genome']
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
    Normalizes ASV abundances by marker gene copy numbers.
    """
    marker_col = [c for c in lf_marker.columns if c not in [id_col]][0]
    
    # Prepare marker df
    lf_marker = lf_marker.select(
        pl.col(id_col),
        # Handle division by zero: if marker_copies is 0, treat it as 1 (no change)
        pl.when(pl.col(marker_col) == 0)
        .then(1.0)
        .otherwise(pl.col(marker_col))
        .alias("marker_copies")
    )

    # Join and normalize all sample columns
    lf_norm = lf_table.join(lf_marker, on=id_col, how="inner")
    
    lf_norm = lf_norm.with_columns(
        # Divide each sample column by marker_copies and round
        (pl.col(s) / pl.col("marker_copies")).round(2)
        for s in sample_cols
    )
    
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
    """
    
    # 1. Unpivot KO predictions (ASV x KO -> ASV, KO, Copy_Num)
    lf_func_long = lf_func.select(
        [id_col] + ko_cols
    ).unpivot(
        index=id_col,
        on=ko_cols,
        variable_name="function",
        value_name="copy_num"
    ).filter(pl.col('copy_num') > 0)

    # 2. Unpivot Normalized Abundances (ASV x Sample -> ASV, Sample, Abundance)
    lf_norm_long = lf_norm.select(
        [id_col] + sample_cols
    ).unpivot(
        index=id_col,
        on=sample_cols,
        variable_name="sample",
        value_name="abundance"
    ).filter(pl.col('abundance') > 0)

    # 3. Join
    # (ASV, Sample, Abundance) JOIN (ASV, KO, Copy_Num)
    lf_joined = lf_norm_long.join(lf_func_long, on=id_col, how="inner")

    # 4. Calculate Contribution
    lf_contrib = lf_joined.with_columns(
        (pl.col("abundance") * pl.col("copy_num")).alias("unstrat_abun")
    )

    # 5. Aggregate (Sum by KO and Sample)
    lf_agg = lf_contrib.group_by(["function", "sample"]).agg(
        pl.col("unstrat_abun").sum()
    )

    # 6. Pivot to final (KO x Sample) table
    lf_pivot = lf_agg.pivot(
        values="unstrat_abun",
        index="function",
        on="sample",
        aggregate_function="sum"
    ).fill_null(0.0)

    return lf_pivot

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
        FileNotFoundError: If any input file doesn't exist
        RuntimeError: If output validation fails or file is empty
        pl.exceptions.ColumnNotFoundError: If expected ASV_ID columns aren't found
    """
    # 1. Validate inputs (Polars will raise FileNotFoundError if missing)
    for p in [table_path, marker_path, ko_predicted_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    # 2. Define output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    metagenome_out_dir = output_dir / "KO_metagenome_out" # Keep PICRUSt2 structure
    metagenome_out_dir.mkdir(parents=True, exist_ok=True)
    
    seqtab_norm_path = metagenome_out_dir / "seqtab_norm.tsv.gz"
    pred_unstrat_path = metagenome_out_dir / "pred_metagenome_unstrat.tsv.gz"

    # --- BIOM Conversion Step ---
    # Polars cannot read BIOM. Convert to TSV if necessary.
    if table_path.suffix == ".biom":
        table_tsv_path = table_path.with_suffix(".tsv")
        print("  -> Converting BIOM to TSV for Polars processing...")
        convert_cmd = [
            "biom", "convert",
            "-i", str(table_path),
            "-o", str(table_tsv_path),
            "--to-tsv"
        ]
        # Use 'qiime' env, which contains 'biom-format' [cite: 246]
        run_command("qiime", convert_cmd, check=True)
        table_path_to_load = table_tsv_path
    else:
        table_path_to_load = table_path

    print("  -> Loading tables with Polars (Lazy)...")
    # 3. Load Data (Lazy)
    # Use 'try_parse_dates=False' for performance and to avoid errors
    lf_table = pl.scan_csv(
        table_path_to_load, separator='\t', comment_prefix='#', try_parse_dates=False
    )
    lf_marker = pl.scan_csv(
        marker_path, separator='\t', comment_prefix='#', try_parse_dates=False
    )
    lf_func = pl.scan_csv(
        ko_predicted_path, separator='\t', comment_prefix='#', try_parse_dates=False
    )

    # 4. Find ASV_ID columns
    # Collect schema once to get column names
    table_schema = lf_table.collect_schema()
    marker_schema = lf_marker.collect_schema()
    func_schema = lf_func.collect_schema()

    id_col_table = find_asv_column(table_schema)
    id_col_marker = find_asv_column(marker_schema)
    id_col_func = find_asv_column(func_schema)

    # Standardize ASV_ID column name
    lf_table = lf_table.rename({id_col_table: "ASV_ID"})
    lf_marker = lf_marker.rename({id_col_marker: "ASV_ID"})
    lf_func = lf_func.rename({id_col_func: "ASV_ID"})
    id_col = "ASV_ID"

    # 5. Filter by NSTI (replicates drop_tips_by_nsti)
    print(f"  -> Filtering ASVs by NSTI <= {max_nsti}...")
    lf_marker_filt = _filter_by_nsti_polars(lf_marker, max_nsti, id_col)
    lf_func_filt = _filter_by_nsti_polars(lf_func, max_nsti, id_col)

    # 6. Get Overlapping ASVs (replicates three_df_index_overlap_sort)
    asvs_marker = lf_marker_filt.select(id_col).unique()
    asvs_func = lf_func_filt.select(id_col).unique()
    asvs_table = lf_table.select(id_col).unique()

    common_asvs = asvs_table.join(
        asvs_marker, on=id_col, how="inner"
    ).join(
        asvs_func, on=id_col, how="inner"
    )

    # Filter all tables to common ASVs
    lf_table_common = lf_table.join(common_asvs, on=id_col, how="inner")
    lf_marker_common = lf_marker_filt.join(common_asvs, on=id_col, how="inner")
    lf_func_common = lf_func_filt.join(common_asvs, on=id_col, how="inner")

    # Get column lists
    sample_cols = [c for c in table_schema.names() if c != id_col_table]
    ko_cols = [c for c in func_schema.names() if c.startswith("ko:")]

    # --- 7. Normalization (replicates norm_by_marker_copies) ---
    print("  -> Normalizing abundances by marker copies...")
    lf_norm = _normalize_by_marker_polars(
        lf_table_common, lf_marker_common, sample_cols, id_col
    )
    
    # Save normalized table
    # Rename ID col for compatibility
    lf_norm_to_save = lf_norm.rename({id_col: "normalized"})
    
    with gzip.open(seqtab_norm_path, 'wb') as f:
        lf_norm_to_save.collect(streaming=True).write_csv(f, separator='\t')
    
    _validate_output(seqtab_norm_path, "Polars Normalization", "normalized sequence table")
    print(f"  -> Normalized table saved: {seqtab_norm_path.name}")

    # --- 8. Unstratified Prediction (replicates unstrat_funcs_only_by_samples) ---
    print("  -> Generating unstratified KO predictions...")
    lf_unstrat = _unstrat_funcs_only_by_samples_polars(
        lf_func_common, lf_norm, ko_cols, sample_cols, id_col
    )

    # Save unstratified table
    with gzip.open(pred_unstrat_path, 'wb') as f:
        lf_unstrat.collect(streaming=True).write_csv(f, separator='\t')

    _validate_output(pred_unstrat_path, "Polars Unstratified", "unstratified metagenome predictions")
    print(f"  -> Unstratified predictions saved: {pred_unstrat_path.name}")
    
    # 9. Return paths
    return {
        'seqtab_norm': seqtab_norm_path,
        'pred_metagenome_unstrat': pred_unstrat_path
    }