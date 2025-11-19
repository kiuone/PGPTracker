# clr_normalize.py

from skbio.stats.composition import clr, multi_replace
import polars as pl
import numpy as np
from typing import Dict, List
from pathlib import Path
import shutil
import gc
from pgptracker.utils.profiling_tools.profiler import profile_memory

@profile_memory
def apply_clr(
    input_path: Path,
    input_format: str,
    output_dir: Path,
    base_name: str,
    export_sparcc_format: bool = False,
    keep_feature_cols_separate: bool = False,
    
    # --- Explicit params for 'long' format ---
    long_sample_col: str = "Sample",
    long_value_col: str = "Total_PGPT_Abundance",
    
    # --- Explicit params for 'wide' format  ---
    wide_orientation: str = "D_N",
    wide_id_col: str = "Lv3"

) -> Dict[str, Path]:
    """
    Reads an input file, applies Centered Log-Ratio (CLR) transformation,
    and exports standardized multi-format outputs for downstream analyses.

    This script relies on explicit parameters, not heuristics, to determine
    data orientation and column roles.

    Args:
        input_path: Path to the raw input abundance table
        input_format: Must be one of ['long', 'stratified', 'wide', 'unstratified']
        output_dir: Directory to save output files
        base_name: Original filename for output naming
        export_sparcc_format: If True, exports raw_wide_D_N for SparCC
        keep_feature_cols_separate: If True, exports clr_wide_D_N_split for Tensor
        
        long_sample_col: For 'long' format, the name of the sample column.
        long_value_col: For 'long' format, the name of the abundance column.
        
        wide_orientation: For 'wide' format, the orientation ('D_N' or 'N_D').
        wide_id_col: For 'wide' format, the name of the ID column
                     (e.g., 'Lv3' for D_N, 'Sample' for N_D).

    Returns:
        Dictionary mapping format names to Path objects
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}
    
    if input_format in ('long', 'stratified'):
        # --- Case 1: Long Input ---
        raw_long_path = output_dir / f"raw_long_{base_name}.tsv"
        shutil.copyfile(input_path, raw_long_path)
        output_paths['raw_long'] = raw_long_path
        
        df_long = pl.read_csv(input_path, separator="\t")
        
        # Combine feature columns (e.g., Family, Lv3) into 'FeatureID'
        df_long_combined = _combine_feature_columns(
            df_long, 
            sample_col=long_sample_col, 
            value_col=long_value_col)
        del df_long  # Free memory
        
        # Pivot to N×D (samples × features)
        df_wide_N_D = _pivot_long_to_wide_N_D(
            df_long_combined, 
            sample_col=long_sample_col, 
            value_col=long_value_col)
        del df_long_combined  # Free memory
        gc.collect()

    elif input_format in ('wide', 'unstratified'):
        # --- Case 2: Wide Input (Deterministico) ---
        df_wide = pl.read_csv(input_path, separator="\t")

        # Identify ID col for sanitization
        target_id_col = wide_id_col
        
        # If the ID columns exists, we remove '|' to avoid errors in Tensor split
        if target_id_col in df_wide.columns:
             df_wide = df_wide.with_columns(
                pl.col(target_id_col)
                .cast(pl.String)
                .str.replace_all(r"\|", "_")) # Substitute '|' with '_'
        
        if wide_orientation == "D_N":
            # Input is D×N (e.g., 'Lv3' | S1 | S2)
            # Transpose to N×D (e.g., 'Sample' | Feat1 | Feat2)
            df_wide_N_D = _transpose_D_N_to_N_D(df_wide, id_col_name=wide_id_col)
            
        elif wide_orientation == "N_D":
            # Input is already N×D (e.g., 'Sample' | Feat1 | Feat2)
            # Rename ID col to 'Sample' for consistency
            if wide_id_col != "Sample":
                df_wide_N_D = df_wide.rename({wide_id_col: "Sample"})
            else:
                df_wide_N_D = df_wide
            
        else:
            raise ValueError(
                f"Invalid 'wide_orientation': {wide_orientation}. "
                f"Must be 'D_N' or 'N_D'." )
        
        del df_wide # Free memory
        gc.collect()
        
    else:
        raise ValueError(
            f"Invalid format: '{input_format}'. "
            f"Must be one of ['long', 'stratified', 'wide', 'unstratified']")
    
    # --- Process and Export N×D Data ---
    # This logic is shared for both long and wide inputs
    output_paths.update(
        _process_and_export_N_D(
            df_wide_N_D=df_wide_N_D,
            output_dir=output_dir,
            base_name=base_name,
            export_sparcc_format=export_sparcc_format,
            keep_feature_cols_separate=keep_feature_cols_separate))
    
    del df_wide_N_D  # Free memory
    gc.collect()
        
    return output_paths

@profile_memory
def _process_and_export_N_D(
    df_wide_N_D: pl.DataFrame,
    output_dir: Path,
    base_name: str,
    export_sparcc_format: bool,
    keep_feature_cols_separate: bool
) -> Dict[str, Path]:
    """
    Handles all processing (CLR) and exporting from the N×D table.
    This function is extracted to avoid duplicating logic.
    """
    output_paths: Dict[str, Path] = {}
    
    # 1. Save raw_wide_N_D
    raw_wide_N_D_path = output_dir / f"raw_wide_N_D_{base_name}.tsv"
    df_wide_N_D.write_csv(raw_wide_N_D_path, separator="\t")
    output_paths['raw_wide_N_D'] = raw_wide_N_D_path
    
    # 2. Apply CLR to N×D
    df_clr_wide_N_D = _clr_wide_N_D(df_wide_N_D)
    
    # 3. Save clr_wide_N_D
    clr_wide_N_D_path = output_dir / f"clr_wide_N_D_{base_name}.tsv"
    df_clr_wide_N_D.write_csv(clr_wide_N_D_path, separator="\t")
    output_paths['clr_wide_N_D'] = clr_wide_N_D_path
    
    # 4. Optional: Split CLR table for Tensor Decomposition
    if keep_feature_cols_separate:
        df_clr_split_D_N = _create_D_N_split_table(df_clr_wide_N_D)
        
        clr_split_path = output_dir / f"clr_wide_D_N_split_{base_name}.tsv"
        df_clr_split_D_N.write_csv(clr_split_path, separator="\t")
        output_paths['clr_wide_D_N_split'] = clr_split_path
        
        del df_clr_split_D_N  # Free memory
    
    # 5. Optional: Transpose raw table to D×N for SparCC
    if export_sparcc_format:
        df_wide_D_N = _transpose_N_D_to_D_N(df_wide_N_D)
        
        raw_wide_D_N_path = output_dir / f"raw_wide_D_N_{base_name}.tsv"
        df_wide_D_N.write_csv(raw_wide_D_N_path, separator="\t")
        output_paths['raw_wide_D_N'] = raw_wide_D_N_path
        
        del df_wide_D_N  # Free memory

    del df_clr_wide_N_D  # Free memory
    gc.collect()
    
    return output_paths

@profile_memory
def _combine_feature_columns(
    df: pl.DataFrame,
    sample_col: str,
    value_col: str
) -> pl.DataFrame:
    """
    Combines all columns that are NOT sample_col or value_col into 'FeatureID'.

    Args:
        df: Input long-format DataFrame
        sample_col: Name of the sample column
        value_col: Name of the value/abundance column

    Returns:
        DataFrame with 'FeatureID', sample_col, and value_col.

    Example:
        Input: (sample_col='Sample', value_col='Abundance')
            Family | Lv3      | Sample | Abundance
            -------|----------|--------|-----------
            Fam1   | NITROGEN | S1     | 10
            Fam1   | NITROGEN | S2     | 5
        
        Output:
            FeatureID     | Sample | Abundance
            --------------|--------|-----------
            Fam1|NITROGEN | S1     | 10
            Fam1|NITROGEN | S2     | 5
    """
    feature_cols = [c for c in df.columns if c not in [sample_col, value_col]]
    
    if len(feature_cols) == 0:
        raise ValueError(
            f"No feature columns found. Input columns: {df.columns}. "
            f"Expected columns other than '{sample_col}' and '{value_col}'.")
    
    if len(feature_cols) == 1:
        # Already single column, just rename
        return df.rename({feature_cols[0]: 'FeatureID'})
    
    # Remove any '|' characters in feature columns to avoid confusion and replace with '_'
    df = df.with_columns([
        pl.col(c).cast(pl.String).str.replace_all(r"\|", "_") 
        for c in feature_cols])
    
    # Multiple feature columns - combine with '|' separator
    df = df.with_columns(
        pl.concat_str(feature_cols, separator='|').alias('FeatureID'))
    
    # Keep only FeatureID, Sample, Value
    return df.select(['FeatureID', sample_col, value_col])

@profile_memory
def _pivot_long_to_wide_N_D(
    df: pl.DataFrame,
    sample_col: str,
    value_col: str
) -> pl.DataFrame:
    """
    Pivots long format to wide N×D (samples × features).

    Args:
        df: Long-format DataFrame with 'FeatureID', sample_col, value_col
        sample_col: Name of the sample column
        value_col: Name of the value/abundance column

    Returns:
        Pivoted N×D DataFrame.

    Example:
        Input (long):
            FeatureID     | Sample | Abundance
            --------------|--------|-----------
            Fam1|NITROGEN | S1     | 10
            Fam1|NITROGEN | S2     | 5
            Fam2|PHOSPHATE | S1     | 15
        
        Output (N×D):
            Sample | Fam1|NITROGEN | Fam2|PHOSPHATE
            -------|---------------|---------------
            S1     | 10            | 15
            S2     | 5             | 0
    """
    df_wide = df.pivot(
        values=value_col,
        index=sample_col,      # Samples become ROWS
        on='FeatureID',        # Features become COLUMNS
        aggregate_function="sum" # Explicitly sum duplicates
    ).fill_null(0.0)
    
    # Rename index column to 'Sample' for consistency
    if sample_col != "Sample":
        df_wide = df_wide.rename({sample_col: "Sample"})
        
    return df_wide

@profile_memory
def _transpose_D_N_to_N_D(df: pl.DataFrame, id_col_name: str) -> pl.DataFrame:
    """
    Transposes D×N (features × samples) to N×D (samples × features).
    VALIDATION: Ensures all non-ID columns are numeric.

    Args:
        df: DataFrame in D×N format
        id_col_name: The name of the column holding feature IDs (e.g., "Lv3")

    Returns:
        Transposed N×D DataFrame with "Sample" as the first column.

    Example:
        Input (D×N): (id_col_name='FeatureID')
            FeatureID | S1 | S2
            ----------|----|----
            Feat1     | 10 | 5
            Feat2     | 15 | 20
        
        Output (N×D):
            Sample | Feat1 | Feat2
            -------|-------|-------
            S1     | 10    | 15
            S2     | 5     | 20
    """
    if id_col_name not in df.columns:
        raise ValueError(
            f"ID column '{id_col_name}' not found in wide (D_N) table. "
            f"Available columns: {df.columns}"
        )
        
    # Sample names are the column headers
    sample_names = [c for c in df.columns if c != id_col_name]
    
    # --- VALIDATION (Fix 1) ---
    # Ensure all detected sample columns are actually numeric
    non_numeric_samples = [c for c in sample_names if not df[c].dtype.is_numeric()]
    if non_numeric_samples:
        raise ValueError(
            f"Non-numeric metadata columns found in D×N table: {non_numeric_samples}. "
            f"Only the ID column ('{id_col_name}') and numeric sample columns are allowed.")
    # --- End Validation ---
    
    # Feature names are the values in the ID column
    feature_names = df.select(id_col_name).to_series().to_list()
    
    # Extract numeric data
    numeric_data = df.select(sample_names).to_numpy()
    
    # Transpose: (D, N) → (N, D)
    transposed_data = numeric_data.T
    
    # Create N×D DataFrame
    df_N_D = pl.DataFrame(
        data=transposed_data,
        schema=feature_names,
        orient="row"
    )
    
    # Add Sample column
    df_N_D = df_N_D.insert_column(0, pl.Series("Sample", sample_names))
    
    return df_N_D

@profile_memory
def _clr_wide_N_D(df: pl.DataFrame) -> pl.DataFrame:
    """
    Applies CLR transformation to wide N×D format (samples × features).
    Assumes first non-numeric column is the ID column (e.g., 'Sample').

    Args:
        df: DataFrame in N×D format (samples × features)

    Returns:
        CLR-transformed N×D DataFrame

    Example (N×D input):
        Input:
            Sample | Feat1 | Feat2
            -------|-------|-------
            S1     | 10    | 15
            S2     | 5     | 20
        
        Output:
            Sample | Feat1     | Feat2
            -------|-----------|-----------
            S1     | -0.223144 | 0.223144
            S2     | -0.693147 | 0.693147
    """
    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    id_cols = [c for c in df.columns if c not in numeric_cols]
    
    if not id_cols:
        raise ValueError("No ID column (e.g., 'Sample') found in N×D table.")
    
    if not numeric_cols:
        return df # No data to transform
    
    # Extract numeric data: (N_samples, D_features)
    abundance_matrix = df.select(numeric_cols).to_numpy()
    N, D = abundance_matrix.shape
    
    if N == 0:
        return df
    
    # Apply CLR row-wise (axis=1)
    clr_matrix = np.zeros_like(abundance_matrix, dtype=float)
    
    # Guard rail: skbio.clr fails if D=1
    if D > 1:
        # Guard rail: skbio.multi_replace fails on all-zero rows
        valid_rows_mask = np.any(abundance_matrix != 0, axis=1)
        
        if np.any(valid_rows_mask):
            valid_abundance_matrix = abundance_matrix[valid_rows_mask]
            
            # multi_replace handles 0s by replacing them with a small delta
            filled_matrix = multi_replace(valid_abundance_matrix)
            
            # clr applies transformation
            valid_clr_matrix = clr(filled_matrix)
            clr_matrix[valid_rows_mask] = valid_clr_matrix
    
    # Reconstruct DataFrame
    clr_df = pl.DataFrame(
        data=clr_matrix,
        schema=numeric_cols,
        orient="row")
    
    return pl.concat([df.select(id_cols), clr_df], how='horizontal')

@profile_memory
def _transpose_N_D_to_D_N(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transposes N×D (samples × features) to D×N (features × samples).
    Assumes the first non-numeric column is 'Sample'.
    NOTE: This function assumes 'Sample' column values are unique,
    which is guaranteed if the N×D table was created by _pivot_long_to_wide_N_D.

    Args:
        df: DataFrame in N×D format

    Returns:
        Transposed D×N DataFrame

    Example:
        Input (N×D):
            Sample | Feat1 | Feat2
            -------|-------|-------
            S1     | 10    | 15
            S2     | 5     | 20
        
        Output (D×N):
            FeatureID | S1 | S2
            ----------|----|----
            Feat1     | 10 | 5
            Feat2     | 15 | 20
    """
    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    id_cols = [c for c in df.columns if c not in numeric_cols]

    if not id_cols:
        raise ValueError("No ID column (e.g., 'Sample') found in N×D table.")
        
    id_col = id_cols[0] # Use the first ID column
    
    # Extract sample IDs (from 'Sample' column values) and feature names (column headers)
    sample_ids = df.select(id_col).to_series().to_list()
    feature_names = numeric_cols
    
    # Extract numeric data
    numeric_data = df.select(numeric_cols).to_numpy()
    
    # Transpose: (N, D) → (D, N)
    transposed_data = numeric_data.T
    
    # Create D×N DataFrame
    df_D_N = pl.DataFrame(
        data=transposed_data,
        schema=sample_ids,
        orient="row" )
    
    # Add FeatureID column
    df_D_N = df_D_N.insert_column(0, pl.Series("FeatureID", feature_names))
    
    return df_D_N

@profile_memory
def _create_D_N_split_table(df_N_D: pl.DataFrame) -> pl.DataFrame:
    """
    Creates D×N table with split feature columns for Tensor Decomposition.
    VALIDATION: Ensures all FeatureIDs have a consistent number of split levels.

    Args:
        df_N_D: An N×D DataFrame (e.g., the CLR-transformed table)

    Returns:
        A D×N DataFrame with 'FeatureID' split into 'Feature_Level_...'

    Example:
        Input (N×D):
            Sample | Fam1|NITROGEN | Fam2|PHOSPHATE
            -------|---------------|---------------
            S1     | -0.22         | 0.22
            S2     | -0.69         | 0.69
        
        Output (D×N Split):
            Feature_Level_1 | Feature_Level_2 | S1    | S2
            ----------------|-----------------|-------|-------
            Fam1            | NITROGEN        | -0.22 | -0.69
            Fam2            | PHOSPHATE       | 0.22  | 0.69
    """
    # 1. Transpose N×D -> D×N
    df_D_N = _transpose_N_D_to_D_N(df_N_D)
    
    if df_D_N.height == 0:
        return pl.DataFrame() # Return empty if no data
    
    feature_ids = df_D_N.get_column("FeatureID")
        
    # 2. Check if splitting is needed
    if not feature_ids.str.contains(r"\|").any():
        # No separator found, return the D_N table as-is
        return df_D_N

    # 3. --- VALIDATION (Fix 2) ---
    # Check that all features have the same number of levels
    split_lengths = feature_ids.str.split("|").list.len().unique()
    
    if split_lengths.len() > 1:
        raise ValueError(
            f"Inconsistent FeatureID levels detected. "
            f"Found features with {split_lengths.to_list()} parts. "
            f"All features must have the same number of '|' separators."
        )
    # --- End Validation ---
    n_levels = split_lengths[0]
    level_cols = [f"Feature_Level_{i+1}" for i in range(n_levels)]

    # 4. Split the 'FeatureID' column
    df_split = df_D_N.with_columns(
        pl.col("FeatureID").str.split("|").alias("_split_features"))

    # 5. Create new feature level columns
    for i in range(n_levels):
        df_split = df_split.with_columns(
            pl.col("_split_features").list.get(i).alias(level_cols[i]) )
    
    # 6. Select final columns in order
    sample_cols = [c for c in df_D_N.columns if c != "FeatureID"]
    df_final = df_split.select(level_cols + sample_cols)
    
    return df_final