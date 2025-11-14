from skbio.stats.composition import clr, multi_replace
import polars as pl
import numpy as np
from typing import Dict
from pathlib import Path
import shutil
from pgptracker.utils.profiling_tools.profiler import profile_memory

@profile_memory
def apply_clr(
    input_path: Path,
    input_format: str,
    output_dir: Path,
    base_name: str,
    sample_col: str = "Sample",
    value_col: str = "Total_PGPT_Abundance"
) -> Dict[str, Path]:
    """
    Reads an input file, applies Centered Log-Ratio (CLR) transformation
    based on the specified format, and saves standardized output files.
    
    This function handles I/O operations (read, copy, write) and
    serves as the pre-processing step for analysis.

    Case 1: input_format == 'long'
    - Copies input to 'raw_long_{base_name}'
    - Pivots input to 'raw_wide_{base_name}'
    - Applies CLR to wide format and saves 'clr_wide_{base_name}'

    Case 2: input_format == 'wide'
    - Copies input to 'raw_wide_{base_name}'
    - Applies CLR to wide format and saves 'clr_wide_{base_name}'

    Args:
        input_path: Path to the raw input abundance table.
        input_format: Table format. Must be one of ['wide', 'unstratified', 
                      'long', 'stratified']. 
        output_dir: Directory to save the output files.
        base_name: The original filename (e.g., "my_table.tsv")
                   used to construct output names.
        sample_col (str): Name of the sample column (for 'long' format). 
        value_col (str): Name of the abundance/value column (for 'long' format).

    Returns:
        A dictionary mapping standardized names to the Path
        objects of the created files (e.g., 
        {'raw_wide': Path(...), 'clr_wide': Path(...)}).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}
    
    # Define standard output paths
    raw_wide_path = output_dir / f"raw_wide_{base_name}"
    clr_wide_path = output_dir / f"clr_wide_{base_name}"

    if input_format in ('long', 'stratified'):
        # --- Case 1: Long Input ---
        
        # 1. Define and copy raw_long
        raw_long_path = output_dir / f"raw_long_{base_name}"
        shutil.copyfile(input_path, raw_long_path)
        output_paths['raw_long'] = raw_long_path
        
        # 2. Pivot long to wide
        df_long = pl.read_csv(input_path, separator="\t")
        df_wide = _pivot_long_to_wide(df_long, sample_col, value_col)
        
        # 3. Save raw_wide
        df_wide.write_csv(raw_wide_path, separator="\t")
        output_paths['raw_wide'] = raw_wide_path
        
        # 4. Apply CLR (using the in-memory wide df)
        df_wide_clr = _clr_wide(df_wide)
        
        # 5. Save clr_wide
        df_wide_clr.write_csv(clr_wide_path, separator="\t")
        output_paths['clr_wide'] = clr_wide_path

    elif input_format in ('wide', 'unstratified'):
        # --- Case 2: Wide Input ---

        # 1. Copy raw_wide
        shutil.copyfile(input_path, raw_wide_path)
        output_paths['raw_wide'] = raw_wide_path
        
        # 2. Load the newly copied raw_wide file
        df_wide = pl.read_csv(raw_wide_path, separator="\t")
        
        # 3. Apply CLR
        df_wide_clr = _clr_wide(df_wide)
        
        # 4. Save clr_wide
        df_wide_clr.write_csv(clr_wide_path, separator="\t")
        output_paths['clr_wide'] = clr_wide_path
        
    else:
        raise ValueError(
            f"Invalid format: '{input_format}'. "
            f"Must be one of ['wide', 'unstratified', 'long', 'stratified']"
        )
        
    return output_paths

@profile_memory
def _pivot_long_to_wide(
    df: pl.DataFrame, 
    sample_col: str, 
    value_col: str
) -> pl.DataFrame:
    """
    Internal function: Pivots a long-format DataFrame to wide format.
    
    - Identifies feature columns (non-sample, non-value).
    - Pivots the table on the sample column.
    - Fills missing values (implicit zeros) with 0.0.
    """
    # 1. Identify all feature/taxonomy columns
    non_abundance_cols = [
        c for c in df.columns if c not in [sample_col, value_col]
    ]
    
    # 2. Pivot to wide
    df_wide = df.pivot(
        values=value_col,
        index=non_abundance_cols,
        on=sample_col
    ).fill_null(0.0) 
    
    return df_wide

@profile_memory
def _clr_wide(df: pl.DataFrame) -> pl.DataFrame:
    """
    (Core Engine) 
    Performs CLR on wide/unstratified format (features x samples) data.
    - Identifies numeric (sample) columns.
    - Transposes (D, N) -> (N, D) for scikit-bio.
    - Handles D=1 and all-zero sample edge cases (result = 0). 
    - Applies multi_replace and clr on valid samples.
    - Transposes (N, D) -> (D, N) and reconstructs DataFrame. 
    
    Example (WIDE):
        Input:
            PGPT_ID           Sample_A  Sample_B
            NITROGEN_FIXATION 10        5
            PHOSPHATE_SOL     15        20
        
        Output:
            PGPT_ID           Sample_A   Sample_B
            NITROGEN_FIXATION -0.223144  -0.693147
            PHOSPHATE_SOL      0.223144   0.693147
    """
    
    # 1. Identify columns by dtype
    sample_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    feature_cols = [c for c in df.columns if c not in sample_cols] 
    
    if not sample_cols:
        return df # No numeric columns found

    # (D_features, N_samples)
    abundance_matrix = df.select(sample_cols).to_numpy()
    
    # 2. Transpose to (N_samples, D_features) for skbio
    abundance_matrix_T = abundance_matrix.T
    
    # 3. Get shape (N, D). Handle 1D array edge case.
    if abundance_matrix_T.ndim == 1:
         # This happens if N=1 (single sample). Reshape to (1, D). 
         abundance_matrix_T = abundance_matrix_T.reshape(1, -1)
    
    if abundance_matrix_T.shape[0] == 0:
         clr_matrix_T = abundance_matrix_T # Handle empty matrix
    else:
        _, D_features = abundance_matrix_T.shape

        # 4. Initialize output matrix.
        clr_matrix_T = np.zeros_like(abundance_matrix_T) 
        # 5. GUARD RAIL 1: skbio.clr fails if D=1.
        if D_features > 1: 
            
            # 6. GUARD RAIL 2: skbio.multi_replace fails on all-zero rows.
            valid_rows_mask = np.any(abundance_matrix_T != 0, axis=1) 
            
            if np.any(valid_rows_mask):
                valid_abundance_matrix = abundance_matrix_T[valid_rows_mask]
                
                # 7. Run skbio ONLY on the valid subset (N_valid, D)
                filled_matrix_T = multi_replace(valid_abundance_matrix) 
                valid_clr_matrix_T = clr(filled_matrix_T)
                
                # 8. Place the results back into the correct rows
                clr_matrix_T[valid_rows_mask] = valid_clr_matrix_T

    # 9. Transpose back to (D, N) to match original df structure
    # clr_matrix = clr_matrix_T.T

    # Reconstruct using a dictionary to explicitly map sample names
    # to the correct data columns (which are the rows of clr_matrix_T)
    clr_df = pl.DataFrame(
        dict(zip(sample_cols, clr_matrix_T)))
    
    return pl.concat([df.select(feature_cols), clr_df], how='horizontal')