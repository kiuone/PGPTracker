from skbio.stats.composition import clr, multi_replace
import polars as pl
import numpy as np
from typing import Dict, Tuple
import sys

# --- [DEBUG] Helper Function ---
def print_df_head(df: pl.DataFrame, name: str):
    """Helper function to print the head (4 rows) and first 7 columns, plus shape."""
    if df is None:
        print(f"\n--- [DEBUG] DataFrame: {name} is None ---")
        return
    
    print(f"\n--- [DEBUG] DataFrame: {name} (Original Shape: {df.shape}) ---")
    
    df_head = df.head(4)
    
    # select the first 7 columns 
    num_cols_to_show = min(7, len(df.columns))
    df_subset = df_head.select(df.columns[:num_cols_to_show])
    
    with pl.Config(tbl_width_chars=200, tbl_cols=-1, tbl_rows=4):
        print(df_subset)
        
    if len(df.columns) > 7:
        print(f"    ... (showing 7 of {len(df.columns)} total columns)")
        
    print(f"--- [DEBUG] End of DataFrame: {name} ---")
    sys.stdout.flush() # Grants the print appears before potential errors/logs
# --- [DEBUG] End Helper Function ---


def apply_clr(
    df: pl.DataFrame, 
    format: str,
    sample_col: str = "Sample",
    value_col: str = "Total_PGPT_Abundance"
) -> Dict[str, pl.DataFrame]:
    """
    Applies Centered Log-Ratio (CLR) transformation
    and returns a dictionary of DataFrames.

    - 'wide' format: Returns {'unstratified_clr': df_wide_clr}
    - 'long' format: Returns {'stratified_wide_clr': df_wide_clr,
                              'stratified_long_clr': df_long_clr}
    
    This allows the CLI to save the 'stratified_wide_clr' output
    while using the 'stratified_long_clr' for other steps.

    Handles both unstratified (wide) and stratified (long) formats
    based on the provided 'format' flag. Applies multiplicative
    replacement for zeros before transformation via skbio.

    Args:
        df: Abundance table (Polars DataFrame).
        format: Table format. Must be one of ['wide', 'unstratified', 
                'long', 'stratified']. 
        sample_col (str): Name of the sample column (for 'long' format).
                          Defaults to "Sample".
        value_col (str): Name of the abundance/value column (for 'long' format).
                         Defaults to "Total_PGPT_Abundance".

    Returns:
        A dictionary mapping output names to CLR-transformed DataFrames.
    """
    print_df_head(df, f"df (input to apply_clr, format='{format}')")
    
    if format in ('wide', 'unstratified'):
        df_wide_clr = _clr_wide(df)
        print_df_head(df_wide_clr, "df_wide_clr (returned by _clr_wide)")
        return {'unstratified_clr': df_wide_clr}
    
    elif format in ('long', 'stratified'):
        # _clr_long now returns both wide and long versions
        df_wide_clr, df_long_clr = _clr_long(df, sample_col, value_col)
        
        print_df_head(df_wide_clr, "df_wide_clr (returned by _clr_long)")
        print_df_head(df_long_clr, "df_long_clr (returned by _clr_long)")
        
        return {
            'stratified_wide_clr': df_wide_clr,
            'stratified_long_clr': df_long_clr
        }
    
    else:
        raise ValueError(
            f"Invalid format: '{format}'. "
            f"Must be one of ['wide', 'unstratified', 'long', 'stratified']"
        )

def _clr_long(
    df: pl.DataFrame, 
    sample_col: str, 
    value_col: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Internal function: Pivots long to wide, applies CLR, 
    and unpivots back to long.
    
    Returns a tuple containing: (df_wide_clr, df_long_clr)

    - Pivots long data to wide creating zeros from missing values.
    - Calls _clr_wide() to perform the transformation (which handles zeros).
    - Unpivots data back to long format with 'CLR_Abundance' column.
    
      Example (LONG):
        Input:
            Order    Lv3                Sample        Total_PGPT_Abundance
            Bacilli  NITROGEN_FIXATION  Sample_A      25.0
            Bacilli  PHOSPHATE_SOL      Sample_A      15.0
        
        Output:
            Order    Lv3                Sample        CLR_Abundance
            Bacilli  NITROGEN_FIXATION  Sample_A      0.223144
            Bacilli  PHOSPHATE_SOL      Sample_A     -0.223144
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
    print_df_head(df_wide, "df_wide (after pivot in _clr_long)")
    
    # 3. Apply CLR (delegated to _clr_wide)
    # This IS the 'stratified_wide_clr' output
    df_wide_clr = _clr_wide(df_wide)
    print_df_head(df_wide_clr, "df_wide_clr (returned from _clr_wide in _clr_long)")
    
    # 4. Identify sample columns that were created by the pivot
    sample_cols_in_wide = [
        c for c in df_wide_clr.columns if c not in non_abundance_cols
    ] 
    
    # 5. Create dynamic output name
    new_value_name = f"CLR_{value_col}"

    # 6. Unpivot back to create the long version
    df_long_clr = df_wide_clr.unpivot(
        index=non_abundance_cols,
        on=sample_cols_in_wide,
        variable_name=sample_col, 
        value_name=new_value_name
    )
    print_df_head(df_long_clr, "df_long_clr (after unpivot in _clr_long)")
    
    # 7. Return BOTH dataframes
    return df_wide_clr, df_long_clr

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
            PGPT_ID            Sample_A  Sample_B
            NITROGEN_FIXATION  10        5
            PHOSPHATE_SOL      15        20
        
        Output:
            PGPT_ID            Sample_A   Sample_B
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
    # THIS IS THE CRITICAL FIX for the 0.6019... bug
    abundance_matrix_T = abundance_matrix.T
    
    # 3. Get shape (N, D). Handle 1D array edge case.
    if abundance_matrix_T.ndim == 1:
        # This happens if N=1 (single sample). Reshape to (1, D).
        abundance_matrix_T = abundance_matrix_T.reshape(1, -1)
    
    if abundance_matrix_T.shape[0] == 0:
         clr_matrix_T = abundance_matrix_T # Handle empty matrix
    else:
        N_samples, D_features = abundance_matrix_T.shape

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
    # clr_matrix = clr_matrix_T.T # This variable is not used

    # Reconstruct using a dictionary to explicitly map sample names
    # to the correct data columns (which are the rows of clr_matrix_T)
    clr_df = pl.DataFrame(
        dict(zip(sample_cols, clr_matrix_T)))
    print_df_head(clr_df, "clr_df (reconstructed from numpy in _clr_wide)")
    
    final_df = pl.concat([df.select(feature_cols), clr_df], how='horizontal')
    print_df_head(final_df, "final_df (concatenated in _clr_wide)")
    
    return final_df