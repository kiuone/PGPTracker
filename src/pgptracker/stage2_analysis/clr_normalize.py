from skbio.stats.composition import clr, multi_replace
import polars as pl
import numpy as np
from typing import Dict

def apply_clr(
    df: pl.DataFrame, 
    format: str,
    sample_col: str = "Sample",
    value_col: str = "Total_PGPT_Abundance"
) -> Dict[str, pl.DataFrame]:
    """
    Applies Centered Log-Ratio (CLR) transformation
    and returns a dictionary of WIDE DataFrames.

    - 'wide' format: Returns {'unstratified_clr': df_wide_clr}
    - 'long' format: Pivots to wide, applies CLR, and
                      Returns {'stratified_wide_clr': df_wide_clr}

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
    if format in ('wide', 'unstratified'):
        df_wide_clr = _clr_wide(df)
        return {'unstratified_clr': df_wide_clr}
    
    elif format in ('long', 'stratified'):
        # _clr_long now returns only the wide version
        df_wide_clr = _clr_long(df, sample_col, value_col)

        # Return the output 'wide' for the stratified
        return {
            'stratified_wide_clr': df_wide_clr,
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
) -> pl.DataFrame:
    """
    Internal function: Pivots long to wide and applies CLR.

    Returns the df_wide_clr DataFrame.

   - Pivots long data to wide creating zeros from missing values.
   - Calls _clr_wide() to perform the transformation (which handles zeros).
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
    
    # 3. Apply CLR (delegated to _clr_wide)
    # This IS the 'stratified_wide_clr' output
    df_wide_clr = _clr_wide(df_wide)

    # [UNPIVOT STEP REMOVED]

    # 4. Return ONLY the wide transformed dataframe
    return df_wide_clr

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
    clr_matrix = clr_matrix_T.T

    # Reconstruct using a dictionary to explicitly map sample names
    # to the correct data columns (which are the rows of clr_matrix_T)
    clr_df = pl.DataFrame(
        dict(zip(sample_cols, clr_matrix_T)))
    
    return pl.concat([df.select(feature_cols), clr_df], how='horizontal')