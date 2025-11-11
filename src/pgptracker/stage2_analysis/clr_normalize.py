from skbio.stats.composition import clr, multi_replace
import polars as pl
import numpy as np

def detect_tbl_format(df: pl.DataFrame, format: str) -> pl.DataFrame:
    """
    Centered Log-Ratio transformation for compositional data.
    
    Handles BOTH unstratified (wide) and stratified (long) formats.
    Automatically applies multiplicative_replacement before transformation.
    
    Args:
        df: Abundance table (Polars DataFrame).
        format: Table format. Must be one of ['wide', 'unstratified', 
                'long', 'stratified'].
    Returns:
        A Polars DataFrame with CLR-transformed values.
        - Wide format: Sample columns are replaced.
        - Long format: 'Total_PGPT_Abundance' is replaced with 
                        'CLR_Abundance'.
    
    Notes:
        - Uses scikit-bio validated CLR implementation
        - Converts Polars → numpy → Polars (unavoidable for matrix operations)
    """
    # Detect format of table
    if format in ('wide', 'unstratified'):
        return _clr_wide(df)
    elif format in ('long', 'stratified'):
        return _clr_long(df)
    else:
        raise ValueError(
            f"Invalid format: '{format}'. \nMust be one of ['wide', 'unstratified', 'long', 'stratified']")

def _clr_wide(df: pl.DataFrame) -> pl.DataFrame:
    """
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
        # This happens if D=1 (single feature)
        abundance_matrix_T = abundance_matrix_T.reshape(-1, 1)
    
    if abundance_matrix_T.shape[0] == 0:
         clr_matrix_T = abundance_matrix_T # Handle empty matrix
    else:
        N_samples, D_features = abundance_matrix_T.shape

        # 4. Initialize output matrix.
        # CLR for D=1 or all-zero-sample is 0, so zeros is a safe default.
        clr_matrix_T = np.zeros_like(abundance_matrix_T)

        # 5. GUARD RAIL 1: skbio.clr fails if D=1.
        # The result is 0, which is already set.
        if D_features > 1:
            
            # 6. GUARD RAIL 2: skbio.multi_replace fails on all-zero rows.
            # Find rows (samples) that are NOT all-zero.
            valid_rows_mask = np.any(abundance_matrix_T != 0, axis=1)
            
            if np.any(valid_rows_mask):
                # Subset the matrix to only valid rows
                valid_abundance_matrix = abundance_matrix_T[valid_rows_mask]
                
                # 7. Run skbio ONLY on the valid subset (N_valid, D)
                # This is the correct place for multi_replace
                filled_matrix_T = multi_replace(valid_abundance_matrix)
                valid_clr_matrix_T = clr(filled_matrix_T)
                
                # 8. Place the results back into the correct rows
                clr_matrix_T[valid_rows_mask] = valid_clr_matrix_T

    # 9. Transpose back to (D, N) to match original df structure
    clr_matrix = clr_matrix_T.T
    
    clr_df = pl.DataFrame(clr_matrix, schema=sample_cols)
    return pl.concat([df.select(feature_cols), clr_df], how='horizontal')

def _clr_long(df: pl.DataFrame) -> pl.DataFrame:
    """CLR for long format (stratified): Pivot → CLR → Unpivot.

    - Pivots long data to wide (creating zeros from missing values).
    - Calls _clr_wide() to perform the transformation (which handles zeros).
    - Unpivots data back to long format with 'CLR_Abundance' column.
    
     Example (LONG):
        Input:
            Order    Lv3                Sample           Total_PGPT_Abundance
            Bacilli  NITROGEN_FIXATION  Sample_A         25.0
            Bacilli  PHOSPHATE_SOL      Sample_A         15.0
        
        Output:
            Order    Lv3                Sample           CLR_Abundance
            Bacilli  NITROGEN_FIXATION  Sample_A         0.223144
            Bacilli  PHOSPHATE_SOL      Sample_A        -0.223144
    """
    abundance_col = 'Total_PGPT_Abundance'
    non_abundance_cols = [c for c in df.columns if c not in ['Sample', abundance_col]]
    
    # Pivot to wide
    df_wide = df.pivot(
        values=abundance_col,
        index=non_abundance_cols,
        on='Sample'
    ).fill_null(0.0)
    
    # Apply CLR
    df_clr_wide = _clr_wide(df_wide)
    
    # Unpivot back
    sample_cols = [c for c in df_clr_wide.columns if c not in non_abundance_cols]
    return df_clr_wide.unpivot(
        index=non_abundance_cols,
        on=sample_cols,
        variable_name='Sample',
        value_name='CLR_Abundance')