from skbio.stats.composition import clr, multi_replace
import polars as pl
import numpy as np


def clr_transform(df: pl.DataFrame) -> pl.DataFrame:
    """
    Centered Log-Ratio transformation for compositional data.
    
    Handles BOTH unstratified (wide) and stratified (long) formats.
    Automatically applies multiplicative_replacement before transformation.
    
    Args:
        df: Abundance table
            UNSTRATIFIED (wide): PGPT_ID × Samples
            STRATIFIED (long): [Taxon, PGPT, Sample, Abundance]
        format: 'wide', 'long', or 'auto' (auto-detect based on columns)
    
    Returns:
        CLR-transformed DataFrame (same format as input)
    
    Example (WIDE):
        Input:
            PGPT_ID           Sample_A  Sample_B
            NITROGEN_FIXATION 10        5
            PHOSPHATE_SOL     15        20
        
        Output:
            PGPT_ID           Sample_A   Sample_B
            NITROGEN_FIXATION -0.223144  -0.693147
            PHOSPHATE_SOL      0.223144   0.693147
    
    Example (LONG):
        Input:
            Order    Lv3                Sample           Total_PGPT_Abundance
            Bacilli  NITROGEN_FIXATION  Sample_A         25.0
            Bacilli  PHOSPHATE_SOL      Sample_A         15.0
        
        Output:
            Order    Lv3                Sample           CLR_Abundance
            Bacilli  NITROGEN_FIXATION  Sample_A         0.223144
            Bacilli  PHOSPHATE_SOL      Sample_A        -0.223144
    
    Notes:
        - Uses scikit-bio validated CLR implementation
        - Converts Polars → numpy → Polars (unavoidable for matrix operations)
        - Memory overhead: ~2-3x data size during conversion (acceptable for <1GB data)
    """
    # Detect format inline
    if 'Sample' in df.columns:
        return _clr_long(df)
    return _clr_wide(df)


def _clr_wide(df: pl.DataFrame) -> pl.DataFrame:
    """CLR for wide format (unstratified)."""
    feature_col = df.columns[0]
    sample_cols = df.columns[1:]
    
    abundance_matrix = df.select(sample_cols).to_numpy()
    filled_matrix = multi_replace(abundance_matrix)
    clr_matrix = clr(filled_matrix.T).T
    
    clr_df = pl.DataFrame(clr_matrix, schema=sample_cols)
    return pl.concat([df.select(feature_col), clr_df], how='horizontal')


def _clr_long(df: pl.DataFrame) -> pl.DataFrame:
    """CLR for long format (stratified): Pivot → CLR → Unpivot."""
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