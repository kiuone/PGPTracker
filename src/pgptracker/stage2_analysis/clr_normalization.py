# src/pgptracker/analysis/transforms.py
"""
Compositional data transformations for microbiome analysis.

Author: Vivian Mello
"""

import polars as pl
import numpy as np
from skbio.stats.composition import clr, multiplicative_replacement as skbio_mr
from typing import Literal


def multiplicative_replacement(
    df: pl.DataFrame, 
    delta: float = 0.65,
    format: Literal['auto', 'wide', 'long'] = 'auto'
) -> pl.DataFrame:
    """
    Replace zeros with multiplicative replacement (Martín-Fernández et al. 2003).
    
    Required before CLR transformation to handle compositionality.
    
    Args:
        df: Abundance table (wide or long format)
        delta: Replacement proportion of minimum nonzero value (default: 0.65)
        format: 'wide', 'long', or 'auto' (auto-detect)
    
    Returns:
        DataFrame with zeros replaced by small positive values
    
    Notes:
        Uses scikit-bio implementation for validated methodology.
        Preserves original data structure (wide stays wide, long stays long).
    """
    detected_format = _detect_format(df) if format == 'auto' else format
    
    if detected_format == 'wide':
        return _mr_wide(df, delta)
    else:
        return _mr_long(df, delta)


def clr_transform_polars(
    df: pl.DataFrame,
    format: Literal['auto', 'wide', 'long'] = 'auto'
) -> pl.DataFrame:
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
    detected_format = _detect_format(df) if format == 'auto' else format
    
    if detected_format == 'wide':
        return _clr_wide(df)
    else:
        return _clr_long(df)


# ==================== INTERNAL FUNCTIONS ====================

def _detect_format(df: pl.DataFrame) -> Literal['wide', 'long']:
    """
    Auto-detect table format based on column names.
    
    Logic:
        - LONG: Contains 'Sample' column (stratified output signature)
        - WIDE: No 'Sample' column (unstratified output signature)
    """
    if 'Sample' in df.columns:
        return 'long'
    return 'wide'


def _mr_wide(df: pl.DataFrame, delta: float) -> pl.DataFrame:
    """
    Apply multiplicative replacement to wide format.
    
    Structure: First column = feature IDs, rest = numeric samples
    """
    feature_col = df.columns[0]
    sample_cols = df.columns[1:]
    
    # Extract abundance matrix (features × samples)
    abundance_matrix = df.select(sample_cols).to_numpy()
    
    # Apply scikit-bio multiplicative replacement
    filled_matrix = skbio_mr(abundance_matrix, delta=delta)
    
    # Reconstruct DataFrame
    filled_df = pl.DataFrame(filled_matrix, schema=sample_cols)
    result = pl.concat([df.select(feature_col), filled_df], how='horizontal')
    
    return result


def _mr_long(df: pl.DataFrame, delta: float) -> pl.DataFrame:
    """
    Apply multiplicative replacement to long format.
    
    Strategy: Pivot → MR → Unpivot
    """
    # Identify columns (assuming standard stratified output)
    abundance_col = 'Total_PGPT_Abundance'
    non_abundance_cols = [c for c in df.columns if c not in ['Sample', abundance_col]]
    
    # Pivot to wide format
    df_wide = df.pivot(
        values=abundance_col,
        index=non_abundance_cols,
        on='Sample'
    ).fill_null(0.0)
    
    # Apply MR on wide format
    df_filled_wide = _mr_wide(df_wide, delta)
    
    # Unpivot back to long
    sample_cols = [c for c in df_filled_wide.columns if c not in non_abundance_cols]
    df_long = df_filled_wide.unpivot(
        index=non_abundance_cols,
        on=sample_cols,
        variable_name='Sample',
        value_name=abundance_col
    )
    
    return df_long


def _clr_wide(df: pl.DataFrame) -> pl.DataFrame:
    """
    CLR transformation for wide format (unstratified).
    
    Strategy:
    1. Apply multiplicative replacement
    2. Extract numeric matrix
    3. Apply CLR (scikit-bio)
    4. Reconstruct DataFrame
    """
    # Step 1: Fill zeros
    df_filled = _mr_wide(df, delta=0.65)
    
    # Step 2: Extract matrix
    feature_col = df.columns[0]
    sample_cols = df.columns[1:]
    abundance_matrix = df_filled.select(sample_cols).to_numpy()
    
    # Step 3: Apply CLR (per sample = per column)
    # scikit-bio CLR expects samples as ROWS, so we transpose
    clr_matrix = clr(abundance_matrix.T).T
    
    # Step 4: Reconstruct
    clr_df = pl.DataFrame(clr_matrix, schema=sample_cols)
    result = pl.concat([df.select(feature_col), clr_df], how='horizontal')
    
    return result


def _clr_long(df: pl.DataFrame) -> pl.DataFrame:
    """
    CLR transformation for long format (stratified).
    
    Strategy: Pivot → CLR (wide) → Unpivot
    Output: Replaces 'Total_PGPT_Abundance' with 'CLR_Abundance'
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
    df_clr_long = df_clr_wide.unpivot(
        index=non_abundance_cols,
        on=sample_cols,
        variable_name='Sample',
        value_name='CLR_Abundance'  # Renamed column
    )
    
    return df_clr_long