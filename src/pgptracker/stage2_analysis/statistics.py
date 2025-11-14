# -*- coding: utf-8 -*-
"""
Performs statistical comparisons (Kruskal-Wallis, Mann-Whitney U),
calculates effect sizes (Cliff's Delta), and applies FDR correction.

These functions are designed to operate on LONG-FORMAT data,
(e.g., 'stratified_long_clr') as it is the most efficient
format for grouped statistical tests.
"""

import polars as pl
import numpy as np
import scipy.stats as sp_stats
from statsmodels.stats.multitest import multipletests
from typing import List, Dict, Optional, Any, Tuple

def _kruskal_helper(df_group: pl.DataFrame, group_col: str, value_col: str) -> pl.DataFrame:
    """Internal helper function to be called by group_by.apply()"""
    
    # 1. Aggregate values into lists, one list per group
    value_lists = df_group.group_by(group_col).agg(
        pl.col(value_col)
    )[value_col].to_list()

    # 2. Convert each list of values into a numpy array
    group_data = [np.array(values) for values in value_lists]
    
    # 3. Unpack the list of arrays into scipy.stats.kruskal
    stat, pval = sp_stats.kruskal(*group_data)
    return pl.DataFrame({'statistic': [stat], 'p_value': [pval]})

def kruskal_wallis_test(
    df_long: pl.DataFrame,
    feature_col: str,
    group_col: str,
    value_col: str
) -> pl.DataFrame:
    """
    Performs Kruskal-Wallis H-test for all features.

    Args:
        df_long: A LONG-format DataFrame (e.g., 'stratified_long_clr').
        feature_col: The feature to group by (e.g., 'Lv3', 'PGPT_ID').
        group_col: The metadata column with groups (e.g., 'Treatment').
        value_col: The abundance column to test (e.g., 'CLR_Abundance').

    Returns:
        A DataFrame with [feature_col, statistic, p_value].
    """
    print(f"Running Kruskal-Wallis on '{feature_col}' grouped by '{group_col}'...")
    
    # Group by the feature (e.g., 'PGPT_ID'), then apply the test
    # to the subgroups (e.g., 'Treatment' A, B, C)
    results = df_long.group_by(feature_col).apply( #type: ignore[arg-type]
        lambda df_group: _kruskal_helper(df_group, group_col, value_col)) 
    
    return results.sort('p_value')

def mann_whitney_u_test(
    df_long: pl.DataFrame,
    feature_col: str,
    group_col: str,
    value_col: str,
    group_1: str,
    group_2: str
) -> pl.DataFrame:
    """
    Performs Mann-Whitney U-test for all features between two specific groups.

    Args:
        df_long: A LONG-format DataFrame.
        feature_col: The feature to group by (e.g., 'Lv3').
        group_col: The metadata column with groups (e.g., 'Treatment').
        value_col: The abundance column to test (e.g., 'CLR_Abundance').
        group_1: The name of the first group.
        group_2: The name of the second group.

    Returns:
        A DataFrame with [feature_col, statistic, p_value].
    """
    print(f"Running Mann-Whitney U between '{group_1}' and '{group_2}'...")
    
    # Filter data to only include the two groups of interest
    df_filtered = df_long.filter(pl.col(group_col).is_in([group_1, group_2]))

    def _mw_helper(df_group: pl.DataFrame) -> pl.DataFrame:
        g1_data = df_group.filter(pl.col(group_col) == group_1)[value_col].to_numpy()
        g2_data = df_group.filter(pl.col(group_col) == group_2)[value_col].to_numpy()
        
        if len(g1_data) == 0 or len(g2_data) == 0:
            return pl.DataFrame({'statistic': [np.nan], 'p_value': [np.nan]})
            
        stat, pval = sp_stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
        return pl.DataFrame({'statistic': [stat], 'p_value': [pval]})

    results = df_filtered.group_by(feature_col).apply(_mw_helper) #type: ignore[arg-type]
    return results.sort('p_value')

def fdr_correction(
    pvalues: pl.Series,
    method: str = 'fdr_bh'
) -> pl.Series:
    """
    Applies False Discovery Rate correction to a Series of p-values.

    Args:
        pvalues: A Polars Series containing p-values.
        method: Correction method (from statsmodels). 'fdr_bh' is default.

    Returns:
        A Polars Series containing the corrected p-values (q-values).
    """
    # 1. Create a DataFrame with original p-values and a row index
    df = pvalues.to_frame().with_row_count("idx")
    
    # 2. Filter to get only non-null p-values and their original indices
    df_filtered = df.filter(pl.col(pvalues.name).is_not_null())
    
    # 3. Extract p-values to correct
    pvals_to_correct = df_filtered[pvalues.name].to_numpy()
    
    if len(pvals_to_correct) == 0:
        # Handle empty or all-null input
        return pl.Series(name="q_value", values=[None] * len(pvalues), dtype=pl.Float64)

    # 4. Run FDR correction
    reject, qvalues, _, _ = multipletests(
        pvals_to_correct,
        alpha=0.05,
        method=method
    )
    
    # 5. Add q-values back to the filtered DataFrame
    df_filtered = df_filtered.with_columns(
        pl.Series(name="q_value", values=qvalues)
    )
    
    # 6. Join back to the original DataFrame to restore shape and nulls
    df_final = df.join(
        df_filtered.select(["idx", "q_value"]),
        on="idx",
        how="left"
    )
    
    return df_final["q_value"]

def cliffs_delta(
    group1: np.ndarray,
    group2: np.ndarray
) -> float:
    """
    Calculates Cliff's Delta (d) effect size.

    A 1-liner implementation.
    - d = 0: No difference
    - d = 1: Group 1 is 100% larger than Group 2
    - d = -1: Group 2 is 100% larger than Group 1

    Args:
        group1: Numpy array for group 1.
        group2: Numpy array for group 2.

    Returns:
        The Cliff's Delta effect size (float).
    """
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
        
    # 1-liner calculation
    return float (np.mean(np.sign(np.subtract.outer(group1, group2))))