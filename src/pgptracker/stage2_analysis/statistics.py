# src/pgptracker/stage2_analysis/statistics.py
import polars as pl
import numpy as np
import scipy.stats as ss
from typing import List, Optional, Literal
from statsmodels.stats.multitest import multipletests

def _prepare_stats_long_df(
    df_wide_N_D: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_id_col: str,
    feature_col: str,
    group_col: str,
    value_col: str
) -> pl.DataFrame:
    """
    Prepares the long-format DataFrame needed for statistical tests.
    
    1. Reads N×D wide table (e.g., CLR-transformed).
    2. Reads metadata table.
    3. Joins them on Sample ID.
    4. Unpivots (melts) the N×D table to long format.
    
    Returns:
        A long DataFrame: [sample_id_col, group_col, feature_col, value_col]
    """
    if sample_id_col not in df_wide_N_D.columns:
        raise KeyError(f"Sample ID column '{sample_id_col}' not in wide data.")
    if sample_id_col not in metadata.columns:
        raise KeyError(f"Sample ID column '{sample_id_col}' not in metadata.")
    if group_col not in metadata.columns:
        raise KeyError(f"Group column '{group_col}' not in metadata.")

    # Select only the necessary columns from metadata
    md_subset = metadata.select([sample_id_col, group_col])
    
    # Join wide data with metadata
    df_joined = df_wide_N_D.join(md_subset, on=sample_id_col, how="inner")
    
    if df_joined.height == 0:
        raise ValueError(
            f"No samples matched between data and metadata on column '{sample_id_col}'.")

    # Unpivot (melt) to long format
    feature_cols = [c for c in df_wide_N_D.columns if c != sample_id_col]
    
    df_long = df_joined.unpivot(
        index=[sample_id_col, group_col],
        on=feature_cols,
        variable_name=feature_col,
        value_name=value_col)
    
    return df_long

def kruskal_wallis_test(
    df_wide_N_D: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_id_col: str,
    feature_col: str,
    group_col: str,
    value_col: str
) -> pl.DataFrame:
    """
    Performs Kruskal-Wallis H-test on each feature, grouped by the group_col.
    
    Args:
        df_wide_N_D: The N×D (samples x features) CLR-transformed data.
        metadata: The metadata table.
        sample_id_col: Name of the sample ID column (e.g., "Sample").
        feature_col: Name to give the 'feature' column in the output (e.g., "Feature").
        group_col: Name of the metadata grouping column (e.g., "Group").
        value_col: Name to give the 'value' column in the output (e.g., "Value").

    Returns:
        A Polars DataFrame: [Feature, test_statistic, p_value]
    """
    print(f"Running Kruskal-Wallis on '{feature_col}' grouped by '{group_col}'...")
    
    try:
        df_long = _prepare_stats_long_df(
            df_wide_N_D, metadata, sample_id_col, 
            feature_col, group_col, value_col)
    except (KeyError, ValueError) as e:
        print(f"  -> Error preparing data: {e}")
        return pl.DataFrame(schema={"Feature": pl.String, "test_statistic": pl.Float64, "p_value": pl.Float64})

    results = []
    for feature_tuple, data in df_long.group_by(feature_col):
        feature = feature_tuple[0]

        groups = data.get_column(group_col).unique().to_list()

        if len(groups) < 2:
            results.append((feature, np.nan, np.nan))
            continue
            
        group_arrays = [
            data.filter(pl.col(group_col) == g).get_column(value_col).to_numpy()
            for g in groups]
        
        # Remove empty arrays just in case
        group_arrays = [arr for arr in group_arrays if len(arr) > 0]
        
        if len(group_arrays) < 2:
            results.append((feature, np.nan, np.nan))
            continue
            
        try:
            stat, pval = ss.kruskal(*group_arrays)
            results.append((feature, stat, pval))
        except ValueError:
            results.append((feature, np.nan, np.nan))
            
    return pl.DataFrame(
        results,
        schema={"Feature": pl.String, "test_statistic": pl.Float64, "p_value": pl.Float64},
        orient="row"
    ).sort("Feature")


def mann_whitney_u_test(
    df_wide_N_D: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_id_col: str,
    feature_col: str,
    group_col: str,
    value_col: str,
    group_1: str,
    group_2: str
) -> pl.DataFrame:
    """
    Performs pairwise Mann-Whitney U-test on each feature between two groups.
    
    (See kruskal_wallis_test for arg descriptions)
    """
    print(f"Running Mann-Whitney U between '{group_1}' and '{group_2}'...")
    
    try:
        df_long = _prepare_stats_long_df(
            df_wide_N_D, metadata, sample_id_col, 
            feature_col, group_col, value_col)
    except (KeyError, ValueError) as e:
        print(f"  -> Error preparing data: {e}")
        return pl.DataFrame(schema={"Feature": pl.String, "test_statistic": pl.Float64, "p_value": pl.Float64})
    
    results = []
    for feature_tuple, data in df_long.group_by(feature_col):
        feature = feature_tuple[0]
        g1_data = data.filter(pl.col(group_col) == group_1).get_column(value_col).to_numpy()
        g2_data = data.filter(pl.col(group_col) == group_2).get_column(value_col).to_numpy()
        
        if len(g1_data) == 0 or len(g2_data) == 0:
            results.append((feature, np.nan, np.nan))
            continue
            
        try:
            stat, pval = ss.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
            results.append((feature, stat, pval))
        except ValueError:
            results.append((feature, np.nan, np.nan))

    return pl.DataFrame(results, schema={"Feature": pl.String, 
                         "test_statistic": pl.Float64, "p_value": pl.Float64},
        orient="row"
    ).sort("Feature")


def cliffs_delta(g1: np.ndarray, g2: np.ndarray) -> float:
    """Calculates Cliff's Delta effect size."""
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
        
    n_gt = 0
    n_lt = 0
    for x in g1:
        for y in g2:
            if x > y:
                n_gt += 1
            elif x < y:
                n_lt += 1
                
    return (n_gt - n_lt) / (len(g1) * len(g2))

def fdr_correction(
    p_values: pl.Series, 
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> pl.Series:
    """Applies FDR correction to a Polars Series of p-values."""
    if p_values.is_empty():
        return pl.Series("q_value", [], dtype=pl.Float64)
        
    mask_not_null = p_values.is_not_null()
    
    # Get non-null p-values for correction
    pvals_to_correct = p_values.filter(mask_not_null).to_numpy()
    
    if len(pvals_to_correct) == 0:
        return p_values.alias("q_value")

    reject, qvals, _, _ = multipletests(
        pvals_to_correct, alpha=alpha, method=method
    )
    
    # Create a new series with nulls in the right places
    qvals_full = pl.Series(
        "q_value", 
        np.full(len(p_values), np.nan), 
        dtype=pl.Float64
    )
    
    qvals_full = qvals_full.scatter(
        mask_not_null.arg_true(), 
        qvals
    )
    
    return qvals_full