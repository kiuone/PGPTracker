# -*- coding: utf-8 -*-
"""
Calculates alpha and beta diversity metrics and performs PERMANOVA tests.

Assumes input DataFrames for alpha/beta calculations are in 'wide' format
(features x samples), as the 'long' to 'wide' pivot is now handled
by the upstream 'clr_normalize.py' script.

Functions in this module perform calculations only and return data.
Visualization is handled by 'exports.visualizations'.
"""

import polars as pl
import numpy as np
import skbio.stats.distance
import skbio.diversity
import patsy 
from patsy import dmatrix #type: ignore[attr-defined]
from typing import List, Literal, Optional, Dict, Any, Tuple
from pgptracker.stage2_analysis.clr_normalize import apply_clr

# Define common metric types
AlphaMetric = Literal['shannon', 'simpson', 'pielou_e', 'observed_features']
BetaMetric = Literal['braycurtis', 'jaccard', 'aitchison', 'euclidean']

def _prepare_skbio_matrix(
    df_wide: pl.DataFrame,
    feature_col: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Internal helper: Converts wide (features x samples) df to skbio-ready format.

    Args:
        df_wide: A wide DataFrame (features are rows, samples are columns).
        feature_col: The name of the column containing feature IDs.

    Returns:
        A tuple of:
        - (np.ndarray): The (samples x features) matrix.
        - (List[str]): The list of sample IDs, matching the matrix rows.
    """
    # 1. Transpose to (samples x features) using pandas
    matrix_pd = df_wide.to_pandas().set_index(feature_col).T
    
    # 2. Get sample IDs from the new index
    sample_ids = matrix_pd.index.to_list()
    
    # 3. Get the numpy matrix
    counts_matrix = matrix_pd.to_numpy()
    
    return counts_matrix, sample_ids

def calculate_alpha_diversity(
    df_wide: pl.DataFrame,
    feature_col: str,
    metrics: List[AlphaMetric]
) -> pl.DataFrame:
    """
    Calculate one or more alpha diversity metrics.

    Args:
        df_wide: A wide format abundance DataFrame (features x samples).
                 This table should contain RAW ABUNDANCES, not CLR.
        feature_col: The column name representing features (e.g., 'PGPT_ID').
        metrics: List of scikit-bio metrics (e.g., ['shannon', 'simpson']).

    Returns:
        A Polars DataFrame in long format: [Sample, Metric, Value]
    """
    # 1. Convert (features x samples) df to (samples x features) matrix
    counts_matrix, sample_ids = _prepare_skbio_matrix(df_wide, feature_col)
    
    results = []
    # 2. Calculate each metric (1-liner)
    for metric in metrics:
        alpha_div = skbio.diversity.alpha_diversity(
            metric, counts_matrix, ids=sample_ids)
        
        # 3. Format output
        metric_df = pl.from_pandas(alpha_div.to_frame(name='Value')).with_columns(
            pl.lit(metric).alias('Metric')
        ).select(
            pl.col('index').alias('Sample'),
            pl.col('Metric'),
            pl.col('Value'))
        results.append(metric_df)

    return pl.concat(results)

def calculate_beta_diversity(
    df_wide: pl.DataFrame,
    feature_col: str,
    metric: BetaMetric
) -> skbio.stats.distance.DistanceMatrix:
    """
    Calculate a beta diversity distance matrix.

    CRITICAL:
    - For 'braycurtis', 'jaccard', etc., pass the RAW ABUNDANCE wide table.
    - For 'aitchison', pass the CLR-TRANSFORMED wide table
      (e.g., 'unstratified_clr' or 'stratified_wide_clr').
    
    Args:
        df_wide: A wide format abundance DataFrame (features x samples).
        feature_col: The column name representing features.
        metric: The scikit-bio metric. If 'aitchison', this function
                will run 'euclidean' (as Aitchison = Euclidean on CLR).

    Returns:
        A scikit-bio DistanceMatrix object (samples x samples).
    """
    
    # 1. Convert (features x samples) df to (samples x features) matrix
    data_matrix, sample_ids = _prepare_skbio_matrix(df_wide, feature_col)

    effective_metric = metric
    if metric == 'aitchison':
        # Aitchison distance IS Euclidean distance on CLR-transformed data.
        # We assume the user passed the CLR-transformed table.
        effective_metric = 'euclidean'

    # 2. Calculate beta diversity (1-liner)
    dist_matrix = skbio.diversity.beta_diversity(
        effective_metric, data_matrix, ids=sample_ids)
    
    # Aitchison distance IS Euclidean distance on CLR-transformed data.
    # The 'metric' attribute on the result object is not writable.

    return dist_matrix

def permanova_test(
    distance_matrix: skbio.stats.distance.DistanceMatrix,
    metadata: pl.DataFrame,
    sample_id_col: str,
    formula: str,
    permutations: int = 999
) -> Dict[str, Any]:
    """
    Performs a PERMANOVA test using an R-style formula.

    Args:
        distance_matrix: A (samples x samples) distance matrix.
        metadata: A DataFrame containing sample metadata.
        sample_id_col: The name of the column in metadata matching matrix IDs.
        formula: An R-style formula (e.g., "~Treatment + pH").
        permutations: Number of permutations to run.

    Returns:
        A dictionary of results from scikit-bio (e.g., 'p_value', 'test_statistic').
    
    Raises:
        ValueError: If metadata is missing samples or columns.
    """
    if sample_id_col not in metadata.columns:
        raise ValueError(
            f"Sample ID column '{sample_id_col}' not found in metadata. "
            f"Available columns: {metadata.columns}")
        
    metadata_pd = metadata.to_pandas().set_index(sample_id_col)
    
    # Align metadata to the distance matrix order
    try:
        metadata_aligned = metadata_pd.loc[distance_matrix.ids]
    except KeyError:
        raise ValueError(
            "Metadata is missing SampleIDs present in the distance matrix. "
            "Please check for mismatches."
        )

    # Use patsy to create the design matrix from the formula
    try:
        grouping_df = dmatrix(
            formula, data=metadata_aligned, return_type='dataframe')
    except Exception as e:
        raise ValueError(
            f"Failed to parse formula '{formula}'. "
            f"Check column names. Error: {e}")

    # 1-liner calculation
    results = skbio.stats.distance.permanova(
        distance_matrix, grouping=grouping_df, permutations=permutations)
    
    # Return as a standard dict
    return dict(results)