# src/pgptracker/stage2_analysis/diversity.py
import polars as pl
import numpy as np
import pandas as pd
import skbio
from skbio.stats.distance import DistanceMatrix
import skbio.diversity
import skbio.stats.distance
from scipy.spatial.distance import euclidean
from typing import List, Literal, Dict, Any, Tuple

# Define the literal type for Pylance/MyPy
AlphaMetric = Literal['observed_features', 'shannon', 'simpson', 'pielou_e']

def _prepare_skbio_counts(
    df_wide_N_D: pl.DataFrame, 
    id_col: str
) -> tuple[np.ndarray, List[str]]:
    """
    Converts N×D Polars DataFrame to (N, D) numpy array and sample IDs
    for skbio alpha diversity.
    """
    sample_ids = df_wide_N_D.get_column(id_col).to_list()
    counts_matrix = df_wide_N_D.drop(id_col).to_numpy()
    return counts_matrix, sample_ids

def _prepare_skbio_beta(
    df_wide: pl.DataFrame, 
    id_col: str
) -> tuple[np.ndarray, List[str]]:
    """
    Converts N×D (samples x features) or D×N (features x samples) DataFrame
    to the (N, D) numpy array and sample IDs required by skbio.
    
    This function is now robust to either orientation, but N×D is preferred.
    """
    if id_col not in df_wide.columns:
        raise KeyError(f"ID column '{id_col}' not in DataFrame.")

    # Check if first non-ID column is numeric.
    first_data_col = next(c for c in df_wide.columns if c != id_col)
    
    if df_wide[first_data_col].dtype.is_numeric():
        # --- Input is N×D (samples × features) ---
        # e.g., 'Sample' | 'Feat_A' | 'Feat_B'
        sample_ids = df_wide.get_column(id_col).to_list()
        matrix = df_wide.drop(id_col).to_numpy()
        return matrix, sample_ids
    else:
        # --- Input is D×N (features × samples) ---
        # e.g., 'Feature' | 'S1' | 'S2'
        # Transpose to N×D for skbio
        sample_ids = df_wide.columns[1:]
        matrix = df_wide.drop(id_col).to_numpy().T # Transpose
        return matrix, list(sample_ids)

def calculate_alpha_diversity(
    df_wide_N_D_raw: pl.DataFrame,
    sample_id_col: str,
    metrics: List[AlphaMetric]
) -> pl.DataFrame:
    """
    Calculates alpha diversity metrics from a raw N×D (samples × features) table.

    Args:
        df_wide_N_D_raw: N×D Polars DataFrame (samples × features) of RAW COUNTS.
        sample_id_col: Name of the sample ID column (e.g., "Sample").
        metrics: List of metrics to calculate (e.g., ['shannon', 'observed_features']).

    Returns:
        A long-format Polars DataFrame with [Sample, Metric, Value].
    """
    counts_matrix, sample_ids = _prepare_skbio_counts(
        df_wide_N_D_raw, sample_id_col
    )
    
    results_list = []
    for metric in metrics:
        # **BUG FIX**: Use np.log2 for Shannon, not np.log (base e)
        if metric == 'shannon':
            # skbio 'shannon' metric uses base e. np.log2(S) is H_max.
            # We must calculate it manually for log base 2.
            def shannon_log2(counts):
                freqs = counts / np.sum(counts)
                non_zero_freqs = freqs[freqs > 0]
                return -np.sum(non_zero_freqs * np.log2(non_zero_freqs))
            
            values = [shannon_log2(row) if np.sum(row) > 0 else 0.0 for row in counts_matrix]
        
        elif metric == 'pielou_e':
            # Pielou's Evenness J = H' / H_max = H_shannon / log2(S)
            def pielou(counts):
                S = (counts > 0).sum()
                if S < 2:
                    return None # Evenness is undefined for 1 feature
                
                freqs = counts / np.sum(counts)
                non_zero_freqs = freqs[freqs > 0]
                H = -np.sum(non_zero_freqs * np.log2(non_zero_freqs))
                H_max = np.log2(S)
                return H / H_max
                
            values = [pielou(row) if np.sum(row) > 0 else 0.0 for row in counts_matrix]
        
        else:
            # Use skbio's built-in functions for 'observed_features' and 'simpson'
            # Note: skbio.shannon uses base e, so we implemented it manually.
            values = skbio.diversity.alpha_diversity(
                metric, counts_matrix, ids=sample_ids
            ).to_list()
        
        for sample, value in zip(sample_ids, values):
            results_list.append({
                "Sample": sample,
                "Metric": metric,
                "Value": value
            })
            
    return pl.DataFrame(results_list)

def calculate_beta_diversity(
    df_wide: pl.DataFrame,
    id_col: str,
    metric: Literal['braycurtis', 'jaccard', 'aitchison']
) -> DistanceMatrix:
    """
    Calculates beta diversity from a wide DataFrame.
    
    Handles N×D (preferred) or D×N inputs.
    - 'braycurtis', 'jaccard': Use N×D raw counts table.
    - 'aitchison': Use N×D CLR-transformed table.

    Args:
        df_wide: Polars DataFrame (N×D or D×N).
        id_col: Name of the ID column ('Sample' for N×D, 'Feature' for D×N).
        metric: Distance metric to use.

    Returns:
        A skbio DistanceMatrix object.
    """
    matrix, sample_ids = _prepare_skbio_beta(df_wide, id_col)
    
    # --- BUG FIX START ---
    # `skbio.diversity.beta_diversity` CANNOT handle negative values (CLR data)
    # It assumes "counts", even for euclidean.
    if metric == 'aitchison':
        # Aitchison = Euclidean distance on CLR data.
        # We must use `DistanceMatrix.from_iterable` for pairwise Euclidean
        # which correctly handles negative values.
        # --- BUG FIX START ---
        # Passar a *função* 'euclidean', não a string.
        dm = DistanceMatrix.from_iterable(
            matrix, metric=euclidean, keys=sample_ids)
    else:
        # Bray-Curtis and Jaccard work as they expect non-negative counts
        dm = skbio.diversity.beta_diversity(
            metric, matrix, ids=sample_ids
        )
    # --- BUG FIX END ---
    
    return dm

def permanova_test(
    distance_matrix: DistanceMatrix,
    metadata: pl.DataFrame,
    sample_id_col: str,
    formula: str
) -> Dict[str, Any]:
    """
    Performs a PERMANOVA test on a distance matrix and metadata.

    Args:
        distance_matrix: An skbio DistanceMatrix.
        metadata: A Polars DataFrame containing metadata.
        sample_id_col: The name of the sample ID column in the metadata.
        formula: The R-style formula (e.g., "~Group").

    Returns:
        A dictionary of PERMANOVA results.
    """
    # Ensure metadata IDs match the distance matrix IDs
    dm_ids = set(distance_matrix.ids)
    md_ids = set(metadata.get_column(sample_id_col).to_list())
    
    if not dm_ids.issubset(md_ids):
        missing_ids = dm_ids - md_ids
        raise ValueError(
            f"Metadata is missing SampleIDs that are in the "
            f"distance matrix: {missing_ids}"
        )
    
    # Filter and reorder metadata to match the distance matrix
    metadata_pd = (
        metadata
        .filter(pl.col(sample_id_col).is_in(dm_ids))
        .to_pandas()
        .set_index(sample_id_col)
        # --- BUG FIX START ---
        # Must be list(ids), not tuple(ids), to avoid Pandas IndexingError
        .loc[list(distance_matrix.ids)]
        # --- BUG FIX END ---
    )

    try:
        results = skbio.stats.distance.permanova(
            distance_matrix,
            metadata_pd,
            column=formula.replace("~", "").strip(),
            permutations=999
        )
    except Exception as e:
        # Catch errors from skbio (e.g., bad column name)
        raise ValueError(f"Failed to run PERMANOVA: {e}")

    # **BUG FIX**: skbio results key is 'p-value' (hyphen), not 'p_value'
    return {
        'method': 'PERMANOVA',
        'formula': formula,
        'p_value': results['p-value'],
        'test_statistic': results['test statistic'],
        'permutations': results['number of permutations']
    }