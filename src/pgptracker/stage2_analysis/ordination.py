# -*- coding: utf-8 -*-
"""
Calculates ordination methods (PCA, PCoA, t-SNE).

Assumes input DataFrames for PCA/t-SNE are in 'wide' format
(features x samples) or are pre-computed matrices.

Functions in this module perform calculations only and return data.
Visualization is handled by 'exports.visualizations'.
"""

import polars as pl
import skbio
import numpy as np
import skbio.stats.ordination
import skbio.stats.distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skbio.stats.ordination import OrdinationResults
from sklearn.decomposition import PCA as SklearnPCA
from typing import Literal, Optional, Tuple, List, Any

def _prepare_skbio_matrix(
    df_wide: pl.DataFrame,
    feature_col: str
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Internal helper: Converts wide (features x samples) df to sklearn-ready format.

    Args:
        df_wide: A wide DataFrame (features are rows, samples are columns).
        feature_col: The name of the column containing feature IDs.

    Returns:
        A tuple of:
        - (np.ndarray): The (samples x features) matrix.
        - (List[str]): The list of sample IDs, matching the matrix rows.
        - (List[str]): The list of feature IDs, matching the matrix columns.
    """
    matrix_pd = df_wide.to_pandas().set_index(feature_col).T
    sample_ids = matrix_pd.index.to_list()
    feature_ids = matrix_pd.columns.to_list()
    matrix = matrix_pd.to_numpy()
    
    return matrix, sample_ids, feature_ids

def run_pcoa(
    distance_matrix: skbio.stats.distance.DistanceMatrix
) -> skbio.stats.ordination.OrdinationResults:
    """
    Performs Principal Coordinate Analysis (PCoA).

    This is a 1-line wrapper for the scikit-bio implementation.

    Args:
        distance_matrix: A (samples x samples) distance matrix,
                         e.g., from calculate_beta_diversity().

    Returns:
        An skbio OrdinationResults object.
    """
    # 1-liner calculation
    return skbio.stats.ordination.pcoa(distance_matrix)

def calculate_variance_explained(
    ordination_results: (OrdinationResults | SklearnPCA), # skbio.OrdinationResults or sklearn.PCA
    n_components: int = 3
) -> pl.DataFrame:
    """
    Extracts the variance explained from PCoA or PCA results.

    Args:
        ordination_results: The output object from run_pcoa() or run_pca().
        n_components: How many components to report.

    Returns:
        A DataFrame with [Component, Proportion_Explained, Cumulative_Proportion].
    """
    if isinstance(ordination_results, OrdinationResults):
        # scikit-bio PCoA results
        variance_explained = ordination_results.proportion_explained
    elif isinstance(ordination_results, SklearnPCA):
        # scikit-learn PCA results
        variance_explained = ordination_results.explained_variance_ratio_
    else:
        raise TypeError(
            "Input must be skbio.stats.ordination.OrdinationResults or "
            "sklearn.decomposition.PCA"
        )

    components = [f"PC{i+1}" for i in range(len(variance_explained))] # type: ignore[arg-type]
    
    df = pl.DataFrame({
        "Component": components,
        "Proportion_Explained": variance_explained
    }).with_columns(
        pl.col("Proportion_Explained").cum_sum().alias("Cumulative_Proportion")
    )
    
    return df.head(n_components)

def run_pca(
    df_clr_wide: pl.DataFrame,
    feature_col: str,
    n_components: int = 3
) -> Tuple[pl.DataFrame, PCA]:
    """
    Performs Principal Component Analysis (PCA).

    CRITICAL: This function *must* be run on a CLR-transformed
    wide (features x samples) table.

    Args:
        df_clr_wide: A wide, CLR-transformed DataFrame (features x samples).
        feature_col: The name of the feature ID column.
        n_components: Number of components to calculate.

    Returns:
        A tuple containing:
        - pc_scores (pl.DataFrame): The PC scores [Sample, PC1, PC2, ...].
        - pca_model (PCA): The fitted scikit-learn model object.
    """
    # 1. Convert (features x samples) df to (samples x features) matrix
    matrix, sample_ids, _ = _prepare_skbio_matrix(df_clr_wide, feature_col)
    
    # 2. 2-liner calculation
    pca_model = PCA(n_components=n_components)
    pc_scores_np = pca_model.fit_transform(matrix)
    
    # 3. Format as DataFrame
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    pc_scores = pl.DataFrame(pc_scores_np, schema=pc_cols).with_columns(
        pl.Series("Sample", sample_ids)
    ).select("Sample", *pc_cols) # Put Sample column first
    
    return pc_scores, pca_model

def run_tsne(
    data_or_matrix: np.ndarray,
    sample_ids: List[str],
    metric: Literal['euclidean', 'precomputed'] = 'euclidean',
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: Optional[int] = 42
) -> pl.DataFrame:
    """
    Performs t-distributed Stochastic Neighbor Embedding (t-SNE).

    CRITICAL:
    - If metric='euclidean', input *must* be (samples x features) CLR data.
    - If metric='precomputed', input *must* be a (samples x samples) distance matrix.

    Args:
        data_or_matrix: The (samples x ...) numpy array.
        sample_ids: List of sample IDs matching the matrix rows.
        metric: 'euclidean' (for CLR data) or 'precomputed' (for distance matrix).
        n_components: Number of dimensions (usually 2).
        perplexity: t-SNE perplexity parameter.
        random_state: Seed for reproducibility.

    Returns:
        A DataFrame with t-SNE embeddings [Sample, tSNE1, tSNE2].
    """
    # 1. 2-liner calculation
    tsne_model = TSNE(
        n_components=n_components,
        metric=metric,
        perplexity=min(perplexity, len(sample_ids) - 1), # Perplexity must be < n_samples
        random_state=random_state,
        init='random'
    )
    tsne_embedding = tsne_model.fit_transform(data_or_matrix)
    
    # 2. Format as DataFrame
    tsne_cols = [f"tSNE{i+1}" for i in range(n_components)]
    return pl.DataFrame(tsne_embedding, schema=tsne_cols).with_columns(
        pl.Series("Sample", sample_ids)
    ).select("Sample", *tsne_cols)