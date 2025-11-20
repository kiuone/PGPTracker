# src/pgptracker/stage2_analysis/ordination.py
import polars as pl
import numpy as np
import skbio
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import OrdinationResults, pcoa
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage
from typing import Tuple, List

def _prepare_skbio_matrix(
    df_wide_N_D: pl.DataFrame, 
    sample_id_col: str
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Validates N×D DataFrame and extracts matrix, sample IDs, and feature IDs.
    
    Args:
        df_wide_N_D: N×D (samples × features) Polars DataFrame.
        sample_id_col: Name of the sample ID column (e.g., "Sample").

    Returns:
        Tuple: (matrix, sample_ids, feature_ids)
    """
    if sample_id_col not in df_wide_N_D.columns:
        raise KeyError(f"Sample ID column '{sample_id_col}' not in DataFrame.")
        
    sample_ids = df_wide_N_D.get_column(sample_id_col).to_list()
    feature_ids = [c for c in df_wide_N_D.columns if c != sample_id_col]
    
    if not feature_ids:
        raise ValueError("No feature columns found in DataFrame.")
        
    matrix = df_wide_N_D.select(feature_ids).to_numpy()
    
    return matrix, sample_ids, feature_ids

def run_pca(
    df_wide_N_D_clr: pl.DataFrame, 
    sample_id_col: str, 
    n_components: int = 2
) -> Tuple[pl.DataFrame, pl.DataFrame, SklearnPCA]:
    """
    Performs PCA on an N×D CLR-transformed DataFrame.
    
    Args:
        df_wide_N_D_clr: N×D (samples × features) CLR-transformed data.
        sample_id_col: Name of the sample ID column (e.g., "Sample").
        n_components: Number of principal components.

    Returns:
        Tuple: (Polars DataFrame of PC scores, fitted SklearnPCA model)
    """
    matrix, sample_ids, feature_ids = _prepare_skbio_matrix(df_wide_N_D_clr, sample_id_col)
    pca = SklearnPCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(matrix)
    
    col_names = [f"PC{i+1}" for i in range(n_components)]
    
    df_scores = pl.DataFrame(
        scores,
        schema=col_names
    ).insert_column(0, pl.Series("Sample", sample_ids))
    
    # Create Loadings DataFrame (Vectors)
    # Components shape is (n_components, n_features), so we transpose
    loadings = pca.components_.T 
    df_loadings = pl.DataFrame(loadings, schema=col_names).insert_column(0, pl.Series("Feature", feature_ids))
    
    return df_scores, df_loadings, pca

def run_pcoa(distance_matrix: DistanceMatrix) -> OrdinationResults:
    """
    Performs PCoA on a pre-computed distance matrix.
    """
    return pcoa(distance_matrix)

def run_tsne(
    matrix_N_D: np.ndarray, 
    sample_ids: List[str],
    metric: str = 'euclidean',
    perplexity: float = 2.0,
    random_state: int = 42
) -> pl.DataFrame:
    """
    Runs t-SNE on an N×D matrix.
    
    Args:
        matrix_N_D: N×D (samples × features) numpy array.
        sample_ids: List of sample IDs.
        metric: Distance metric.
        perplexity: t-SNE perplexity. Should be < n_samples.
        random_state: Seed for reproducibility.

    Returns:
        Polars DataFrame of t-SNE scores.
    """
    n_samples = matrix_N_D.shape[0]
    if perplexity >= n_samples:
        perplexity = max(1.0, n_samples - 1.0) # Adjust perplexity
        
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    
    scores = tsne.fit_transform(matrix_N_D)
    
    df_scores = pl.DataFrame(
        scores,
        schema=["tSNE1", "tSNE2"]
    ).insert_column(0, pl.Series("Sample", sample_ids))
    
    return df_scores

def run_hierarchical_clustering(
    distance_matrix: DistanceMatrix, 
    method: str = 'ward'
) -> Tuple[np.ndarray, List[str]]:
    """
    Performs hierarchical clustering on a distance matrix.
    
    Args:
        distance_matrix: An skbio DistanceMatrix.
        method: Clustering method (e.g., 'ward', 'single', 'complete').

    Returns:
        Tuple: (linkage_matrix, sample_ids)
    """
    sample_ids = distance_matrix.ids
    # Convert skbio DistanceMatrix to condensed distance vector
    condensed_dm = distance_matrix.condensed_form()
    
    linkage_matrix = linkage(condensed_dm, method=method)
    
    return linkage_matrix, list(sample_ids)