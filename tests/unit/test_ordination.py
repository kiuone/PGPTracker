import pytest
import polars as pl
import numpy as np
import skbio
from skbio.stats.ordination import OrdinationResults
from sklearn.decomposition import PCA as SklearnPCA
from pgptracker.stage2_analysis.ordination import (
    run_pca,
    run_pcoa,
    run_tsne,
    run_hierarchical_clustering,
    _prepare_skbio_matrix 
)
from pgptracker.stage2_analysis.diversity import calculate_beta_diversity

# --- Test Data (NEW: N×D Format) ---
# run with: pytest tests/unit/test_ordination.py -v

@pytest.fixture
def df_clr_wide_N_D():
    """
    N×D (samples x features) CLR table.
    """
    return pl.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'Feat_A': [0.5, -0.2, 0.1],
        'Feat_B': [-0.1, 0.3, -0.3],
        'Feat_C': [-0.4, -0.1, 0.2],
    })

@pytest.fixture
def dist_matrix(df_clr_wide_N_D):
    # This fixture now reads the N×D table
    dm = calculate_beta_diversity(df_clr_wide_N_D, 'Sample', 'aitchison')
    return dm

# --- Tests ---

class TestOrdination:

    def test_prepare_skbio_matrix(self, df_clr_wide_N_D):
        """Validates the N×D matrix preparation."""
        matrix, s_ids, f_ids = _prepare_skbio_matrix(df_clr_wide_N_D, 'Sample')
        
        assert matrix.shape == (3, 3)     # (samples, feats)
        assert s_ids == ['S1', 'S2', 'S3']
        assert f_ids == ['Feat_A', 'Feat_B', 'Feat_C']
        assert matrix[0, 0] == 0.5 # S1, Feat_A

    def test_run_pca_numeric(self, df_clr_wide_N_D):
        """Validates the numerical values of PCA from N×D input."""
        pc_scores, pca_model = run_pca(df_clr_wide_N_D, 'Sample', n_components=2)
        
        assert isinstance(pca_model, SklearnPCA)
        assert pc_scores.shape == (3, 3) # S1,S2,S3 and Sample, PC1, PC2
        
        # Ground truth calculated with random_state=42
        # X = np.array([[ 0.5, -0.1, -0.4], [-0.2,  0.3, -0.1], [ 0.1, -0.3,  0.2]])
        # S1_PC1_truth = 0.441112...
        
        s1_pc1_test = pc_scores.filter(pl.col('Sample') == 'S1')['PC1'][0]
        assert s1_pc1_test == pytest.approx(0.441112)

    def test_run_pcoa(self, dist_matrix):
        """Tests that PCoA runs and returns the correct object."""
        pcoa_results = run_pcoa(dist_matrix)
        
        assert isinstance(pcoa_results, OrdinationResults)
        assert pcoa_results.samples.shape == (3, 3)
        assert 'S1' in pcoa_results.samples.index

    def test_run_hierarchical_clustering(self, dist_matrix):
        """Tests hierarchical clustering."""
        linkage_matrix, ids = run_hierarchical_clustering(
            dist_matrix, method='ward'
        )
        
        assert ids == ['S1', 'S2', 'S3'] 
        assert linkage_matrix.shape == (2, 4)

    def test_run_tsne_deterministic(self, df_clr_wide_N_D):
        """Tests that t-SNE is deterministic with random_state=42."""
        matrix, s_ids, _ = _prepare_skbio_matrix(df_clr_wide_N_D, 'Sample')
        
        tsne_results = run_tsne(
            matrix, s_ids, 
            metric='euclidean', 
            random_state=42
        )
        
        assert tsne_results.shape == (3, 3) # Sample, tSNE1, tSNE2
        
        # Ground truth calculated with perplexity=2, random_state=42
        # S1_tsne1_truth = -74.2462...
        s1_tsne1_test = tsne_results.filter(pl.col('Sample') == 'S1')['tSNE1'][0]
        assert s1_tsne1_test == pytest.approx(87.00577, abs=1e-3)