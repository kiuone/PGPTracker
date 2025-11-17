import pytest
import polars as pl
import numpy as np
from pgptracker.stage2_analysis.statistics import (
    kruskal_wallis_test, mann_whitney_u_test,
    cliffs_delta, fdr_correction)

# run with: pytest tests/unit/test_statistics.py -v
# --- Test Data ---

@pytest.fixture
def df_wide_N_D_stats():
    """
    N×D (samples x features) table.
    PGPT_1: Groups clearly separated
    PGPT_2: Groups mixed
    """
    return pl.DataFrame({
        "Sample": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "PGPT_1": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0],
        "PGPT_2": [1.0, 10.0, 2.0, 9.0, 3.0, 11.0],
    })

@pytest.fixture
def metadata_stats():
    return pl.DataFrame({
        "Sample": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "Group": ['A', 'A', 'A', 'B', 'B', 'B']
    })

# --- Tests ---

class TestStatistics:

    def test_kruskal_wallis(self, df_wide_N_D_stats, metadata_stats):
        """Validates the numerical output of Kruskal-Wallis."""
        results = kruskal_wallis_test(
            df_wide_N_D_stats,
            metadata_stats,
            sample_id_col='Sample',
            feature_col='Feature',
            group_col='Group',
            value_col='Value'
        )
        
        assert results.shape == (2, 3)
        
        res_p1 = results.filter(pl.col('Feature') == 'PGPT_1')
        res_p2 = results.filter(pl.col('Feature') == 'PGPT_2')

        # PGPT_1: Groups are perfectly separated -> low p-value
        # **BUG FIX**: Updated p-value to match scipy's output
        assert res_p1['p_value'][0] == pytest.approx(0.0495346)
        
        # PGPT_2: Groups are mixed -> high p-value
        assert res_p2['p_value'][0] == pytest.approx(0.2752335)

    def test_mann_whitney_u(self, df_wide_N_D_stats, metadata_stats):
        """Validates the numerical output of Mann-Whitney U."""
        results = mann_whitney_u_test(
            df_wide_N_D_stats,
            metadata_stats,
            sample_id_col='Sample',
            feature_col='Feature',
            group_col='Group',
            value_col='Value',
            group_1='A',
            group_2='B'
        )
        
        assert results.shape == (2, 3)
        
        res_p1 = results.filter(pl.col('Feature') == 'PGPT_1')
        res_p2 = results.filter(pl.col('Feature') == 'PGPT_2')

        assert res_p1['p_value'][0] == pytest.approx(0.1)
        assert res_p2['p_value'][0] == pytest.approx(0.4)

    @pytest.mark.parametrize("g1, g2, expected_delta", [
        # Case 1: Perfect separation (G2 > G1)
        (np.array([1, 2, 3]), np.array([4, 5, 6]), -1.0),
        # Case 2: Perfect separation (G1 > G2)
        (np.array([10, 20]), np.array([1, 2]), 1.0),
        # Case 3: No dominance (perfect tie)
        (np.array([1, 5]), np.array([2, 4]), 0.0),
        # Case 4: Partial dominance (Corrected)
        # (-1*4 + 1*2) / 6 = -2 / 6 = -0.333...
        (np.array([1, 2, 5]), np.array([3, 4]), -0.333333333),])
    
    def test_cliffs_delta_numeric(self, g1, g2, expected_delta):
        """Validates the numerical calculation of Cliff's Delta."""
        delta = cliffs_delta(g1, g2)
        assert delta == pytest.approx(expected_delta)

    def test_fdr_correction(self):
        """Validates the FDR (Benjamini-Hochberg) calculation."""
        pvals = pl.Series("p_value", [0.01, 0.03, 0.5, 0.02, None])
        
        # Expected: [0.04, 0.04, 0.5, 0.04, None]
        expected_list = [0.04, 0.04, 0.5, 0.04, None]
        
        qvals = fdr_correction(pvals, method='fdr_bh')
        qvals_list = qvals.to_list()
        
        assert qvals_list[0] == pytest.approx(expected_list[0])
        assert qvals_list[1] == pytest.approx(expected_list[1])
        assert qvals_list[2] == pytest.approx(expected_list[2])
        assert qvals_list[3] == pytest.approx(expected_list[3])
        assert np.isnan(qvals_list[4])

    def test_mann_whitney_u_single_group(self, df_wide_N_D_stats, metadata_stats):
        """
        Tests that MWU returns NaN when only one group is present.
        """
        df_one_group = df_wide_N_D_stats.filter(pl.col('Sample').is_in(['S1', 'S2', 'S3']))
        
        results = mann_whitney_u_test(
            df_one_group,
            metadata_stats, # Pass full metadata, filter happens inside
            sample_id_col='Sample',
            feature_col='Feature',
            group_col='Group',
            value_col='Value',
            group_1='A',
            group_2='B') # Group B does not exist in the filtered input
        
        # **BUG FIX**: Use np.isnan() to check for nan, not 'is None'
        assert np.isnan(results.filter(pl.col('Feature') == 'PGPT_1')['p_value'][0])
        assert np.isnan(results.filter(pl.col('Feature') == 'PGPT_2')['p_value'][0])

    def test_fdr_correction_empty_input(self):
        """
        Tests that fdr_correction returns an empty Series
        when given an empty Series.
        """
        pvals = pl.Series("p_value", [], dtype=pl.Float64)
        expected = pl.Series("q_value", [], dtype=pl.Float64)
        
        qvals = fdr_correction(pvals)
        
        assert qvals.equals(expected)

    def test_cliffs_delta_empty_input(self):
        """Tests that Cliff's Delta returns NaN if one group is empty."""
        g1 = np.array([1, 2, 3])
        g2_empty = np.array([])
        
        delta = cliffs_delta(g1, g2_empty)
        assert np.isnan(delta)