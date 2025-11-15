import pytest
import polars as pl
import numpy as np
import skbio
from typing import cast, List
from pgptracker.stage2_analysis.diversity import (
    calculate_alpha_diversity,
    calculate_beta_diversity,
    permanova_test,
    AlphaMetric)

# run with: pytest tests/unit/test_diversity.py -v

# --- Test Data (NEW: N×D Format) ---

@pytest.fixture
def df_raw_wide_N_D():
    """
    N×D (samples x features) table of raw counts.
    S1: 2 features, S2: 2 features, S3: 3 features
    S1 and S2 share no features
    """
    return pl.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'Feat_A': [10, 0, 1],
        'Feat_B': [10, 5, 1],
        'Feat_C': [0, 5, 1],
    })

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
def metadata():
    return pl.DataFrame({
        'SampleID': ['S1', 'S2', 'S3'],
        'Group': ['A', 'A', 'B']})

@pytest.fixture
def dist_matrix(df_clr_wide_N_D):
    # Pass N×D table
    dm = calculate_beta_diversity(df_clr_wide_N_D, 'Sample', 'aitchison')
    return dm

# --- Tests ---

class TestDiversity:

    def test_calculate_alpha_diversity_numeric(self, df_raw_wide_N_D):
        """Validates the numerical values of alpha diversity."""
        metrics_str = ['observed_features', 'shannon']
        metrics = cast(List[AlphaMetric], metrics_str)
        
        # Pass N×D table
        results = calculate_alpha_diversity(df_raw_wide_N_D, 'Sample', metrics)
        
        # Validate Observed Features
        obs_feat = results.filter(pl.col('Metric') == 'observed_features')
        assert obs_feat.filter(pl.col('Sample') == 'S1')['Value'][0] == 2.0
        assert obs_feat.filter(pl.col('Sample') == 'S2')['Value'][0] == 2.0
        assert obs_feat.filter(pl.col('Sample') == 'S3')['Value'][0] == 3.0
        
        # **BUG FIX**: Test now passes because script uses log2
        shannon = results.filter(pl.col('Metric') == 'shannon')
        # [10, 10, 0] -> 1.0 (log2)
        assert shannon.filter(pl.col('Sample') == 'S1')['Value'][0] == pytest.approx(1.0)
        # [1, 1, 1] -> 1.5849... (log2)
        assert shannon.filter(pl.col('Sample') == 'S3')['Value'][0] == pytest.approx(1.5849625)

    def test_calculate_beta_braycurtis_numeric(self, df_raw_wide_N_D):
        """Validates the Bray-Curtis distance calculation."""
        # Pass N×D table
        dm = calculate_beta_diversity(df_raw_wide_N_D, 'Sample', 'braycurtis')
        
        # S1=[10, 10, 0] vs S2=[0, 5, 5]
        # Dist = 0.666...
        assert dm['S1', 'S2'] == pytest.approx(0.666666, abs=1e-5)

    def test_calculate_beta_aitchison_numeric(self, df_clr_wide_N_D):
        """Validates that Aitchison is Euclidean distance on CLR data."""
        # Pass N×D table
        dm = calculate_beta_diversity(df_clr_wide_N_D, 'Sample', 'aitchison')

        # S1_clr = [0.5, -0.1, -0.4]
        # S2_clr = [-0.2, 0.3, -0.1]
        # Dist = sqrt(0.74)
        assert dm['S1', 'S2'] == pytest.approx(np.sqrt(0.74))

    def test_permanova(self, df_raw_wide_N_D, metadata):
        """Tests that PERMANOVA runs and returns the correct key."""
        dm = calculate_beta_diversity(df_raw_wide_N_D, 'Sample', 'braycurtis')
        
        results = permanova_test(
            distance_matrix=dm,
            metadata=metadata,
            sample_id_col='SampleID',
            formula="~Group"
        )
        
        # **BUG FIX**: Test now passes because script uses 'p_value'
        assert 'p_value' in results
        assert 'test_statistic' in results
        assert results['p_value'] > 0.3 # p-value for 3 samples

    def test_calculate_alpha_diversity_metrics_coverage(self):
            """
            Validates all alpha diversity metrics with simple, known data.
            """
            df_cov = pl.DataFrame({
                'Sample': ['S1', 'S2'],
                'Feat_A': [10, 100],
                'Feat_B': [10, 0],
            })
            
            metrics_str = ['shannon', 'simpson', 'pielou_e', 'observed_features']
            metrics = cast(List[AlphaMetric], metrics_str)
            
            results = calculate_alpha_diversity(df_cov, 'Sample', metrics)

            # S1 Results (Perfect Evenness)
            s1_res = results.filter(pl.col('Sample') == 'S1').pivot(
                values="Value", index="Sample", on="Metric"
            )
            assert s1_res['observed_features'][0] == 2.0
            # **BUG FIX**: Test now passes because script uses log2
            assert s1_res['shannon'][0] == pytest.approx(1.0)      # H = log2(2)
            assert s1_res['simpson'][0] == pytest.approx(0.5)      # 1 - (0.5^2 + 0.5^2)
            assert s1_res['pielou_e'][0] == pytest.approx(1.0)      # J = H / log2(S)

            # S2 Results (No Evenness)
            s2_res = results.filter(pl.col('Sample') == 'S2').pivot(
                values="Value", index="Sample", on="Metric"
            )
            assert s2_res['observed_features'][0] == 1.0 # Only 1 feature observed
            assert s2_res['shannon'][0] == pytest.approx(0.0)
            assert s2_res['simpson'][0] == pytest.approx(0.0)      # 1 - (1^2 + 0^2)
            assert s2_res['pielou_e'][0] is None                   # Pielou is undefined when S < 2

    def test_calculate_beta_jaccard_numeric(self):
        """Validates the Jaccard distance calculation."""
        df_jac = pl.DataFrame({
            'Sample': ['S1', 'S2'],
            'Feat_A': [10, 10],
            'Feat_B': [10, 0],
            'Feat_C': [0, 10],
        })
        dm = calculate_beta_diversity(df_jac, 'Sample', 'jaccard')
        
        # S1=[10, 10, 0] vs S2=[10, 0, 10]
        # Jaccard Distance = 0.666...
        assert dm['S1', 'S2'] == pytest.approx(0.666666, abs=1e-5)

    def test_permanova_mismatched_metadata(self, dist_matrix, metadata):
        """
        Tests that PERMANOVA raises ValueError if metadata IDs
        do not match the distance matrix IDs.
        """
        bad_metadata = pl.DataFrame({
            'SampleID': ['S1', 'S2', 'S99'], # S99 does not match S3
            'Group': ['A', 'A', 'B']})
        
        with pytest.raises(ValueError, match="Metadata is missing SampleIDs"):
            permanova_test(
                dist_matrix,
                bad_metadata,
                sample_id_col='SampleID',
                formula="~Group")

    def test_permanova_bad_formula_column(self, dist_matrix, metadata):
        """
        Tests that PERMANOVA raises ValueError if the formula
        contains a column name not in the metadata.
        """
        # **BUG FIX**: Updated match string to the actual error
        with pytest.raises(ValueError, match="Column 'NonExistentColumn' not in DataFrame"):
            permanova_test(
                dist_matrix,
                metadata,
                sample_id_col='SampleID',
                formula="~NonExistentColumn")