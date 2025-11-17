import pytest
import polars as pl
import numpy as np
from pgptracker.stage2_analysis.clustering_ML import (
    run_lasso_cv,
    run_random_forest,
    run_boruta,
    _prepare_ml_data)

# Run with: pytest tests/unit/test_machine_leaning.py -v

# --- Test Data (NEW: N×D Format) ---

@pytest.fixture
def df_ml_N_D_clr():
    """
    CLR table N×D (samples x features) with clear signal.
    """
    return pl.DataFrame({
        'Sample': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
        # Feat_A (Signal 1): High in A, Low in B
        'Feat_A': [5.0, 5.1, 5.0, -5.0, -5.1, -5.0],
        # Feat_B (Noise): Stays near 0
        'Feat_B': [0.01, -0.02, 0.0, -0.01, 0.02, 0.01],
        # Feat_C (Signal 2): High in A, Low in B
        'Feat_C': [4.9, 5.0, 5.1, -4.9, -5.0, -5.1],
    })

@pytest.fixture
def metadata_class():
    # Metadata for Classification
    return pl.DataFrame({
        'Sample': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'], 
        'Group': ['A', 'A', 'A', 'B', 'B', 'B']})

@pytest.fixture
def metadata_reg():
    # Metadata for Regression (correlated with Feat_A/C)
    return pl.DataFrame({
        'Sample': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'], 
        'Value': [20.1, 20.3, 19.9, -19.8, -20.2, -20.0]})

# --- Tests ---

class TestMachineLearning:

    def test_prepare_ml_data(self, df_ml_N_D_clr, metadata_class, metadata_reg):
        """Validates alignment and 'y' encoding from N×D input."""
        
        # 1. Test Regression
        X, y, feats, tmap = _prepare_ml_data(
            df_ml_N_D_clr, metadata_reg, 'Sample', 'Value', 'regression'
        )
        assert X.shape == (6, 3) # (6 samples, 3 features)
        assert y.shape == (6,)
        assert feats == ['Feat_A', 'Feat_B', 'Feat_C']
        assert np.array_equal(y, np.array([20.1, 20.3, 19.9, -19.8, -20.2, -20.0]))

        # 2. Test Classification
        X_c, y_c, _, tmap_c = _prepare_ml_data(
            df_ml_N_D_clr, metadata_class, 'Sample', 'Group', 'classification'
        )
        assert X_c.shape == (6, 3)
        assert np.array_equal(y_c, np.array([0, 0, 0, 1, 1, 1]))
        assert tmap_c == {0: 'A', 1: 'B'}

    def test_run_lasso_cv(self, df_ml_N_D_clr, metadata_reg):
        """Validates that LASSO zeros out the noise feature (Feat_B)."""
        results = run_lasso_cv(
            df_ml_N_D_clr, metadata_reg, 'Sample', 'Value', 
            n_cv=3, random_state=42
        )
        
        # Signal features A should have high coefficients
        assert results.filter(pl.col('Feature') == 'Feat_A')['Coefficient'][0] > 1.0
        
        # Noise feature B and C should be zeroed out
        assert results.filter(pl.col('Feature') == 'Feat_B').is_empty()

        coef_c = results.filter(pl.col('Feature') == 'Feat_C')['Coefficient']
        if not coef_c.is_empty():
            assert abs(coef_c[0]) < 0.01
        
    def test_run_random_forest(self, df_ml_N_D_clr, metadata_class):
        """Validates that RF assigns low importance to the noise feature."""
        results = run_random_forest(
            df_ml_N_D_clr, metadata_class, 'Sample', 'Group',
            'classification', random_state=42
        )
        
        res_a = results.filter(pl.col('Feature') == 'Feat_A')['Importance'][0]
        res_b = results.filter(pl.col('Feature') == 'Feat_B')['Importance'][0]
        res_c = results.filter(pl.col('Feature') == 'Feat_C')['Importance'][0]

        # Signal (A, C) > Noise (B)
        assert res_a > res_b
        assert res_c > res_b
        assert (res_a + res_c) > 0.75 # A and C should explain almost everything
        
        # **BUG FIX**: Use pytest.approx for float comparison
        assert res_b < 0.25

    def test_run_boruta(self, df_ml_N_D_clr, metadata_class):
        """Validates that Boruta Confirms the signal and Rejects the noise."""
        results = run_boruta(
            df_ml_N_D_clr, metadata_class, 'Sample', 'Group',
            random_state=42
        )
        
        # Signal (A, C) should be 'Confirmed'
        assert results.filter(pl.col('Feature') == 'Feat_A')['Decision'][0] in ['Confirmed', 'Tentative']
        assert results.filter(pl.col('Feature') == 'Feat_C')['Decision'][0] in ['Confirmed', 'Tentative']
        
        # Noise (B) should be 'Rejected'
        assert results.filter(pl.col('Feature') == 'Feat_B')['Decision'][0] == 'Rejected'

    def test_prepare_ml_data_no_sample_overlap(self, df_ml_N_D_clr):
        """
        Tests that _prepare_ml_data raises ValueError if no samples
        overlap between the CLR table and metadata.
        """
        mismatched_metadata = pl.DataFrame({
        'Sample': ['S10', 'S11', 'S12', 'S13'], 
        'Group': ['A', 'A', 'B', 'B']
    })
        
        with pytest.raises(ValueError, match="No matching samples found"):
            _prepare_ml_data(
                df_ml_N_D_clr, mismatched_metadata, 'Sample', 
                'Group', 'classification'
            )

    def test_prepare_ml_data_missing_columns(self, df_ml_N_D_clr, metadata_class):
        """
        Tests that _prepare_ml_data raises KeyError if critical
        columns are missing from the inputs.
        """
        # 1. 'sample_id_col' faltando no DataFrame de dados
        with pytest.raises(KeyError, match="Sample ID column 'BadSampleCol' not in wide data"):
            _prepare_ml_data(
                df_ml_N_D_clr, metadata_class, 'BadSampleCol', 
                'Group', 'classification'
            )
            
        # 2. 'sample_id_col' faltando nos Metadados
        with pytest.raises(KeyError, match="Sample ID column 'Sample' not in metadata"):
            _prepare_ml_data(
                df_ml_N_D_clr, 
                metadata_class.rename({"Sample": "WrongName"}), # Renomeia col
                'Sample', 'Group', 'classification'
            )
            
        # 3. 'target_col' faltando nos Metadados
        with pytest.raises(KeyError, match="Target column 'BadGroupCol' not in metadata"):
            _prepare_ml_data(
                df_ml_N_D_clr, metadata_class, 'Sample', 
                'BadGroupCol', 'classification'
            )

    def test_prepare_ml_data_non_numeric_regression(self, df_ml_N_D_clr, metadata_class):
        """
        Tests that _prepare_ml_data raises ValueError if 'regression'
        is requested but the target column is non-numeric (e.g., strings).
        """
        with pytest.raises(ValueError, match="target 'Group' must be numeric"):
            _prepare_ml_data(
                df_ml_N_D_clr, metadata_class, 'Sample', 
                'Group', 'regression'
            )