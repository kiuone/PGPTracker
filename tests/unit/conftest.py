# tests/unit/conftest.py
import pytest
import polars as pl
from pathlib import Path

# --- Fixtures for LONG format ---

@pytest.fixture
def df_long_stratified() -> pl.DataFrame:
    """ Long format, multiple feature columns (stratified) """
    data = {
        "Family": ["Fam1", "Fam1", "Fam2", "Fam2", "Fam3"],
        "Lv3": ["NITROGEN", "NITROGEN", "PHOSPHATE", "PHOSPHATE", "SIDEROPHORE"],
        "Sample": ["S1", "S2", "S1", "S3", "S1"],
        "Total_PGPT_Abundance": [10, 5, 15, 20, 100],
    }
    return pl.DataFrame(data)

@pytest.fixture
def df_long_unstratified() -> pl.DataFrame:
    """ Long format, single feature column (unstratified) """
    data = {
        "FeatureID": ["Feat1", "Feat1", "Feat2", "Feat3"],
        "Sample": ["S1", "S2", "S1", "S2"],
        "Abundance": [10, 5, 15, 8],
    }
    return pl.DataFrame(data)

@pytest.fixture
def df_long_inconsistent_features() -> pl.DataFrame:
    """ Long format, inconsistent feature splits (for error testing) """
    data = {
        "FeatureID": ["A|B", "A|B|C"],
        "Sample": ["S1", "S1"],
        "Value": [10, 20],
    }
    return pl.DataFrame(data)

# --- Fixtures for WIDE format ---

@pytest.fixture
def df_wide_N_D() -> pl.DataFrame:
    """ Wide format, N_D (Samples x Features) """
    data = {
        "Sample": ["S1", "S2"],
        "Feat1": [10, 5],
        "Feat2": [15, 20],
    }
    return pl.DataFrame(data, schema={"Sample": pl.String, "Feat1": pl.Float64, "Feat2": pl.Float64})

@pytest.fixture
def df_wide_D_N() -> pl.DataFrame:
    """ Wide format, D_N (Features x Samples) """
    data = {
        "Lv3": ["Feat1", "Feat2"],
        "S1": [10, 15],
        "S2": [5, 20],
    }
    return pl.DataFrame(data, schema={"Lv3": pl.String, "S1": pl.Float64, "S2": pl.Float64})

@pytest.fixture
def df_wide_D_N_with_metadata() -> pl.DataFrame:
    """ Wide format, D_N, with non-numeric metadata cols (for error testing) """
    data = {
        "Lv3": ["Feat1", "Feat2"],
        "Description": ["Desc1", "Desc2"], # Non-numeric
        "S1": [10, 15],
        "S2": [5, 20],
    }
    return pl.DataFrame(data)

# --- Fixtures for CLR Edge Cases ---

@pytest.fixture
def df_wide_N_D_all_zeros() -> pl.DataFrame:
    """ N_D table where one row is all zeros """
    data = {
        "Sample": ["S1", "S2_zeros"],
        "Feat1": [10, 0],
        "Feat2": [15, 0],
    }
    return pl.DataFrame(data, schema={"Sample": pl.String, "Feat1": pl.Float64, "Feat2": pl.Float64})

@pytest.fixture
def df_wide_N_D_single_feature() -> pl.DataFrame:
    """ N_D table with only one feature (D=1) """
    data = {
        "Sample": ["S1", "S2"],
        "Feat1": [10, 5],
    }
    return pl.DataFrame(data, schema={"Sample": pl.String, "Feat1": pl.Float64})