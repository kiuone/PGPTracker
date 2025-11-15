# run with: 
# pytest tests/unit/test_clr_normalize_v2.py -v
import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np
from pathlib import Path
import shutil

# Import functions to be tested
from pgptracker.stage2_analysis.clr_normalize import (
    apply_clr,
    _combine_feature_columns,
    _pivot_long_to_wide_N_D,
    _transpose_D_N_to_N_D,
    _transpose_N_D_to_D_N,
    _clr_wide_N_D,
    _create_D_N_split_table
)

# Tolerance for floating point comparisons
TOLERANCE = 1e-5

# --- Helper Function for creating test files ---
def _create_test_file(tmp_path: Path, df: pl.DataFrame, name: str) -> Path:
    """Writes a DataFrame to a temporary TSV file."""
    p = tmp_path / name
    df.write_csv(p, separator="\t")
    return p

# === Unit Tests for Helper Functions ===

def test_combine_feature_columns_multiple(df_long_stratified):
    """ Tests combining 'Family' and 'Lv3' into 'FeatureID' """
    result = _combine_feature_columns(
        df_long_stratified,
        sample_col="Sample",
        value_col="Total_PGPT_Abundance"
    )
    
    expected_data = {
        "FeatureID": [
            "Fam1|NITROGEN", 
            "Fam1|NITROGEN", 
            "Fam2|PHOSPHATE", 
            "Fam2|PHOSPHATE", 
            "Fam3|SIDEROPHORE"
        ],
        "Sample": ["S1", "S2", "S1", "S3", "S1"],
        "Total_PGPT_Abundance": [10, 5, 15, 20, 100],
    }
    expected = pl.DataFrame(expected_data).select(
        "FeatureID", "Sample", "Total_PGPT_Abundance"
    )
    
    assert_frame_equal(result, expected)

def test_combine_feature_columns_single(df_long_unstratified):
    """ Tests when only one feature column exists (just renames) """
    result = _combine_feature_columns(
        df_long_unstratified,
        sample_col="Sample",
        value_col="Abundance"
    )
    # Expect it to just rename 'FeatureID'
    assert "FeatureID" in result.columns
    assert "Abundance" in result.columns
    assert "Sample" in result.columns

def test_combine_feature_columns_no_features_error():
    """ Tests error when no feature columns are found """
    df = pl.DataFrame({"Sample": ["S1"], "Value": [10]})
    with pytest.raises(ValueError, match="No feature columns found"):
        _combine_feature_columns(df, sample_col="Sample", value_col="Value")

def test_pivot_long_to_wide_N_D(df_long_stratified):
    """ Tests pivoting from long (combined) to wide N_D """
    df_combined = _combine_feature_columns(
        df_long_stratified, "Sample", "Total_PGPT_Abundance"
    )
    
    result = _pivot_long_to_wide_N_D(
        df_combined,
        sample_col="Sample",
        value_col="Total_PGPT_Abundance"
    )
    
    expected_data = {
        "Sample": ["S1", "S2", "S3"],
        "Fam1|NITROGEN": [10.0, 5.0, 0.0],
        "Fam2|PHOSPHATE": [15.0, 0.0, 20.0],
        "Fam3|SIDEROPHORE": [100.0, 0.0, 0.0],
    }
    expected = pl.DataFrame(expected_data)
    
    # Sort columns for comparison stability
    result = result.select(sorted(result.columns))
    expected = expected.select(sorted(expected.columns))
    
    assert_frame_equal(result, expected)

def test_transpose_D_N_to_N_D(df_wide_D_N):
    """ Tests transposing D_N (Feat x Sample) to N_D (Sample x Feat) """
    result = _transpose_D_N_to_N_D(df_wide_D_N, id_col_name="Lv3")
    
    expected_data = {
        "Sample": ["S1", "S2"],
        "Feat1": [10.0, 5.0],
        "Feat2": [15.0, 20.0],
    }
    expected = pl.DataFrame(expected_data)
    
    assert_frame_equal(result, expected)

def test_transpose_D_N_to_N_D_non_numeric_error(df_wide_D_N_with_metadata):
    """ Tests that D_N transpose fails if metadata columns are present """
    with pytest.raises(ValueError, match="Non-numeric metadata columns found"):
        _transpose_D_N_to_N_D(df_wide_D_N_with_metadata, id_col_name="Lv3")

def test_transpose_N_D_to_D_N(df_wide_N_D):
    """ Tests transposing N_D (Sample x Feat) to D_N (Feat x Sample) """
    result = _transpose_N_D_to_D_N(df_wide_N_D)
    
    expected_data = {
        "FeatureID": ["Feat1", "Feat2"],
        "S1": [10.0, 15.0],
        "S2": [5.0, 20.0],
    }
    expected = pl.DataFrame(expected_data)
    
    assert_frame_equal(result, expected)

def test_clr_wide_N_D_normal_case(df_wide_N_D):
    """ Tests CLR calculation on a standard N_D table """
    result = _clr_wide_N_D(df_wide_N_D)
    
    # S1: [10, 15] -> gmean = (10*15)**0.5 = 12.247
    # clr(10) = ln(10) - ln(gmean) = -0.20273
    # clr(15) = ln(15) - ln(gmean) =  0.20273
    # S2: [5, 20] -> gmean = (5*20)**0.5 = 10
    # clr(5) = ln(5) - ln(10) = -0.69315
    # clr(20) = ln(20) - ln(10) = 0.69315
    
    expected_data = {
        "Sample": ["S1", "S2"],
        "Feat1": [-0.20273, -0.69315],
        "Feat2": [0.20273, 0.69315],
    }
    expected = pl.DataFrame(expected_data)
    
    assert_frame_equal(result, expected, abs_tol=TOLERANCE)

def test_clr_wide_N_D_all_zeros_row(df_wide_N_D_all_zeros):
    """ Tests CLR guard rail for rows that are all zeros """
    result = _clr_wide_N_D(df_wide_N_D_all_zeros)
    
    # S2_zeros row should remain [0.0, 0.0]
    expected_data = {
        "Sample": ["S1", "S2_zeros"],
        "Feat1": [-0.20273, 0.0],
        "Feat2": [0.20273, 0.0],
    }
    expected = pl.DataFrame(expected_data)
    
    assert_frame_equal(result, expected, abs_tol=TOLERANCE)

def test_clr_wide_N_D_single_feature(df_wide_N_D_single_feature):
    """ Tests CLR guard rail for D=1 (single feature) """
    result = _clr_wide_N_D(df_wide_N_D_single_feature)
    
    # CLR is not defined for D=1, should return zeros
    expected_data = {
        "Sample": ["S1", "S2"],
        "Feat1": [0.0, 0.0],
    }
    expected = pl.DataFrame(expected_data)
    
    assert_frame_equal(result, expected, abs_tol=TOLERANCE)

def test_clr_wide_N_D_no_id_col_error():
    """ Tests CLR error if no non-numeric ID column is found """
    df = pl.DataFrame({"Feat1": [10], "Feat2": [15]})
    with pytest.raises(ValueError, match="No ID column"):
        _clr_wide_N_D(df)

def test_create_D_N_split_table_normal(df_wide_N_D):
    """ Tests splitting FeatureID (no split) """
    # This N_D table has features "Feat1", "Feat2" (no '|')
    # It should just transpose to D_N
    result = _create_D_N_split_table(df_wide_N_D)
    
    expected_data = {
        "FeatureID": ["Feat1", "Feat2"],
        "S1": [10.0, 15.0],
        "S2": [5.0, 20.0],
    }
    expected = pl.DataFrame(expected_data)
    assert_frame_equal(result, expected)

def test_create_D_N_split_table_inconsistent_error(df_long_inconsistent_features):
    """ Tests error on inconsistent feature splits (e.g., 'A|B' and 'A|B|C') """
    df_combined = _combine_feature_columns(
        df_long_inconsistent_features, "Sample", "Value"
    )
    df_N_D = _pivot_long_to_wide_N_D(
        df_combined, "Sample", "Value"
    )
    
    # This should fail validation inside _create_D_N_split_table
    with pytest.raises(ValueError, match="Inconsistent FeatureID levels detected"):
        _create_D_N_split_table(df_N_D)

# === Integration Tests for apply_clr (Main Function) ===

def test_integration_apply_clr_long_stratified(tmp_path, df_long_stratified):
    """ Full pipeline test for 'long'/'stratified' input """
    input_file = _create_test_file(
        tmp_path, df_long_stratified, "long_stratified.tsv"
    )
    output_dir = tmp_path / "output_long"
    base_name = "test.tsv"
    
    paths = apply_clr(
        input_path=input_file,
        input_format='long',
        output_dir=output_dir,
        base_name=base_name,
        export_sparcc_format=True,
        keep_feature_cols_separate=True,
        long_sample_col="Sample",
        long_value_col="Total_PGPT_Abundance"
    )
    
    # 1. Check all files were created
    assert "raw_long" in paths
    assert "raw_wide_N_D" in paths
    assert "clr_wide_N_D" in paths
    assert "raw_wide_D_N" in paths
    assert "clr_wide_D_N_split" in paths
    
    assert paths["raw_long"].exists()
    assert paths["clr_wide_D_N_split"].exists()
    
    # 2. Check the most complex output: clr_wide_D_N_split
    df_split = pl.read_csv(paths["clr_wide_D_N_split"], separator="\t")
    
    assert "Feature_Level_1" in df_split.columns
    assert "Feature_Level_2" in df_split.columns
    assert "S1" in df_split.columns
    assert "S2" in df_split.columns
    assert "S3" in df_split.columns
    
    # Check values (manually calculated CLR for S1: [10, 15, 100])
    # gmean = (10*15*100)**(1/3) = 24.66
    # clr(10) = ln(10) - ln(24.66) = -0.902
    # clr(15) = ln(15) - ln(24.66) = -0.499
    # clr(100) = ln(100) - ln(24.66) = 1.401
    
    s1_values = df_split.select("S1").to_series().to_list()
    assert np.isclose(float(s1_values[0]), -0.902683, atol=TOLERANCE) # Fam1|NITROGEN
    assert np.isclose(float(s1_values[1]), -0.497218, atol=TOLERANCE) # Fam2|PHOSPHATE
    assert np.isclose(float(s1_values[2]), 1.399901, atol=TOLERANCE) # Fam3|SIDEROPHORE

def test_integration_apply_clr_wide_D_N(tmp_path, df_wide_D_N):
    """ Full pipeline test for 'wide' 'D_N' input """
    input_file = _create_test_file(tmp_path, df_wide_D_N, "wide_D_N.tsv")
    output_dir = tmp_path / "output_wide_DN"
    base_name = "test_DN.tsv"

    paths = apply_clr(
        input_path=input_file,
        input_format='wide',
        output_dir=output_dir,
        base_name=base_name,
        export_sparcc_format=False,
        keep_feature_cols_separate=False,
        wide_orientation="D_N",
        wide_id_col="Lv3"
    )
    
    # 1. Check standard files were created
    assert "raw_wide_N_D" in paths
    assert "clr_wide_N_D" in paths
    assert "raw_long" not in paths
    assert "clr_wide_D_N_split" not in paths
    
    # 2. Check CLR values
    df_clr = pl.read_csv(paths["clr_wide_N_D"], separator="\t")
    
    # Values from test_clr_wide_N_D_normal_case
    expected_data = {
        "Sample": ["S1", "S2"],
        "Feat1": [-0.20273, -0.69315],
        "Feat2": [0.20273, 0.69315],
    }
    expected = pl.DataFrame(expected_data)
    assert_frame_equal(df_clr, expected, abs_tol=TOLERANCE)

def test_integration_apply_clr_wide_N_D(tmp_path, df_wide_N_D):
    """ Full pipeline test for 'wide' 'N_D' input """
    input_file = _create_test_file(tmp_path, df_wide_N_D, "wide_N_D.tsv")
    output_dir = tmp_path / "output_wide_ND"
    base_name = "test_ND.tsv"

    paths = apply_clr(
        input_path=input_file,
        input_format='wide',
        output_dir=output_dir,
        base_name=base_name,
        wide_orientation="N_D",
        wide_id_col="Sample"
    )
    
    # Check CLR values (should be identical to previous test)
    df_clr = pl.read_csv(paths["clr_wide_N_D"], separator="\t")
    expected_data = {
        "Sample": ["S1", "S2"],
        "Feat1": [-0.20273, -0.69315],
        "Feat2": [0.20273, 0.69315],
    }
    expected = pl.DataFrame(expected_data)
    assert_frame_equal(df_clr, expected, abs_tol=TOLERANCE)

def test_apply_clr_invalid_format_error(tmp_path):
    """ Tests error on invalid 'input_format' """
    with pytest.raises(ValueError, match="Invalid format: 'invalid_format'"):
        apply_clr(
            input_path=Path("fake.tsv"),
            input_format='invalid_format',
            output_dir=tmp_path,
            base_name="test.tsv"
        )

def test_apply_clr_invalid_wide_orientation_error(tmp_path, df_wide_D_N):
    """ Tests error on invalid 'wide_orientation' """
    input_file = _create_test_file(tmp_path, df_wide_D_N, "wide_D_N.tsv")
    
    with pytest.raises(ValueError, match="Invalid 'wide_orientation'"):
        apply_clr(
            input_path=input_file,
            input_format='wide',
            output_dir=tmp_path,
            base_name="test.tsv",
            wide_orientation="INVALID_ORIENTATION",
            wide_id_col="Lv3"
        )