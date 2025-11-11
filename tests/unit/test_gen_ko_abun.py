"""
Unit tests for generate KO abundances (gen_ko_abun) module

Run with: pytest tests/unit/test_gen_ko_abun.py -v
"""

import pytest
import polars as pl
from polars.testing import assert_frame_equal
from pathlib import Path
import gzip
import io

# Import the functions to be tested
# Adjust the import path based on your project structure
from pgptracker.stage1_processing.gen_ko_abun import (
    _filter_by_nsti_polars,
    _normalize_by_marker_polars,
    _unstrat_funcs_only_by_samples_polars,
    run_metagenome_pipeline 
)

# --- Fixtures for Mock Data ---

@pytest.fixture
def mock_lf_table():
    """Mock feature table (ASV x Sample)"""
    return pl.LazyFrame({
        "ASV_ID": ["ASV1", "ASV2", "ASV3", "ASV4_no_marker"],
        "SampleA": [100, 50, 0, 100],
        "SampleB": [20, 0, 30, 10],
    })

@pytest.fixture
def mock_lf_marker():
    """Mock marker predictions. Includes NSTI, zero-copy, and non-overlapping ASV."""
    return pl.LazyFrame({
        "sequence": ["ASV1", "ASV2", "ASV3", "ASV5_nsti_high", "ASV6_no_table"],
        "marker_copies": [2.0, 1.0, 0.0, 1.0, 1.0], # ASV3 has 0 copies
        "metadata_NSTI": [0.5, 0.8, 1.0, 2.5, 0.5], # ASV5 will be filtered
        "closest_reference_genome": ["g1", "g2", "g3", "g5", "g6"]
    })

@pytest.fixture
def mock_lf_func():
    """Mock KO predictions. Includes NSTI and non-overlapping ASV."""
    return pl.LazyFrame({
        "sequence": ["ASV1", "ASV2", "ASV3", "ASV5_nsti_high", "ASV7_no_table"],
        "ko:K00001": [1, 1, 1, 1, 1],
        "ko:K00002": [0, 2, 0, 1, 1],
        "metadata_NSTI": [0.5, 0.8, 1.0, 2.5, 0.5], # ASV5 will be filtered
        "closest_reference_genome": ["g1", "g2", "g3", "g5", "g7"]
    })

@pytest.fixture
def sample_cols():
    return ["SampleA", "SampleB"]

@pytest.fixture
def ko_cols():
    return ["ko:K00001", "ko:K00002"]

@pytest.fixture
def id_col():
    return "ASV_ID" # Standardized name

# --- Unit Tests for Helper Functions ---

def test_filter_by_nsti_polars(mock_lf_marker, id_col):
    """
    Tests that NSTI filtering and metadata column dropping works.
    """
    # Test filtering
    lf_filtered = _filter_by_nsti_polars(
        mock_lf_marker.rename({"sequence": id_col}), 
        max_nsti=2.0, 
        id_col=id_col
    )
    df_filtered = lf_filtered.collect()
    
    # ASV5_nsti_high (2.5) should be removed
    assert df_filtered.shape[0] == 4
    assert "ASV5_nsti_high" not in df_filtered[id_col]
    
    # Metadata columns should be dropped
    assert "metadata_NSTI" not in df_filtered.columns
    assert "closest_reference_genome" not in df_filtered.columns
    assert "marker_copies" in df_filtered.columns # Data col remains


def test_normalize_by_marker_polars_division_by_zero(mock_lf_table, mock_lf_marker, sample_cols, id_col):
    """
    Tests the normalization logic, specifically for division by zero.
    This validates the change requested by the user.
    """
    # Standardize ID col names for test
    lf_table = mock_lf_table
    lf_marker = mock_lf_marker.rename({"sequence": id_col})
    
    # Manually filter marker for this test
    lf_marker_filt = lf_marker.filter(pl.col(id_col).is_in(["ASV1", "ASV2", "ASV3"]))

    lf_norm = _normalize_by_marker_polars(
        lf_table, lf_marker_filt, sample_cols, id_col
    )
    df_norm = lf_norm.collect().sort(id_col)

    # Expected results
    # ASV1: [100/2, 20/2] = [50.0, 10.0]
    # ASV2: [50/1, 0/1]   = [50.0, 0.0]
    # ASV3: [0/0, 30/0]   = [NaN, inf] -> Polars div(0,0)=NaN, div(>0,0)=inf
    
    expected_df = pl.DataFrame({
        id_col: ["ASV1", "ASV2", "ASV3"],
        "SampleA": [50.00, 50.00, float('nan')], # 0 / 0 = NaN
        "SampleB": [10.00, 0.00, float('inf')], # 30 / 0 = inf
    })

    # Handle NaN comparison: fill NaN with None (null) on both frames
    df_norm = df_norm.fill_nan(None)
    expected_df = expected_df.fill_nan(None)

    # Use Polars test function, checking NaNs are equal
    assert_frame_equal(df_norm, expected_df, check_dtypes=False)

def test_unstrat_funcs_only_by_samples_polars(id_col, sample_cols, ko_cols):
    """
    Tests the unstratified calculation logic with known inputs/outputs.
    """
    # Mock normalized abundance data (post-normalization)
    lf_norm = pl.LazyFrame({
        id_col: ["ASV1", "ASV2"],
        "SampleA": [50.0, 50.0],
        "SampleB": [10.0, 0.0],
    })
    
    # Mock function data (post-filtering)
    lf_func = pl.LazyFrame({
        id_col: ["ASV1", "ASV2"],
        "ko:K00001": [1, 1],
        "ko:K00002": [0, 2],
    })
    
    lf_unstrat = _unstrat_funcs_only_by_samples_polars(
        lf_func, lf_norm, ko_cols, sample_cols, id_col
    )
    df_unstrat = lf_unstrat.collect().sort("function")
    
    # --- Manual Calculation ---
    # KO1, SampleA = (ASV1_Abun * ASV1_KO1) + (ASV2_Abun * ASV2_KO1)
    #               = (50.0 * 1) + (50.0 * 1) = 100.0
    # KO1, SampleB = (10.0 * 1) + (0.0 * 1) = 10.0
    # KO2, SampleA = (50.0 * 0) + (50.0 * 2) = 100.0
    # KO2, SampleB = (10.0 * 0) + (0.0 * 2) = 0.0
    # ---
    
    expected_df = pl.DataFrame({
        "function": ["ko:K00001", "ko:K00002"],
        "SampleA": [100.0, 100.0],
        "SampleB": [10.0, 0.0],
    }).fill_null(0.0) # Ensure no nulls
    
    # Reorder columns to match pivot output
    expected_df = expected_df.select(df_unstrat.columns)

    assert_frame_equal(df_unstrat, expected_df, check_dtypes=False)

# --- Integration Test ---

def test_run_metagenome_pipeline_integration(
    tmp_path, 
    mock_lf_table, 
    mock_lf_marker, 
    mock_lf_func,
    mocker # Use pytest-mock to stub external calls
):
    """
    Tests the full run_metagenome_pipeline function.
    This validates the "collect-once" logic and file I/O.
    """
    # 1. Setup mock environment
    output_dir = tmp_path / "pgptracker_output"
    
    # Define file paths
    table_path = tmp_path / "table.tsv"
    marker_path = tmp_path / "marker_nsti_predicted.tsv.gz"
    ko_path = tmp_path / "KO_predicted.tsv.gz"
    
    # Write mock data to temp files
    mock_lf_table.collect().write_csv(table_path, separator='\t')
    
    # Write gzipped files
    with gzip.open(marker_path, 'wb') as f:
        mock_lf_marker.collect().write_csv(f, separator='\t')
    with gzip.open(ko_path, 'wb') as f:
        mock_lf_func.collect().write_csv(f, separator='\t')
        
    # Mock the external command (biom convert)
    # This path MUST match where it is imported and USED
    mocker.patch("pgptracker.stage1_processing.gen_ko_abun.run_command", return_value=None)
    
    # 2. Run the function
    result_paths = run_metagenome_pipeline(
        table_path=table_path,
        marker_path=marker_path,
        ko_predicted_path=ko_path,
        output_dir=output_dir,
        max_nsti=2.0 # Filter ASV5
    )
    
    # 3. Validate outputs
    norm_path = result_paths['seqtab_norm']
    unstrat_path = result_paths['pred_metagenome_unstrat']
    
    # Check that files were created
    assert norm_path.exists()
    assert unstrat_path.exists()
    assert norm_path.name == "seqtab_norm.tsv.gz"
    assert unstrat_path.name == "pred_metagenome_unstrat.tsv.gz"

    # 4. Read results back and validate content
    
    # --- Validate Normalized Table ---
    # Only ASV1 and ASV2 should overlap and pass NSTI
    # ASV3 has marker=0, but is in all files
    # ASV4_no_marker is not in marker/func
    
    df_norm = pl.read_csv(norm_path, separator='\t').sort("normalized")
    
    # Expected results (ASV1, ASV2, ASV3)
    # ASV1: [100/2, 20/2] = [50.0, 10.0]
    # ASV2: [50/1, 0/1]   = [50.0, 0.0]
    # ASV3: [0/0, 30/0]   = [NaN, inf]
    expected_norm_df = pl.DataFrame({
        "normalized": ["ASV1", "ASV2", "ASV3"],
        "SampleA": [50.00, 50.00, float('nan')],
        "SampleB": [10.00, 0.00, float('inf')],
    })

    # Handle NaN comparison
    assert_frame_equal(
        df_norm.fill_nan(None), 
        expected_norm_df.fill_nan(None), 
        check_dtypes=False)
    
    # --- Validate Unstratified Table ---
    df_unstrat = pl.read_csv(unstrat_path, separator='\t').sort("function")

    # Manual Calculation (based on ASV1, ASV2):
    # lf_norm (input to unstrat):
    #   ASV_ID  SampleA  SampleB
    #   ASV1    50.0     10.0
    #   ASV2    50.0     0.0
    #   ASV3    NaN      inf
    # lf_func (input to unstrat):
    #   ASV_ID  K00001  K00002
    #   ASV1    1       0
    #   ASV2    1       2
    #   ASV3    1       0
    
    # KO1, SampleA = (50*1) + (50*1) + (NaN*1) = 100.0 + NaN = NaN
    # KO1, SampleB = (10*1) + (0*1)  + (inf*1) = 10.0 + inf  = inf
    # KO2, SampleA = (50*0) + (50*2) + (NaN*0) = 100.0 + 0   = 100.0
    # KO2, SampleB = (10*0) + (0*2)  + (inf*0) = 0 + 0     = 0.0
    
    expected_unstrat_df = pl.DataFrame({
        "function": ["ko:K00001", "ko:K00002"],
        "SampleA": [float('nan'), 100.0],
        "SampleB": [float('inf'), 0.0],
    }).fill_null(0.0)
    
    # Reorder columns to match pivot output
    expected_unstrat_df = expected_unstrat_df.select(df_unstrat.columns)

    # Handle NaN comparison
    assert_frame_equal(df_unstrat.fill_nan(None), 
                       expected_unstrat_df.fill_nan(None), 
                       check_dtypes=False)