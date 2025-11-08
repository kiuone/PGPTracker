"""
Unit tests for unstratified PGPT analysis functions.

Tests cover:
- Loading pathways database with different PGPT levels
- Processing KO->PGPT mappings
- Generating unstratified abundance tables
- Error handling for missing files and invalid levels

Run with: pytest tests/unit/test_unstratified.py -v
"""
import pytest
import polars as pl
from pathlib import Path
import gzip
from unittest.mock import patch, MagicMock
import io

from pgptracker.analysis.unstratified import (
    load_pathways_db,
    generate_unstratified_pgpt
)

# --- Fixtures ---

@pytest.fixture
def mock_pathways_df():
    """Mock pathways database with KO mappings at multiple levels."""
    return pl.DataFrame({
        "PGPT_ID": [
            "PGPT_001-gene-K00001",
            "PGPT_002-gene-K00002",
            "PGPT_003-gene-K00003",
            "PGPT_004-invalid"  # Invalid KO pattern - will be filtered out
        ],
        "Lv1": ["BIO-FERTILIZATION", "BIO-FERTILIZATION", "BIOCONTROL", "STRESS"],
        "Lv2": ["NITROGEN", "PHOSPHORUS", "PATHOGEN_SUPPRESSION", "DROUGHT"],
        "Lv3": ["NITROGEN_FIXATION", "P_SOLUBILIZATION", "ANTIBIOTIC_PRODUCTION", "OSMOLYTE_SYNTHESIS"],
        "Lv4": ["N2_REDUCTION", "ORGANIC_P_HYDROLYSIS", "BETA_LACTAM_SYNTHESIS", "PROLINE_ACCUMULATION"],
    })

@pytest.fixture
def mock_pathways_file(tmp_path):
    """Create actual TSV file for integration tests."""
    p = tmp_path / "pathways_plabase.txt"
    content = (
        "PGPT_ID\tLv1\tLv2\tLv3\tLv4\n"
        "PGPT_001-gene-K00001\tBIO-FERTILIZATION\tNITROGEN\tNITROGEN_FIXATION\tN2_REDUCTION\n"
        "PGPT_002-gene-K00002\tBIO-FERTILIZATION\tPHOSPHORUS\tP_SOLUBILIZATION\tORGANIC_P_HYDROLYSIS\n"
        "PGPT_003-gene-K00003\tBIOCONTROL\tPATHOGEN_SUPPRESSION\tANTIBIOTIC_PRODUCTION\tBETA_LACTAM_SYNTHESIS\n"
        "PGPT_004-invalid\tSTRESS\tDROUGHT\tOSMOLYTE_SYNTHESIS\tPROLINE_ACCUMULATION\n"
    )
    p.write_text(content)
    return p

@pytest.fixture
def mock_ko_unstrat_file(tmp_path):
    """Mock unstratified KO abundance file (pred_metagenome_unstrat.tsv.gz)."""
    p = tmp_path / "pred_metagenome_unstrat.tsv.gz"
    content = (
        "#KO Abundances\n"
        "function\tSampleA\tSampleB\tSampleC\n"
        "ko:K00001\t100.0\t50.0\t0.0\n"
        "ko:K00002\t0.0\t75.0\t25.0\n"
        "ko:K00003\t25.0\t25.0\t50.0\n"
    )
    with gzip.open(p, 'wt') as f:
        f.write(content)
    return p

# --- Unit Tests (with mocks) ---

@patch('polars.read_csv')
@patch('importlib.resources.as_file')
def test_load_pathways_db_lv3(mock_as_file, mock_read_csv, mock_pathways_df):
    """Test loading pathways at Lv3 level."""
    # Setup mocks - return RAW data, function does processing
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_path
    mock_context.__exit__.return_value = None
    mock_as_file.return_value = mock_context
    mock_read_csv.return_value = mock_pathways_df
    
    # Execute - REAL function processes the raw data
    result = load_pathways_db(pgpt_level="Lv3")
    
    # Assert - validate TRANSFORMED output
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ['KO', 'Lv3']  # NOT renamed to PGPT_ID
    assert len(result) == 3  # 4th entry filtered (invalid KO pattern)
    
    # Check KO extraction (REGEX + PREFIX) was done correctly
    kos = result['KO'].to_list()
    assert 'ko:K00001' in kos
    assert 'ko:K00002' in kos
    assert 'ko:K00003' in kos
    # ko:K99999 would be extracted from "PGPT_004-gene-K99999" if it existed
    # But "PGPT_004-invalid" has no KO pattern, so it's filtered
    
    # Check PGPT values
    pgpts = result['Lv3'].to_list()
    assert 'NITROGEN_FIXATION' in pgpts
    assert 'P_SOLUBILIZATION' in pgpts
    assert 'ANTIBIOTIC_PRODUCTION' in pgpts

@patch('polars.read_csv')
@patch('importlib.resources.as_file')
def test_load_pathways_db_lv1(mock_as_file, mock_read_csv, mock_pathways_df):
    """Test loading pathways at Lv1 level."""
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_path
    mock_context.__exit__.return_value = None
    mock_as_file.return_value = mock_context
    mock_read_csv.return_value = mock_pathways_df
    
    result = load_pathways_db(pgpt_level="Lv1")
    
    assert result.columns == ['KO', 'Lv1']
    assert 'BIO-FERTILIZATION' in result['Lv1'].to_list()
    assert 'BIOCONTROL' in result['Lv1'].to_list()

@patch('polars.read_csv')
@patch('importlib.resources.as_file')
def test_load_pathways_db_invalid_level(mock_as_file, mock_read_csv, mock_pathways_df):
    """Test error handling for invalid PGPT level."""
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_path
    mock_context.__exit__.return_value = None
    mock_as_file.return_value = mock_context
    mock_read_csv.return_value = mock_pathways_df
    
    with pytest.raises(ValueError, match="PGPT level 'Lv99' not found"):
        load_pathways_db(pgpt_level="Lv99")

@patch('importlib.resources.as_file')
def test_load_pathways_db_missing_file(mock_as_file):
    """Test error handling when bundled database is missing."""
    mock_context = MagicMock()
    mock_context.__enter__.side_effect = FileNotFoundError("File not found")
    mock_as_file.return_value = mock_context
    
    with pytest.raises(RuntimeError, match="Failed to load bundled database"):
        load_pathways_db(pgpt_level="Lv3")

@patch('polars.read_csv')
@patch('importlib.resources.as_file')
def test_load_pathways_db_filters_null_kos(mock_as_file, mock_read_csv):
    """Test that entries without valid KOs are filtered out."""
    # DataFrame with some PGPT_IDs that don't have KO pattern
    df_with_nulls = pl.DataFrame({
        "PGPT_ID": ["PGPT_001-gene-K00001", "PGPT_002-invalid", "PGPT_003"],
        "Lv3": ["VALID_1", "INVALID_1", "INVALID_2"]
    })
    
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_path
    mock_context.__exit__.return_value = None
    mock_as_file.return_value = mock_context
    mock_read_csv.return_value = df_with_nulls
    
    result = load_pathways_db(pgpt_level="Lv3")
    
    # Only the valid KO should remain
    assert len(result) == 1
    assert result['KO'][0] == 'ko:K00001'
    assert result['Lv3'][0] == 'VALID_1'

# --- Tests for generate_unstratified_pgpt ---

@patch('pgptracker.analysis.unstratified.load_pathways_db')
def test_generate_unstratified_pgpt_lv3(mock_load_pathways, mock_ko_unstrat_file, tmp_path):
    """Test full unstratified PGPT generation at Lv3."""
    # Mock pathways database (already processed by load_pathways_db which has its own tests)
    mock_pathways = pl.DataFrame({
        "KO": ["ko:K00001", "ko:K00002", "ko:K00003"],
        "Lv3": ["NITROGEN_FIXATION", "P_SOLUBILIZATION", "ANTIBIOTIC_PRODUCTION"]
    })
    mock_load_pathways.return_value = mock_pathways
    
    output_dir = tmp_path / "output"
    
    # Execute - tests REAL logic: unpivot, join, aggregate, pivot, save
    result_path = generate_unstratified_pgpt(
        unstrat_ko_path=mock_ko_unstrat_file,
        output_dir=output_dir,
        pgpt_level="Lv3"
    )
    
    # Assert file created
    assert result_path.exists()
    assert result_path.name == "unstratified_pgpt_Lv3_abundances.tsv"
    
    # Read and validate output
    df = pl.read_csv(result_path, separator='\t')
    
    # Check structure
    assert 'Lv3' in df.columns  # Column name preserved
    assert 'SampleA' in df.columns
    assert 'SampleB' in df.columns
    assert 'SampleC' in df.columns
    
    # Validate MATHEMATICAL CORRECTNESS (not replicating calculation in test)
    # Input: K00001 = [100, 50, 0], K00002 = [0, 75, 25], K00003 = [25, 25, 50]
    # Expected: each KO maps to one PGPT (no aggregation needed at Lv3)
    
    n_fix = df.filter(pl.col('Lv3') == 'NITROGEN_FIXATION')
    assert len(n_fix) == 1
    assert n_fix['SampleA'][0] == 100.0  # Direct from K00001
    assert n_fix['SampleB'][0] == 50.0
    assert n_fix['SampleC'][0] == 0.0
    
    p_sol = df.filter(pl.col('Lv3') == 'P_SOLUBILIZATION')
    assert p_sol['SampleA'][0] == 0.0  # Direct from K00002
    assert p_sol['SampleB'][0] == 75.0
    assert p_sol['SampleC'][0] == 25.0
    
    anti = df.filter(pl.col('Lv3') == 'ANTIBIOTIC_PRODUCTION')
    assert anti['SampleA'][0] == 25.0  # Direct from K00003
    assert anti['SampleB'][0] == 25.0
    assert anti['SampleC'][0] == 50.0

@patch('pgptracker.analysis.unstratified.load_pathways_db')
def test_generate_unstratified_pgpt_lv1_aggregation(mock_load_pathways, mock_ko_unstrat_file, tmp_path):
    """Test aggregation when multiple KOs map to same PGPT."""
    # Mock: K00001 and K00002 BOTH map to same Lv1 category
    mock_pathways = pl.DataFrame({
        "KO": ["ko:K00001", "ko:K00002", "ko:K00003"],
        "Lv1": ["BIO-FERTILIZATION", "BIO-FERTILIZATION", "BIOCONTROL"]
    })
    mock_load_pathways.return_value = mock_pathways
    
    output_dir = tmp_path / "output"
    
    result_path = generate_unstratified_pgpt(
        unstrat_ko_path=mock_ko_unstrat_file,
        output_dir=output_dir,
        pgpt_level="Lv1"
    )
    
    df = pl.read_csv(result_path, separator='\t')
    
    # Should have 2 PGPT categories
    assert len(df) == 2
    
    # Validate AGGREGATION MATH (K00001 + K00002 -> BIO-FERTILIZATION)
    # K00001: [100, 50, 0]
    # K00002: [0, 75, 25]
    # Sum:    [100, 125, 25]
    bio_fert = df.filter(pl.col('Lv1') == 'BIO-FERTILIZATION')
    assert bio_fert['SampleA'][0] == 100.0
    assert bio_fert['SampleB'][0] == 125.0
    assert bio_fert['SampleC'][0] == 25.0
    
    # BIOCONTROL = K00003 only
    biocontrol = df.filter(pl.col('Lv1') == 'BIOCONTROL')
    assert biocontrol['SampleA'][0] == 25.0
    assert biocontrol['SampleB'][0] == 25.0
    assert biocontrol['SampleC'][0] == 50.0

@patch('pgptracker.analysis.unstratified.load_pathways_db')
def test_generate_unstratified_pgpt_filters_zeros(mock_load_pathways, tmp_path):
    """Test that zero abundances are properly handled."""
    # Create input with many zeros
    input_path = tmp_path / "input.tsv.gz"
    content = (
        "function\tS1\tS2\n"
        "ko:K00001\t100.0\t0.0\n"
        "ko:K00002\t0.0\t0.0\n"  # All zeros
        "ko:K00003\t0.0\t50.0\n"
    )
    with gzip.open(input_path, 'wt') as f:
        f.write(content)
    
    mock_pathways = pl.DataFrame({
        "KO": ["ko:K00001", "ko:K00002", "ko:K00003"],
        "Lv3": ["PGPT_A", "PGPT_B", "PGPT_C"]
    })
    mock_load_pathways.return_value = mock_pathways
    
    output_dir = tmp_path / "output"
    
    result_path = generate_unstratified_pgpt(
        unstrat_ko_path=input_path,
        output_dir=output_dir,
        pgpt_level="Lv3"
    )
    
    df = pl.read_csv(result_path, separator='\t')
    
    assert len(df) == 2
    assert 'PGPT_B' not in df['Lv3'].to_list()
    # pgpt_b = df.filter(pl.col('Lv3') == 'PGPT_B')
    # assert pgpt_b['S1'][0] == 0.0
    # assert pgpt_b['S2'][0] == 0.0

def test_generate_unstratified_pgpt_missing_input(tmp_path):
    """Test error handling for missing input file."""
    fake_path = tmp_path / "nonexistent.tsv.gz"
    output_dir = tmp_path / "output"
    
    with pytest.raises(FileNotFoundError, match="Unstratified KO file not found"):
        generate_unstratified_pgpt(
            unstrat_ko_path=fake_path,
            output_dir=output_dir,
            pgpt_level="Lv3"
        )

@patch('pgptracker.analysis.unstratified.load_pathways_db')
def test_generate_unstratified_pgpt_no_matching_kos(mock_load_pathways, tmp_path):
    """Test behavior when no KOs match between input and pathways."""
    # Create input with KOs not in pathways
    input_path = tmp_path / "input.tsv.gz"
    content = (
        "function\tS1\n"
        "ko:K99999\t100.0\n"  # Not in pathways
    )
    with gzip.open(input_path, 'wt') as f:
        f.write(content)
    
    mock_pathways = pl.DataFrame({
        "KO": ["ko:K00001"],
        "Lv3": ["PGPT_A"]
    })
    mock_load_pathways.return_value = mock_pathways
    
    output_dir = tmp_path / "output"
    
    result_path = generate_unstratified_pgpt(
        unstrat_ko_path=input_path,
        output_dir=output_dir,
        pgpt_level="Lv3"
    )
    
    # Output should exist but be empty (only headers)
    df = pl.read_csv(result_path, separator='\t')
    assert len(df) == 0
    assert 'Lv3' in df.columns

# --- Integration Test (NO mocks) ---

def test_load_pathways_db_integration(mock_pathways_file):
    """Integration test: load REAL file without mocks."""
    # Temporarily point to test file
    with patch('importlib.resources.files') as mock_files:
        mock_files.return_value.joinpath.return_value = mock_pathways_file
        
        result = load_pathways_db(pgpt_level="Lv3")
        
        # Validate ENTIRE pipeline worked (only 3 valid KOs)
        assert len(result) == 3
        assert result.columns == ['KO', 'Lv3']
        
        # Check transformations
        assert 'ko:K00001' in result['KO'].to_list()
        assert 'NITROGEN_FIXATION' in result['Lv3'].to_list()
        
        # Test different level
        result_lv1 = load_pathways_db(pgpt_level="Lv1")
        assert 'BIO-FERTILIZATION' in result_lv1['Lv1'].to_list()