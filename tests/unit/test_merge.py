"""
Unit tests for the merge.py module.

Tests two parts:
1. The pure Polars logic in _process_taxonomy_polars.
2. The orchestration (mocked I/O) in merge_taxonomy_to_table.

Run with: pytest tests/unit/test_merge.py -v
"""

import pytest
import polars as pl
from polars.testing import assert_frame_equal
from pathlib import Path
from unittest.mock import patch, Mock, call
import gzip
import shutil
from subprocess import CalledProcessError

# Import functions to be tested
from pgptracker.utils.merge import merge_taxonomy_to_table, _process_taxonomy_polars
from pgptracker.utils.validator import ValidationError

# --- Fixtures ---

@pytest.fixture
def mock_lazy_frame() -> pl.LazyFrame:
    """Provides a realistic Polars LazyFrame for taxonomy processing."""
    # This data mimics the output of 'biom convert --to-tsv'
    data = {
        "ASV_ID": ["ASV_1", "ASV_2", "ASV_3", "ASV_4"],
        "SampleA": [10, 0, 5, 8],
        "SampleB": [20, 30, 0, 15],
        "taxonomy": [
            "k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacterales; f__Enterobacteriaceae; g__Escherichia; s__coli",
            "k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Lactobacillaceae; g__Lactobacillus", # Missing species
            "k__Archaea; p__Crenarchaeota; c__Thermoprotei; o__Sulfolobales; f__Sulfolobaceae; g__; s__", # Empty genus/species
            "k__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Bacteroides; s__fragilis"
        ],
        "confidence": [0.99, 0.95, 0.80, 0.98]
    }
    return pl.DataFrame(data).lazy()

@pytest.fixture
def mock_paths(tmp_path):
    """Provides a dictionary of mock Path objects for orchestration tests."""
    output_dir = tmp_path / "test_output"
    work_dir = output_dir / "merge_work"
    return {
        "output_dir": output_dir,
        "work_dir": work_dir,
        "seqtab_norm_gz": tmp_path / "seqtab_norm.tsv.gz",
        "taxonomy_tsv": tmp_path / "taxonomy.tsv",
        "seqtab_tsv": work_dir / "seqtab_norm.tsv",
        "seqtab_biom": work_dir / "seqtab_norm.biom",
        "merged_biom": work_dir / "feature_table_with_taxonomy.biom",
        "raw_merged_tsv": work_dir / "temp_merged_table.tsv",
        "final_tsv": output_dir / "norm_wt_feature_table.tsv"
    }


# --- Test 1: Pure Logic (_process_taxonomy_polars) ---

def test_process_taxonomy_polars_logic(mock_lazy_frame):
    """
    Tests the _process_taxonomy_polars function with a real Polars frame.
    This validates the splitting, cleaning, and reordering logic.
    """
    # Arrange
    # Input frame is mock_lazy_frame
    
    # Act
    # Call the function we want to test
    processed_lazy = _process_taxonomy_polars(mock_lazy_frame)
    result_df = processed_lazy.collect()

    # Assert
    # 1. Check final column order
    expected_columns = [
        'ASV_ID', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species',
        'SampleA', 'SampleB'
    ]
    assert result_df.columns == expected_columns

    # 2. Check taxonomy splitting and prefix cleaning
    assert result_df.item(0, "Kingdom") == "Bacteria"
    assert result_df.item(0, "Genus") == "Escherichia"
    assert result_df.item(0, "Species") == "coli"
    
    # 3. Check for correct handling of missing (short) taxonomy
    assert result_df.item(1, "Genus") == "Lactobacillus"
    assert result_df.item(1, "Species") is None # Missing species becomes None

    # 4. Check for correct handling of empty (g__) taxonomy
    assert result_df.item(2, "Genus") is None # 'g__' becomes None
    assert result_df.item(2, "Species") is None # 's__' becomes None
    
    # 5. Check sample data integrity
    assert result_df.item(3, "ASV_ID") == "ASV_4"
    assert result_df.item(3, "SampleB") == 15

# --- Test 2: Orchestration (merge_taxonomy_to_table) ---

@patch('pgptracker.utils.merge.shutil.rmtree')
@patch('pgptracker.utils.merge.pl.LazyFrame.write_csv')
@patch('pgptracker.utils.merge.pl.LazyFrame.collect')
@patch('pgptracker.utils.merge._process_taxonomy_polars')
@patch('pgptracker.utils.merge.pl.scan_csv')
@patch('pgptracker.utils.merge._validate_output')
@patch('pgptracker.utils.merge.run_command')
@patch('pgptracker.utils.merge.shutil.copyfileobj')
@patch('pgptracker.utils.merge.gzip.open')
def test_merge_taxonomy_happy_path_with_cleanup(
    mock_gzip_open, mock_copy, mock_run_command, mock_validate,
    mock_scan_csv, mock_process_tax, mock_collect, mock_write_csv,
    mock_rmtree, mock_paths
):
    """
    Tests the full orchestration of merge_taxonomy_to_table,
    simulating a successful run where BIOM steps are needed
    and cleanup is enabled.
    """
    # Arrange
    # Mock all I/O functions
    mock_paths["raw_merged_tsv"].exists.return_value = False # Force BIOM steps
    mock_lazy_frame = Mock(spec=pl.LazyFrame)
    mock_lazy_frame.rename.return_value = mock_lazy_frame
    mock_processed_frame = Mock(spec=pl.LazyFrame)
    mock_collected_frame = Mock()

    mock_scan_csv.return_value = mock_lazy_frame
    mock_process_tax.return_value = mock_processed_frame
    mock_processed_frame.collect.return_value = mock_collected_frame

    # Act
    result_path = merge_taxonomy_to_table(
        mock_paths["seqtab_norm_gz"],
        mock_paths["taxonomy_tsv"],
        mock_paths["output_dir"],
        save_intermediates=False # Test cleanup
    )

    # Assert
    # 1. Check final path
    assert result_path == mock_paths["final_tsv"]
    
    # 2. Check BIOM pipeline was called
    assert mock_gzip_open.called
    assert mock_run_command.call_count == 3
    
    # 3. Check Polars pipeline was called
    mock_scan_csv.assert_called_once_with(mock_paths["raw_merged_tsv"], separator='\t', skip_rows=1)
    mock_lazy_frame.rename.assert_called_once_with({'#OTUID': 'ASV_ID'})
    mock_process_tax.assert_called_once_with(mock_lazy_frame)
    
    # 4. Check Polars streaming save
    mock_processed_frame.collect.assert_called_with(engine="streaming")
    mock_collected_frame.write_csv.assert_called_once_with(mock_paths["final_tsv"], separator='\t')

    # 5. Check validation
    assert mock_validate.call_count == 5 # 1 gunzip + 3 biom + 1 polars

    # 6. Check cleanup
    mock_rmtree.assert_called_once_with(mock_paths["work_dir"])

@patch('pgptracker.utils.merge.shutil.rmtree')
@patch('pgptracker.utils.merge.pl.LazyFrame.write_csv')
@patch('pgptracker.utils.merge.pl.LazyFrame.collect')
@patch('pgptracker.utils.merge._process_taxonomy_polars')
@patch('pgptracker.utils.merge.pl.scan_csv')
@patch('pgptracker.utils.merge._validate_output')
@patch('pgptracker.utils.merge.run_command')
@patch('pgptracker.utils.merge.shutil.copyfileobj')
@patch('pgptracker.utils.merge.gzip.open')
def test_merge_taxonomy_skip_biom_and_save_intermediates(
    mock_gzip_open, mock_copy, mock_run_command, mock_validate,
    mock_scan_csv, mock_process_tax, mock_collect, mock_write_csv,
    mock_rmtree, mock_paths
):
    """
    Tests the orchestration when raw_merged_tsv already exists
    and save_intermediates=True.
    """
    # Arrange
    # Mock all I/O functions
    mock_paths["raw_merged_tsv"].exists.return_value = True # Skip BIOM steps
    mock_lazy_frame = Mock(spec=pl.LazyFrame)
    mock_lazy_frame.rename.return_value = mock_lazy_frame
    mock_processed_frame = Mock(spec=pl.LazyFrame)
    mock_collected_frame = Mock()

    mock_scan_csv.return_value = mock_lazy_frame
    mock_process_tax.return_value = mock_processed_frame
    mock_processed_frame.collect.return_value = mock_collected_frame

    # Act
    result_path = merge_taxonomy_to_table(
        mock_paths["seqtab_norm_gz"],
        mock_paths["taxonomy_tsv"],
        mock_paths["output_dir"],
        save_intermediates=True # Test no cleanup
    )

    # Assert
    # 1. Check BIOM pipeline was *not* called
    assert not mock_gzip_open.called
    assert mock_run_command.call_count == 0
    
    # 2. Check Polars pipeline *was* called
    mock_scan_csv.assert_called_once_with(mock_paths["raw_merged_tsv"], separator='\t', skip_rows=1)
    mock_process_tax.assert_called_once_with(mock_lazy_frame)
    mock_processed_frame.collect.assert_called_with(engine="streaming")
    
    # 3. Check validation
    # Only the final validation should be called
    mock_validate.assert_called_once_with(
        mock_paths["final_tsv"], "Polars Processing", "final processed table"
    )

    # 4. Check cleanup
    mock_rmtree.assert_not_called()

@patch('pgptracker.utils.merge.run_command', Mock(side_effect=CalledProcessError(1, "cmd")))
def test_merge_taxonomy_biom_fails(mock_paths):
    """Tests that a failure in run_command raises RuntimeError."""
    mock_paths["raw_merged_tsv"].exists.return_value = False
    
    with pytest.raises(RuntimeError, match="BIOM merging pipeline failed."):
        merge_taxonomy_to_table(
            mock_paths["seqtab_norm_gz"],
            mock_paths["taxonomy_tsv"],
            mock_paths["output_dir"]
        )

@patch('pgptracker.utils.merge.run_command', Mock())
@patch('pgptracker.utils.merge.pl.scan_csv', Mock(side_effect=Exception("Polars failed")))
def test_merge_taxonomy_polars_fails(mock_paths):
    """Tests that a failure in Polars processing raises RuntimeError."""
    mock_paths["raw_merged_tsv"].exists.return_value = True # Skip BIOM
    
    with pytest.raises(RuntimeError, match="Taxonomy processing failed."):
        merge_taxonomy_to_table(
            mock_paths["seqtab_norm_gz"],
            mock_paths["taxonomy_tsv"],
            mock_paths["output_dir"]
        )