"""
Unit tests for the merge.py module.

run with: pytest tests/unit/test_merge.py -v
"""

import pytest
from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import patch, Mock, call
import gzip
import shutil

# Import the function to be tested
from pgptracker.utils.merge import merge_taxonomy_to_table

@pytest.fixture
def mock_merge_paths(tmp_path):
    """Create fixture for input/output paths."""
    # Inputs
    seqtab_norm_gz = tmp_path / "seqtab_norm.tsv.gz"
    taxonomy_tsv = tmp_path / "taxonomy.tsv"
    
    # Working dir
    work_dir = tmp_path / "test_output" / "merge_work"
    
    # Intermediates
    seqtab_tsv = work_dir / "seqtab_norm.tsv"
    seqtab_biom = work_dir / "seqtab_norm.biom"
    merged_biom = work_dir / "feature_table_with_taxonomy.biom"
    
    # Final output
    final_tsv = tmp_path / "test_output" / "feature_table_with_taxonomy.tsv"
    
    # Create fake inputs
    with gzip.open(seqtab_norm_gz, 'wb') as f:
        # Convert bytes to string before passing to write
        data = b"fake gzip data"
        # decoded_data = data.decode("utf-8")  # Decode bytes to string
        f.write(data)
    taxonomy_tsv.write_text("fake taxonomy data")
    
    return {
        "seqtab_norm_gz": seqtab_norm_gz,
        "taxonomy_tsv": taxonomy_tsv,
        "output_dir": tmp_path / "test_output",
        "work_dir": work_dir,
        "seqtab_tsv": seqtab_tsv,
        "seqtab_biom": seqtab_biom,
        "merged_biom": merged_biom,
        "final_tsv": final_tsv
    }

@patch('pgptracker.utils.merge.shutil.rmtree')
@patch('pgptracker.utils.merge.run_command')
def test_merge_happy_path_and_cleanup(mock_run_command, mock_rmtree, mock_merge_paths):
    """
    Test the full, successful execution of the merge pipeline
    and verify that cleanup (rmtree) is called.
    """
    # --- Setup ---
    # Mock run_command to create dummy output files
    def run_command_side_effect(env, cmd, check):
        if "convert" in cmd and str(mock_merge_paths["seqtab_tsv"]) in cmd:
            mock_merge_paths["seqtab_biom"].touch() # Create 1st biom
        elif "add-metadata" in cmd:
            mock_merge_paths["merged_biom"].touch() # Create merged biom
        elif "convert" in cmd and str(mock_merge_paths["merged_biom"]) in cmd:
            mock_merge_paths["final_tsv"].write_text("final tsv content") # Create final tsv
        return Mock(returncode=0)
    
    mock_run_command.side_effect = run_command_side_effect

    # --- Execute ---
    result_path = merge_taxonomy_to_table(
        seqtab_norm_gz=mock_merge_paths["seqtab_norm_gz"],
        taxonomy_tsv=mock_merge_paths["taxonomy_tsv"],
        output_dir=mock_merge_paths["output_dir"],
        save_intermediates=False # Test cleanup
    )

    # --- Assert ---
    # 1. Check final path
    assert result_path == mock_merge_paths["final_tsv"]
    assert mock_merge_paths["final_tsv"].exists()

    # 2. Check run_command calls
    assert mock_run_command.call_count == 3
    calls = mock_run_command.call_args_list
    
    # Check environments
    assert calls[0][0][0] == "PGPTracker"
    assert calls[1][0][0] == "PGPTracker"
    assert calls[2][0][0] == "PGPTracker"
    
    # Check chain logic (output of step N is input of step N+1)
    cmd1_out = calls[0][0][1][calls[0][0][1].index("-o") + 1] # seqtab_biom
    cmd2_in = calls[1][0][1][calls[1][0][1].index("-i") + 1]  # (must be seqtab_biom)
    cmd2_out = calls[1][0][1][calls[1][0][1].index("-o") + 1] # merged_biom
    cmd3_in = calls[2][0][1][calls[2][0][1].index("-i") + 1]  # (must be merged_biom)
    
    assert cmd1_out == cmd2_in
    assert cmd2_out == cmd3_in

    # 3. Check cleanup
    mock_rmtree.assert_called_once_with(mock_merge_paths["work_dir"])

@patch('pgptracker.utils.merge.shutil.rmtree')
@patch('pgptracker.utils.merge.run_command')
def test_merge_save_intermediates(mock_run_command, mock_rmtree, mock_merge_paths):
    """Test that cleanup is skipped when save_intermediates=True."""
    
    # Mock run_command to succeed
    def run_command_side_effect(env, cmd, check):
        if "convert" in cmd and str(mock_merge_paths["seqtab_tsv"]) in cmd:
            mock_merge_paths["seqtab_biom"].touch()
        elif "add-metadata" in cmd:
            mock_merge_paths["merged_biom"].touch()
        elif "convert" in cmd and str(mock_merge_paths["merged_biom"]) in cmd:
            mock_merge_paths["final_tsv"].write_text("final tsv content")
    mock_run_command.side_effect = run_command_side_effect

    # --- Execute ---
    merge_taxonomy_to_table(
        seqtab_norm_gz=mock_merge_paths["seqtab_norm_gz"],
        taxonomy_tsv=mock_merge_paths["taxonomy_tsv"],
        output_dir=mock_merge_paths["output_dir"],
        save_intermediates=True # Test flag
    )

    # --- Assert ---
    # 3. Check cleanup
    mock_rmtree.assert_not_called()

@patch('gzip.open', Mock(side_effect=gzip.BadGzipFile("Not a gzip file")))
def test_merge_gunzip_fails(mock_merge_paths):
    """Test failure at Step 1 (gunzip)."""
    with pytest.raises(RuntimeError, match="BIOM merging pipeline failed."):
        merge_taxonomy_to_table(
            seqtab_norm_gz=mock_merge_paths["seqtab_norm_gz"],
            taxonomy_tsv=mock_merge_paths["taxonomy_tsv"],
            output_dir=mock_merge_paths["output_dir"]
        )

@patch('pgptracker.utils.merge.run_command', Mock(side_effect=CalledProcessError(1, "cmd")))
def test_merge_biom_convert1_fails(mock_merge_paths):
    """Test failure at Step 2 (biom convert 1)."""
    with pytest.raises(RuntimeError, match="BIOM merging pipeline failed."):
        merge_taxonomy_to_table(
            seqtab_norm_gz=mock_merge_paths["seqtab_norm_gz"],
            taxonomy_tsv=mock_merge_paths["taxonomy_tsv"],
            output_dir=mock_merge_paths["output_dir"]
        )

@patch('pgptracker.utils.merge.run_command')
def test_merge_biom_add_metadata_fails(mock_run_command, mock_merge_paths):
    """Test failure at Step 3 (biom add-metadata)."""
    # Fail on the *second* call
    mock_run_command.side_effect = [
        Mock(returncode=0), # Call 1 (convert 1) succeeds
        CalledProcessError(1, "cmd"), # Call 2 (merge) fails
        Mock(returncode=0) # Call 3 (not reached)
    ]
    
    with pytest.raises(RuntimeError, match="BIOM merging pipeline failed."):
        merge_taxonomy_to_table(
            seqtab_norm_gz=mock_merge_paths["seqtab_norm_gz"],
            taxonomy_tsv=mock_merge_paths["taxonomy_tsv"],
            output_dir=mock_merge_paths["output_dir"]
        )