"""
Unit tests for place_seqs.py module.

run by: pytest tests/unit/test_place_seqs.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from subprocess import CalledProcessError
from pgptracker.picrust.place_seqs import build_phylogenetic_tree, _validate_output


class TestBuildPhylogeneticTree:
    """Tests for build_phylogenetic_tree function."""
    
    def test_successful_tree_building(self, tmp_path):
        """Test successful phylogenetic tree construction."""
        # Setup
        sequences = tmp_path / "test_seqs.fna"
        sequences.write_text(">seq1\nACGT\n")
        output_dir = tmp_path / "output"
        
        expected_tree = output_dir / "placed_seqs.tre"
        
        # Mock run_command
        mock_run = Mock()
        with patch('pgptracker.picrust.place_seqs.run_command', mock_run):
            # Create output file to simulate success
            output_dir.mkdir()
            expected_tree.write_text("(seq1:0.5);")
            
            # Execute
            result = build_phylogenetic_tree(sequences, output_dir, threads=4)
        
        # Verify
        assert result == expected_tree
        mock_run.assert_called_once()
        
        # Check command structure
        call_args = mock_run.call_args
        
        assert call_args[0][0] == "Picrust2" 
        
        cmd = call_args[0][1]
        assert "place_seqs.py" in cmd
        assert str(sequences) in cmd
        assert str(expected_tree) in cmd
        assert "4" in cmd
        assert "sepp" in cmd
        
        assert "--ref_dir" not in cmd
    
    def test_missing_sequences_file(self, tmp_path):
        """Test error when sequences file doesn't exist."""
        sequences = tmp_path / "nonexistent.fna"
        output_dir = tmp_path / "output"
        
        with pytest.raises(FileNotFoundError, match="Sequences file not found"):
            build_phylogenetic_tree(sequences, output_dir)
    
    def test_empty_sequences_file(self, tmp_path):
        """Test error when sequences file is empty."""
        sequences = tmp_path / "empty.fna"
        sequences.touch()
        output_dir = tmp_path / "output"
        
        with pytest.raises(RuntimeError, match="Sequences file is empty"):
            build_phylogenetic_tree(sequences, output_dir)
    
    def test_picrust_failure(self, tmp_path):
        """Test handling of PICRUSt2 failure."""
        sequences = tmp_path / "test_seqs.fna"
        sequences.write_text(">seq1\nACGT\n")
        output_dir = tmp_path / "output"
        
        # Mock run_command to raise error
        mock_run = Mock(side_effect=CalledProcessError(1, "place_seqs.py"))
        
        with patch('pgptracker.picrust.place_seqs.run_command', mock_run):
            with pytest.raises(CalledProcessError):
                build_phylogenetic_tree(sequences, output_dir)
    
    def test_missing_output(self, tmp_path):
        """Test error when PICRUSt2 doesn't create output."""
        sequences = tmp_path / "test_seqs.fna"
        sequences.write_text(">seq1\nACGT\n")
        output_dir = tmp_path / "output"
        
        mock_run = Mock()
        with patch('pgptracker.picrust.place_seqs.run_command', mock_run):
            # Don't create output file to simulate failure
            output_dir.mkdir()
            
            with pytest.raises(FileNotFoundError, match="did not create expected output"):
                build_phylogenetic_tree(sequences, output_dir)
    
    def test_empty_output(self, tmp_path):
        """Test error when PICRUSt2 creates empty output."""
        sequences = tmp_path / "test_seqs.fna"
        sequences.write_text(">seq1\nACGT\n")
        output_dir = tmp_path / "output"
        expected_tree = output_dir / "placed_seqs.tre"
        
        mock_run = Mock()
        with patch('pgptracker.picrust.place_seqs.run_command', mock_run):
            output_dir.mkdir()
            expected_tree.touch()  # Empty file
            
            with pytest.raises(RuntimeError, match="created empty output file"):
                build_phylogenetic_tree(sequences, output_dir)

class TestValidateOutput:
    """Tests for _validate_output helper function."""
    
    def test_valid_output(self, tmp_path):
        """Test validation passes for valid file."""
        output_file = tmp_path / "output.tre"
        output_file.write_text("valid content")
        
        # Should not raise
        _validate_output(output_file, "test_tool")
    
    def test_missing_output(self, tmp_path):
        """Test validation fails for missing file."""
        output_file = tmp_path / "missing.tre"
        
        with pytest.raises(FileNotFoundError, match="did not create expected output"):
            _validate_output(output_file, "test_tool")
    
    def test_empty_output(self, tmp_path):
        """Test validation fails for empty file."""
        output_file = tmp_path / "empty.tre"
        output_file.touch()
        
        with pytest.raises(RuntimeError, match="created empty output file"):
            _validate_output(output_file, "test_tool")