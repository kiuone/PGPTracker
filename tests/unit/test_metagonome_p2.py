"""
Unit tests for metagenome_p2.py module.

run by: pytest tests/unit/test_metagonome_p2.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from subprocess import CalledProcessError
from pgptracker.picrust.metagenome_p2 import run_metagenome_pipeline


class TestRunMetagenomePipeline:
    """Tests for run_metagenome_pipeline function."""
    
    def test_successful_normalization(self, tmp_path):
        """Test successful abundance normalization."""
        # Setup input files
        table = tmp_path / "feature_table.biom"
        marker = tmp_path / "marker_nsti_predicted.tsv.gz"
        ko = tmp_path / "KO_predicted.tsv.gz"
        output_dir = tmp_path / "output"
        
        table.write_text("biom content")
        marker.write_text("marker content")
        ko.write_text("ko content")
        
        # Expected outputs
        metagenome_out = output_dir / "KO_metagenome_out"
        expected_norm = metagenome_out / "seqtab_norm.tsv.gz"
        expected_unstrat = metagenome_out / "pred_metagenome_unstrat.tsv.gz"
        
        # Mock run_command
        mock_run = Mock()
        with patch('pgptracker.picrust.metagenome_p2.run_command', mock_run):
            # Create output files to simulate success
            metagenome_out.mkdir(parents=True)
            expected_norm.write_text("normalized content")
            expected_unstrat.write_text("unstratified content")
            
            # Execute
            result = run_metagenome_pipeline(table, marker, ko, output_dir, max_nsti=2.0)
        
        # Verify
        assert result['seqtab_norm'] == expected_norm
        assert result['pred_metagenome_unstrat'] == expected_unstrat
        mock_run.assert_called_once()
        
        # Check command structure
        call_args = mock_run.call_args
        
        assert call_args[0][0] == "Picrust2"
        
        cmd = call_args[0][1]
        assert "metagenome_pipeline.py" in cmd
        assert str(table) in cmd
        assert str(marker) in cmd
        assert str(ko) in cmd
        assert "2.0" in cmd
        
        # Verify NO stratified flags
        assert "--strat_out" not in cmd
        assert "--wide_table" not in cmd
        assert "--metagenome_contrib" not in cmd
    
    def test_missing_table_file(self, tmp_path):
        """Test error when feature table doesn't exist."""
        table = tmp_path / "nonexistent.biom"
        marker = tmp_path / "marker.tsv.gz"
        ko = tmp_path / "ko.tsv.gz"
        output_dir = tmp_path / "output"
        
        marker.write_text("content")
        ko.write_text("content")
        
        with pytest.raises(FileNotFoundError, match="Feature table file not found"):
            run_metagenome_pipeline(table, marker, ko, output_dir)
    
    def test_missing_marker_file(self, tmp_path):
        """Test error when marker predictions don't exist."""
        table = tmp_path / "table.biom"
        marker = tmp_path / "nonexistent.tsv.gz"
        ko = tmp_path / "ko.tsv.gz"
        output_dir = tmp_path / "output"
        
        table.write_text("content")
        ko.write_text("content")
        
        with pytest.raises(FileNotFoundError, match="Marker predictions file not found"):
            run_metagenome_pipeline(table, marker, ko, output_dir)
    
    def test_missing_ko_file(self, tmp_path):
        """Test error when KO predictions don't exist."""
        table = tmp_path / "table.biom"
        marker = tmp_path / "marker.tsv.gz"
        ko = tmp_path / "nonexistent.tsv.gz"
        output_dir = tmp_path / "output"
        
        table.write_text("content")
        marker.write_text("content")
        
        with pytest.raises(FileNotFoundError, match="KO predictions file not found"):
            run_metagenome_pipeline(table, marker, ko, output_dir)
    
    def test_empty_input_files(self, tmp_path):
        """Test error when any input file is empty."""
        table = tmp_path / "table.biom"
        marker = tmp_path / "marker.tsv.gz"
        ko = tmp_path / "ko.tsv.gz"
        output_dir = tmp_path / "output"
        
        # Create empty files
        table.touch()
        marker.write_text("content")
        ko.write_text("content")
        
        with pytest.raises(RuntimeError, match="file is empty"):
            run_metagenome_pipeline(table, marker, ko, output_dir)
    
    def test_picrust_failure(self, tmp_path):
        """Test handling of PICRUSt2 failure."""
        table = tmp_path / "table.biom"
        marker = tmp_path / "marker.tsv.gz"
        ko = tmp_path / "ko.tsv.gz"
        output_dir = tmp_path / "output"
        
        table.write_text("content")
        marker.write_text("content")
        ko.write_text("content")
        
        # Mock run_command to raise error
        mock_run = Mock(side_effect=CalledProcessError(1, "metagenome_pipeline.py"))
        
        with patch('pgptracker.picrust.metagenome_p2.run_command', mock_run):
            with pytest.raises(CalledProcessError):
                run_metagenome_pipeline(table, marker, ko, output_dir)
    
    def test_missing_seqtab_output(self, tmp_path):
        """Test error when seqtab_norm is not created."""
        table = tmp_path / "table.biom"
        marker = tmp_path / "marker.tsv.gz"
        ko = tmp_path / "ko.tsv.gz"
        output_dir = tmp_path / "output"
        
        table.write_text("content")
        marker.write_text("content")
        ko.write_text("content")
        
        metagenome_out = output_dir / "KO_metagenome_out"
        expected_unstrat = metagenome_out / "pred_metagenome_unstrat.tsv.gz"
        
        mock_run = Mock()
        with patch('pgptracker.picrust.metagenome_p2.run_command', mock_run):
            # Only create unstratified file, not seqtab_norm
            metagenome_out.mkdir(parents=True)
            expected_unstrat.write_text("content")
            
            with pytest.raises(FileNotFoundError, match="Normalized sequence table"):
                run_metagenome_pipeline(table, marker, ko, output_dir)
    
    def test_missing_unstrat_output(self, tmp_path):
        """Test error when unstratified predictions are not created."""
        table = tmp_path / "table.biom"
        marker = tmp_path / "marker.tsv.gz"
        ko = tmp_path / "ko.tsv.gz"
        output_dir = tmp_path / "output"
        
        table.write_text("content")
        marker.write_text("content")
        ko.write_text("content")
        
        metagenome_out = output_dir / "KO_metagenome_out"
        expected_norm = metagenome_out / "seqtab_norm.tsv.gz"
        
        mock_run = Mock()
        with patch('pgptracker.picrust.metagenome_p2.run_command', mock_run):
            # Only create seqtab_norm, not unstratified
            metagenome_out.mkdir(parents=True)
            expected_norm.write_text("content")
            
            with pytest.raises(FileNotFoundError, match="Unstratified metagenome predictions"):
                run_metagenome_pipeline(table, marker, ko, output_dir)
    
    def test_custom_max_nsti(self, tmp_path):
        """Test custom max_nsti parameter is passed correctly."""
        table = tmp_path / "table.biom"
        marker = tmp_path / "marker.tsv.gz"
        ko = tmp_path / "ko.tsv.gz"
        output_dir = tmp_path / "output"
        
        table.write_text("content")
        marker.write_text("content")
        ko.write_text("content")
        
        metagenome_out = output_dir / "KO_metagenome_out"
        expected_norm = metagenome_out / "seqtab_norm.tsv.gz"
        expected_unstrat = metagenome_out / "pred_metagenome_unstrat.tsv.gz"
        
        mock_run = Mock()
        with patch('pgptracker.picrust.metagenome_p2.run_command', mock_run):
            metagenome_out.mkdir(parents=True)
            expected_norm.write_text("content")
            expected_unstrat.write_text("content")
            
            run_metagenome_pipeline(table, marker, ko, output_dir, max_nsti=3.5)
        
        # Verify custom max_nsti in command
        call_args = mock_run.call_args

        assert call_args[0][0] == "Picrust2"
        
        cmd = call_args[0][1]
        assert "3.5" in cmd