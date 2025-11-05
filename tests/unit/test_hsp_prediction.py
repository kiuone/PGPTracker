"""
Unit tests for hsp_prediction.py module.

run by: pytest tests/unit/test_hsp_prediction.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, call
from subprocess import CalledProcessError
from pgptracker.picrust.hsp_prediction import predict_gene_content, _run_hsp_prediction


class TestPredictGeneContent:
    """Tests for predict_gene_content function."""
    
    def test_successful_predictions(self, tmp_path):
        """Test successful marker and KO predictions."""
        # Setup
        tree = tmp_path / "placed_seqs.tre"
        tree.write_text("(seq1:0.5);")
        output_dir = tmp_path / "output"
        
        expected_marker = output_dir / "marker_nsti_predicted.tsv.gz"
        expected_ko = output_dir / "KO_predicted.tsv.gz"
        
        # Mock run_command
        mock_run = Mock()
        with patch('pgptracker.picrust.hsp_prediction.run_command', mock_run):
            # Create output files to simulate success
            output_dir.mkdir()
            expected_marker.write_text("marker content")
            expected_ko.write_text("ko content")
            
            # Execute
            result = predict_gene_content(tree, output_dir, threads=4, chunk_size=500)
        
        # Verify
        assert result['marker'] == expected_marker
        assert result['ko'] == expected_ko
        assert mock_run.call_count == 2
        
        # Verify both calls were made
        calls = mock_run.call_args_list
        assert len(calls) == 2
        
        assert calls[0][0][0] == "Picrust2"
        marker_cmd = calls[0][0][1]
        assert "16S" in marker_cmd
        assert "-n" in marker_cmd
        
        assert calls[1][0][0] == "Picrust2"
        ko_cmd = calls[1][0][1]
        assert "KO" in ko_cmd
        assert "-n" not in ko_cmd
    
    def test_missing_tree_file(self, tmp_path):
        """Test error when tree file doesn't exist."""
        tree = tmp_path / "nonexistent.tre"
        output_dir = tmp_path / "output"
        
        with pytest.raises(FileNotFoundError, match="Tree file not found"):
            predict_gene_content(tree, output_dir)
    
    def test_empty_tree_file(self, tmp_path):
        """Test error when tree file is empty."""
        tree = tmp_path / "empty.tre"
        tree.touch()
        output_dir = tmp_path / "output"
        
        with pytest.raises(RuntimeError, match="Tree file is empty"):
            predict_gene_content(tree, output_dir)
    
    def test_marker_prediction_fails(self, tmp_path):
        """Test handling when marker prediction fails."""
        tree = tmp_path / "placed_seqs.tre"
        tree.write_text("(seq1:0.5);")
        output_dir = tmp_path / "output"
        
        # Mock run_command to fail on first call
        mock_run = Mock(side_effect=CalledProcessError(1, "hsp.py"))
        
        with patch('pgptracker.picrust.hsp_prediction.run_command', mock_run):
            with pytest.raises(CalledProcessError):
                predict_gene_content(tree, output_dir)
    
    def test_ko_prediction_fails(self, tmp_path):
        """Test handling when KO prediction fails."""
        tree = tmp_path / "placed_seqs.tre"
        tree.write_text("(seq1:0.5);")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        marker_path = output_dir / "marker_nsti_predicted.tsv.gz"
        
        # Mock to succeed first call, fail second
        def side_effect(*args, **kwargs):
            # Acessa o nome do ambiente e o comando
            env = args[0]
            cmd_list = args[1]
            
            # Simula a primeira chamada (marker)
            if "16S" in cmd_list:
                marker_path.write_text("marker content")
            # Simula a segunda chamada (KO)
            elif "KO" in cmd_list:
                raise CalledProcessError(1, "hsp.py")
        
        mock_run = Mock(side_effect=side_effect)
        
        # Corrigido: patch usa o nome de arquivo correto
        with patch('pgptracker.picrust.hsp_prediction.run_command', mock_run):
            with pytest.raises(CalledProcessError):
                predict_gene_content(tree, output_dir)


class TestRunHspPrediction:
    """Tests for _run_hsp_prediction helper function."""
    
    def test_marker_prediction_with_nsti(self, tmp_path):
        """Test marker prediction includes NSTI flag."""
        tree = tmp_path / "placed_seqs.tre"
        tree.write_text("(seq1:0.5);")
        output = tmp_path / "marker.tsv.gz"
        
        mock_run = Mock()
        with patch('pgptracker.picrust.hsp_prediction.run_command', mock_run):
            output.write_text("content")
            
            result = _run_hsp_prediction(
                prediction_type="16S",
                tree_path=tree,
                output_path=output,
                threads=2,
                chunk_size=500,
                calculate_nsti=True
            )
        
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "Picrust2"
        
        # Verify command includes -n flag
        cmd = mock_run.call_args[0][1]
        assert "-n" in cmd
        assert "16S" in cmd
        assert "500" in cmd
    
    def test_ko_prediction_without_nsti(self, tmp_path):
        """Test KO prediction excludes NSTI flag."""
        tree = tmp_path / "placed_seqs.tre"
        tree.write_text("(seq1:0.5);")
        output = tmp_path / "ko.tsv.gz"
        
        mock_run = Mock()
        with patch('pgptracker.picrust.hsp_prediction.run_command', mock_run):
            output.write_text("content")
            
            result = _run_hsp_prediction(
                prediction_type="KO",
                tree_path=tree,
                output_path=output,
                threads=2,
                chunk_size=500,
                calculate_nsti=False
            )
        
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "Picrust2"
        
        # Verify command excludes -n flag
        cmd = mock_run.call_args[0][1]
        assert "-n" not in cmd
        assert "KO" in cmd
    
    def test_missing_output_file(self, tmp_path):
        """Test error when hsp.py doesn't create output."""
        tree = tmp_path / "placed_seqs.tre"
        tree.write_text("(seq1:0.5);")
        output = tmp_path / "output.tsv.gz"
        
        mock_run = Mock()
        with patch('pgptracker.picrust.hsp_prediction.run_command', mock_run):
            # Don't create output file
            
            with pytest.raises(FileNotFoundError, match="did not create expected output"):
                _run_hsp_prediction(
                    prediction_type="KO",
                    tree_path=tree,
                    output_path=output,
                    threads=1,
                    chunk_size=1000,
                    calculate_nsti=False
                )
    
    def test_empty_output_file(self, tmp_path):
        """Test error when hsp.py creates empty output."""
        tree = tmp_path / "placed_seqs.tre"
        tree.write_text("(seq1:0.5);")
        output = tmp_path / "output.tsv.gz"
        
        mock_run = Mock()
        with patch('pgptracker.picrust.hsp_prediction.run_command', mock_run):
            output.touch()  # Empty file
            
            with pytest.raises(RuntimeError, match="created empty output file"):
                _run_hsp_prediction(
                    prediction_type="KO",
                    tree_path=tree,
                    output_path=output,
                    threads=1,
                    chunk_size=1000,
                    calculate_nsti=False
                )