"""
Unit tests for qiime/export module.

Run with: pytest tests/unit/test_qiime_export.py -v
"""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pgptracker.qiime.export_module import (
    export_qza_files,
    _export_sequences_qza,
    _export_table_qza,
    _copy_file
)


@pytest.fixture
def temp_dir():
    """Creates a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_qza_sequences(temp_dir):
    """Creates a mock .qza sequences file."""
    qza_path = temp_dir / "rep_seqs.qza"
    qza_path.write_text("dummy qza content")
    return qza_path


@pytest.fixture
def mock_qza_table(temp_dir):
    """Creates a mock .qza feature table."""
    qza_path = temp_dir / "feature_table.qza"
    qza_path.write_text("dummy qza content")
    return qza_path


@pytest.fixture
def mock_fna_file(temp_dir):
    """Creates a mock .fna file."""
    fna_path = temp_dir / "sequences.fna"
    fna_path.write_text(">ASV_001\nATCG\n")
    return fna_path


@pytest.fixture
def mock_biom_file(temp_dir):
    """Creates a mock .biom file."""
    biom_path = temp_dir / "table.biom"
    biom_path.write_text("dummy biom content")
    return biom_path


@pytest.fixture
def qza_inputs(mock_qza_sequences, mock_qza_table):
    """Creates inputs dict with .qza files."""
    return {
        'sequences': mock_qza_sequences,
        'table': mock_qza_table,
        'seq_format': 'qza',
        'table_format': 'qza'
    }


@pytest.fixture
def fasta_inputs(mock_fna_file, mock_biom_file):
    """Creates inputs dict with .fna/.biom files."""
    return {
        'sequences': mock_fna_file,
        'table': mock_biom_file,
        'seq_format': 'fasta',
        'table_format': 'biom'
    }


class TestCopyFile:
    """Tests for _copy_file helper function."""
    
    def test_copy_file_success(self, temp_dir):
        """Test successful file copy."""
        src = temp_dir / "source.txt"
        src.write_text("test content")
        dst = temp_dir / "destination.txt"
        
        result = _copy_file(src, dst)
        
        assert result == dst
        assert dst.exists()
        assert dst.read_text() == "test content"
    
    def test_copy_file_nonexistent_source(self, temp_dir):
        """Test copying non-existent file."""
        src = temp_dir / "nonexistent.txt"
        dst = temp_dir / "destination.txt"
        
        with pytest.raises(RuntimeError) as exc_info:
            _copy_file(src, dst)
        
        assert "Failed to copy" in str(exc_info.value)


class TestExportSequencesQza:
    """Tests for _export_sequences_qza function."""
    
    @patch('pgptracker.qiime.export_module.run_command')
    def test_export_sequences_success(self, mock_run, mock_qza_sequences, temp_dir):
        """Test successful sequences export."""
        # Setup: create the file that QIIME2 would create
        export_subdir = temp_dir / "exported_sequences"
        export_subdir.mkdir()
        exported_file = export_subdir / "dna-sequences.fasta"
        exported_file.write_text(">ASV_001\nATCG\n")
        
        # Mock run_command to do nothing (file already created above)
        mock_run.return_value = MagicMock(returncode=0)
        
        result = _export_sequences_qza(mock_qza_sequences, temp_dir)
        
        # Verify command was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        assert call_args[0] == "qiime"
        assert "qiime" in call_args[1]
        assert "tools" in call_args[1]
        assert "export" in call_args[1]
        
        # Verify result
        assert result == temp_dir / "dna-sequences.fna"
        assert result.exists()
        assert result.read_text() == ">ASV_001\nATCG\n"
        
        # Verify cleanup
        assert not export_subdir.exists()
    
    @patch('pgptracker.qiime.export_module.run_command')
    def test_export_sequences_qiime_fails(self, mock_run, mock_qza_sequences, temp_dir):
        """Test when QIIME2 export command fails."""
        mock_run.side_effect = Exception("QIIME2 error")
        
        with pytest.raises(RuntimeError) as exc_info:
            _export_sequences_qza(mock_qza_sequences, temp_dir)
        
        assert "Failed to export sequences" in str(exc_info.value)
    
    @patch('pgptracker.qiime.export_module.run_command')
    def test_export_sequences_file_not_found(self, mock_run, mock_qza_sequences, temp_dir):
        """Test when exported file is not created."""
        # Mock successful run but don't create the file
        mock_run.return_value = MagicMock(returncode=0)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            _export_sequences_qza(mock_qza_sequences, temp_dir)
        
        assert "Expected exported file not found" in str(exc_info.value)


class TestExportTableQza:
    """Tests for _export_table_qza function."""
    
    @patch('pgptracker.qiime.export_module.run_command')
    def test_export_table_success(self, mock_run, mock_qza_table, temp_dir):
        """Test successful table export."""
        # Setup: create the file that QIIME2 would create
        export_subdir = temp_dir / "exported_table"
        export_subdir.mkdir()
        exported_file = export_subdir / "feature-table.biom"
        exported_file.write_text("biom content")
        
        mock_run.return_value = MagicMock(returncode=0)
        
        result = _export_table_qza(mock_qza_table, temp_dir)
        
        # Verify command was called
        mock_run.assert_called_once()
        
        # Verify result
        assert result == temp_dir / "feature-table.biom"
        assert result.exists()
        assert result.read_text() == "biom content"
        
        # Verify cleanup
        assert not export_subdir.exists()
    
    @patch('pgptracker.qiime.export_module.run_command')
    def test_export_table_qiime_fails(self, mock_run, mock_qza_table, temp_dir):
        """Test when QIIME2 export command fails."""
        mock_run.side_effect = Exception("QIIME2 error")
        
        with pytest.raises(RuntimeError) as exc_info:
            _export_table_qza(mock_qza_table, temp_dir)
        
        assert "Failed to export feature table" in str(exc_info.value)


class TestExportQzaFiles:
    """Tests for main export_qza_files function."""
    
    @patch('pgptracker.qiime.export_module._export_sequences_qza')
    @patch('pgptracker.qiime.export_module._export_table_qza')
    def test_export_both_qza(self, mock_table, mock_seqs, qza_inputs, temp_dir):
        """Test exporting when both inputs are .qza."""
        # Setup mocks
        mock_seqs.return_value = temp_dir / "dna-sequences.fna"
        mock_table.return_value = temp_dir / "feature-table.biom"
        
        result = export_qza_files(qza_inputs, temp_dir)
        
        # Verify both export functions were called
        mock_seqs.assert_called_once()
        mock_table.assert_called_once()
        
        # Verify result structure
        assert 'sequences' in result
        assert 'table' in result
        assert result['sequences'] == temp_dir / "dna-sequences.fna"
        assert result['table'] == temp_dir / "feature-table.biom"
    
    @patch('pgptracker.qiime.export_module._copy_file')
    def test_export_both_fasta(self, mock_copy, fasta_inputs, temp_dir):
        """Test when both inputs are already .fna/.biom."""
        # Setup mock to return destination path
        def copy_side_effect(src, dst):
            return dst
        mock_copy.side_effect = copy_side_effect
        
        result = export_qza_files(fasta_inputs, temp_dir)
        
        # Verify copy was called twice
        assert mock_copy.call_count == 2
        
        # Verify result structure
        assert 'sequences' in result
        assert 'table' in result
    
    @patch('pgptracker.qiime.export_module._export_sequences_qza')
    @patch('pgptracker.qiime.export_module._copy_file')
    def test_export_mixed_formats(self, mock_copy, mock_export, temp_dir, mock_qza_sequences, mock_biom_file):
        """Test with mixed formats (.qza sequences + .biom table)."""
        inputs = {
            'sequences': mock_qza_sequences,
            'table': mock_biom_file,
            'seq_format': 'qza',
            'table_format': 'biom'
        }
        
        mock_export.return_value = temp_dir / "dna-sequences.fna"
        mock_copy.return_value = temp_dir / "feature-table.biom"
        
        result = export_qza_files(inputs, temp_dir)
        
        # Verify mixed calls
        mock_export.assert_called_once()  # For .qza sequences
        mock_copy.assert_called_once()     # For .biom table
        
        assert result['sequences'] == temp_dir / "dna-sequences.fna"
        assert result['table'] == temp_dir / "feature-table.biom"
    
    def test_creates_exports_directory(self, fasta_inputs, temp_dir):
        """Test that exports/ subdirectory is created."""
        with patch('pgptracker.qiime.export_module._copy_file') as mock_copy:
            mock_copy.return_value = temp_dir / "exports" / "file.txt"
            
            export_qza_files(fasta_inputs, temp_dir)
            
            exports_dir = temp_dir / "exports"
            assert exports_dir.exists()
            assert exports_dir.is_dir()


class TestExportIntegration:
    """Integration-like tests (still mocked but test full flow)."""
    
    @patch('pgptracker.qiime.export_module.run_command')
    def test_full_qza_export_flow(self, mock_run, qza_inputs, temp_dir):
        """Test complete flow of exporting .qza files."""
        # Setup: simulate QIIME2 creating export files
        def create_export_files(*args, **kwargs):
            cmd = args[1]
            output_path = Path(cmd[cmd.index("--output-path") + 1])
            
            if "exported_sequences" in str(output_path):
                output_path.mkdir(parents=True, exist_ok=True)
                (output_path / "dna-sequences.fasta").write_text(">ASV\nATCG\n")
            elif "exported_table" in str(output_path):
                output_path.mkdir(parents=True, exist_ok=True)
                (output_path / "feature-table.biom").write_text("biom")
            
            return MagicMock(returncode=0)
        
        mock_run.side_effect = create_export_files
        
        result = export_qza_files(qza_inputs, temp_dir)
        
        # Verify both files were exported
        assert result['sequences'].exists()
        assert result['table'].exists()
        assert result['sequences'].read_text() == ">ASV\nATCG\n"
        assert result['table'].read_text() == "biom"
        
        # Verify cleanup happened (no subdirectories left)
        exports_dir = temp_dir / "exports"
        subdirs = [d for d in exports_dir.iterdir() if d.is_dir()]
        assert len(subdirs) == 0