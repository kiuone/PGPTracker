"""
Unit tests for qiime/export module functions.

Run with: pytest tests/unit/test_qiime_export.py -v
"""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
import subprocess # Necessário para testar a falha do comando

from pgptracker.qiime.export_module import (
    export_qza_files,
    _run_export,  # <-- Importa a nova função auxiliar
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
class TestRunExport:
    """Tests for the generic _run_export helper function."""
    
    @patch('pgptracker.qiime.export_module.shutil.rmtree')
    @patch('pgptracker.qiime.export_module.shutil.move')
    @patch('pgptracker.qiime.export_module.run_command')
    def test_run_export_success(self, mock_run, mock_move, mock_rmtree, mock_qza_sequences, temp_dir):
        """Test successful generic export flow."""
        
        # Args para a função
        qza_path = mock_qza_sequences
        export_dir = temp_dir / "exports"
        temp_subdir_name = "temp_seqs"
        expected_filename = "dna-sequences.fasta"
        final_filename = "dna-sequences.fna"
        
        # Simula o QIIME2 criando o arquivo
        temp_export_path = export_dir / temp_subdir_name
        temp_export_path.mkdir(parents=True, exist_ok=True)
        (temp_export_path / expected_filename).write_text(">ASV\nATCG")
        
        mock_run.return_value = MagicMock(returncode=0)
        
        # Executa a função
        result = _run_export(
            qza_path, export_dir, temp_subdir_name,
            expected_filename, final_filename
        )
        
        # Verifica o resultado
        final_path = export_dir / final_filename
        assert result == final_path
        
        # Verifica se o comando foi chamado corretamente
        expected_cmd = [
            "qiime", "tools", "export",
            "--input-path", str(qza_path),
            "--output-path", str(temp_export_path)
        ]
        mock_run.assert_called_once_with(
            "qiime", expected_cmd, check=True, capture_output=True
        )
        
        # Verifica se o arquivo foi movido
        mock_move.assert_called_once_with(
            str(temp_export_path / expected_filename), str(final_path)
        )
        
        # Verifica se o diretório temporário foi limpo
        assert mock_rmtree.call_count == 2

    @patch('pgptracker.qiime.export_module.run_command')
    def test_run_export_qiime_fails(self, mock_run, mock_qza_sequences, temp_dir):
        """Test that the exception from run_command is propagated."""
        
        # Simula o erro que o env_manager.py levanta
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="qiime tools export", stderr="QIIME2 crashed"
        )
        
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            _run_export(
                mock_qza_sequences, temp_dir / "exports", "temp", "a.fasta", "b.fna"
            )
        
        # Verifica se o erro original foi mantido
        assert "QIIME2 crashed" in str(exc_info.value.stderr)

    @patch('pgptracker.qiime.export_module.run_command')
    def test_run_export_file_not_found(self, mock_run, mock_qza_sequences, temp_dir):
        """Test when export runs but expected file is not created."""
        
        # Simula o QIIME2 rodando com sucesso, mas NÃO criando o arquivo
        mock_run.return_value = MagicMock(returncode=0)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            _run_export(
                mock_qza_sequences, temp_dir / "exports", "temp",
                "nonexistent.fasta", "b.fna"
            )
        
        assert "Expected exported file not found" in str(exc_info.value)

class TestExportQzaFiles:
    """Tests for main export_qza_files function."""
    
    @patch('pgptracker.qiime.export_module._run_export')
    def test_export_both_qza(self, mock_run_export, qza_inputs, temp_dir):
        """Test exporting when both inputs are .qza."""
        
        # Configura o mock para retornar os caminhos finais
        # O side_effect é chamado em ordem
        mock_run_export.side_effect = [
            temp_dir / "exports" / "dna-sequences.fna", # 1ª chamada
            temp_dir / "exports" / "feature-table.biom" # 2ª chamada
        ]
        
        result = export_qza_files(qza_inputs, temp_dir)
        
        # Verifica se _run_export foi chamado duas vezes
        assert mock_run_export.call_count == 2
        
        # Verifica a primeira chamada (sequências)
        call_args_seq = mock_run_export.call_args_list[0][1] # kwargs da 1ª chamada
        assert call_args_seq['qza_path'] == qza_inputs['sequences']
        assert call_args_seq['expected_filename'] == "dna-sequences.fasta"
        assert call_args_seq['final_filename'] == "dna-sequences.fna"
        
        # Verifica a segunda chamada (tabela)
        call_args_table = mock_run_export.call_args_list[1][1] # kwargs da 2ª chamada
        assert call_args_table['qza_path'] == qza_inputs['table']
        assert call_args_table['expected_filename'] == "feature-table.biom"
        assert call_args_table['final_filename'] == "feature-table.biom"
        
        # Verifica o resultado
        assert result['sequences'] == temp_dir / "exports" / "dna-sequences.fna"
        assert result['table'] == temp_dir / "exports" / "feature-table.biom"
    
    @patch('pgptracker.qiime.export_module._copy_file')
    def test_export_both_fasta(self, mock_copy, fasta_inputs, temp_dir):
        """Test when both inputs are already .fna/.biom."""
        
        # Simula o _copy_file retornando o destino
        def copy_side_effect(src, dst):
            return dst
        mock_copy.side_effect = copy_side_effect
        
        result = export_qza_files(fasta_inputs, temp_dir)
        
        # Verifica se _copy_file foi chamado duas vezes
        assert mock_copy.call_count == 2
        
        # Verifica o resultado
        assert result['sequences'] == temp_dir / "exports" / "dna-sequences.fna"
        assert result['table'] == temp_dir / "exports" / "feature-table.biom"
    
    @patch('pgptracker.qiime.export_module._run_export')
    @patch('pgptracker.qiime.export_module._copy_file')
    def test_export_mixed_formats(self, mock_copy, mock_export, temp_dir, mock_qza_sequences, mock_biom_file):
        """Test with mixed formats (.qza sequences + .biom table)."""
        inputs = {
            'sequences': mock_qza_sequences,
            'table': mock_biom_file,
            'seq_format': 'qza',
            'table_format': 'biom'
        }
        
        # Configura os mocks
        mock_export.return_value = temp_dir / "exports" / "dna-sequences.fna"
        mock_copy.return_value = temp_dir / "exports" / "feature-table.biom"
        
        result = export_qza_files(inputs, temp_dir)
        
        # Verifica as chamadas
        mock_export.assert_called_once() # Para as sequências .qza
        mock_copy.assert_called_once()   # Para a tabela .biom
        
        assert result['sequences'] == temp_dir / "exports" / "dna-sequences.fna"
        assert result['table'] == temp_dir / "exports" / "feature-table.biom"
    
    def test_creates_exports_directory(self, fasta_inputs, temp_dir):
        """Test that exports/ subdirectory is created."""
        with patch('pgptracker.qiime.export_module._copy_file') as mock_copy:
            mock_copy.return_value = temp_dir / "exports" / "file.txt"
            
            export_qza_files(fasta_inputs, temp_dir)
            
            exports_dir = temp_dir / "exports"
            assert exports_dir.exists()
            assert exports_dir.is_dir()

class TestExportIntegration:
    """Integration-like tests (mocked run_command but test full flow)."""
    
    @patch('pgptracker.qiime.export_module.run_command')
    def test_full_qza_export_flow(self, mock_run, qza_inputs, temp_dir):
        """Test complete flow of exporting .qza files."""
        
        # Simula o QIIME2 criando os arquivos de exportação
        def create_export_files(*args, **kwargs):
            tool = args[0]
            cmd = args[1]
            
            # Encontra o --output-path no comando
            output_path_str = cmd[cmd.index("--output-path") + 1]
            output_path = Path(output_path_str)
            
            if "exported_sequences" in str(output_path):
                output_path.mkdir(parents=True, exist_ok=True)
                (output_path / "dna-sequences.fasta").write_text(">ASV\nATCG\n")
            elif "exported_table" in str(output_path):
                output_path.mkdir(parents=True, exist_ok=True)
                (output_path / "feature-table.biom").write_text("biom")
            
            return MagicMock(returncode=0)
        
        mock_run.side_effect = create_export_files
        
        result = export_qza_files(qza_inputs, temp_dir)
        
        # Verifica se 'run_command' foi chamado 2 vezes (1 p/ seq, 1 p/ table)
        assert mock_run.call_count == 2
        
        # Verifica se os arquivos finais existem
        assert result['sequences'].exists()
        assert result['table'].exists()
        assert result['sequences'].read_text() == ">ASV\nATCG\n"
        assert result['table'].read_text() == "biom"
        
        # Verifica se a limpeza aconteceu (não devem sobrar subdiretórios)
        exports_dir = temp_dir / "exports"
        subdirs = [d for d in exports_dir.iterdir() if d.is_dir()]
        assert len(subdirs) == 0