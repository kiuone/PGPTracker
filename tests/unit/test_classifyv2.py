"""
Unit tests for the classify.py module.

run with: pytest tests/unit/test_classifyv2.py -v
"""

import pytest
from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import patch, Mock, call

# Importa o módulo (arquivo)
from pgptracker.qiime import classify 
from pgptracker.utils.validator import ValidationError # Importar para os testes

@pytest.fixture
def mock_paths(tmp_path):
    """Create fixture for input/output paths."""
    output_dir = tmp_path / "test_output"
    taxonomy_dir = output_dir / "taxonomy"
    # Este é o diretório que o QIIME2 irá criar
    export_dir = taxonomy_dir / "exported_taxonomy" 
    
    return {
        # --- Inputs ---
        "rep_seqs_qza": tmp_path / "rep_seqs.qza",       # Path for QZA input test
        "rep_seqs_fasta": tmp_path / "rep_seqs.fna",   # Path for FASTA input test
        "classifier": tmp_path / "classifier.qza",
        "output_dir": output_dir,
        
        # --- Intermediates ---
        "imported_qza": taxonomy_dir / "imported_rep_seqs.qza", 
        "classified_qza": taxonomy_dir / "taxonomy.qza",
        "export_dir": export_dir,
        "exported_tsv": export_dir / "taxonomy.tsv", 
        
        # --- Final Output ---
        "final_tsv": taxonomy_dir / "taxonomy.tsv"
    }

@patch('pgptracker.qiime.classify.run_command')
def test_classify_happy_path_with_QZA(mock_run_command, mock_paths):
    """
    Testa o caminho feliz quando a entrada é .qza (pula a importação)
    """
    # --- Setup ---
    def run_command_side_effect(env, cmd, check):
        if "classify-sklearn" in cmd:
            # Cria o arquivo E seu diretório pai
            mock_paths["classified_qza"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["classified_qza"].touch()
        
        # ***** CORREÇÃO ESTAVA AQUI *****
        # O 'cmd' é uma lista, não uma string.
        elif "tools" in cmd and "export" in cmd:
            # Cria o arquivo E seu diretório pai
            mock_paths["exported_tsv"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["exported_tsv"].write_text("Feature-ID\ttaxonomy\tconfidence\nASV1\tk__Foo\t0.99")
        return Mock(returncode=0)
    
    mock_run_command.side_effect = run_command_side_effect

    # --- Execute ---
    result_path = classify.classify_taxonomy(
        rep_seqs_path=mock_paths["rep_seqs_qza"], 
        seq_format='qza',                          
        classifier_qza=mock_paths["classifier"],
        output_dir=mock_paths["output_dir"],
        threads=4
    )

    # --- Assert ---
    assert result_path == mock_paths["final_tsv"]
    assert mock_paths["final_tsv"].exists()
    content = mock_paths["final_tsv"].read_text()
    assert content.startswith("#OTUID\ttaxonomy\tconfidence\n")
    assert "ASV1\tk__Foo\t0.99" in content
    assert mock_run_command.call_count == 2
    calls = mock_run_command.call_args_list
    assert "classify-sklearn" in calls[0][0][1]
    assert "tools" in calls[1][0][1] and "export" in calls[1][0][1]

@patch('pgptracker.qiime.classify.run_command')
def test_classify_happy_path_with_FNA(mock_run_command, mock_paths):
    """
    Testa o caminho feliz quando a entrada é .fna (inclui importação)
    """
    # --- Setup ---
    def run_command_side_effect(env, cmd, check):
        if "tools" in cmd and "import" in cmd:
            mock_paths["imported_qza"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["imported_qza"].touch()
        elif "classify-sklearn" in cmd:
            mock_paths["classified_qza"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["classified_qza"].touch()
        elif "tools" in cmd and "export" in cmd:
            mock_paths["exported_tsv"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["exported_tsv"].write_text("Feature-ID\ttaxonomy\tconfidence\nASV1\tk__Foo\t0.99")
        return Mock(returncode=0)
    
    mock_run_command.side_effect = run_command_side_effect

    # --- Execute ---
    result_path = classify.classify_taxonomy(
        rep_seqs_path=mock_paths["rep_seqs_fasta"], # Use FASTA input
        seq_format='fasta',                          # Specify 'fasta' format
        classifier_qza=mock_paths["classifier"],
        output_dir=mock_paths["output_dir"],
        threads=4
    )

    # --- Assert ---
    assert result_path == mock_paths["final_tsv"]
    assert mock_paths["final_tsv"].read_text().startswith("#OTUID\ttaxonomy\tconfidence\n")
    assert mock_run_command.call_count == 3
    calls = mock_run_command.call_args_list
    
    assert "tools" in calls[0][0][1] and "import" in calls[0][0][1]
    assert str(mock_paths["imported_qza"]) in calls[1][0][1] # Check input of step 2
    assert "tools" in calls[2][0][1] and "export" in calls[2][0][1]

@patch('pgptracker.qiime.classify.run_command', Mock(side_effect=CalledProcessError(1, "cmd", "import failed")))
def test_classify_fna_import_fails(mock_paths):
    """Test failure at the NEW Step 0 (qiime tools import)."""
    with pytest.raises(RuntimeError, match="Failed to import .fna to .qza"):
        classify.classify_taxonomy(
            rep_seqs_path=mock_paths["rep_seqs_fasta"], 
            seq_format='fasta',                          
            classifier_qza=mock_paths["classifier"],
            output_dir=mock_paths["output_dir"],
            threads=4
        )

@patch('pgptracker.qiime.classify.run_command', Mock(side_effect=CalledProcessError(1, "cmd", "classify failed")))
def test_classify_sklearn_fails_with_qza(mock_paths):
    """Test failure at Step 1 (classify-sklearn)."""
    with pytest.raises(RuntimeError, match="Taxonomic classification failed."):
        classify.classify_taxonomy(
            rep_seqs_path=mock_paths["rep_seqs_qza"], 
            seq_format='qza',                          
            classifier_qza=mock_paths["classifier"],
            output_dir=mock_paths["output_dir"],
            threads=4
        )

@patch('pgptracker.qiime.classify.run_command')
def test_export_fails_with_qza(mock_run_command, mock_paths):
    """
    Test failure at Step 2 (tools export).
    ***** CORREÇÃO ESTAVA AQUI *****
    """
    # --- Setup ---
    # Precisamos de um side_effect que CRIE o arquivo da etapa 1
    # e DEPOIS falhe na etapa 2.
    def run_command_side_effect(env, cmd, check):
        if "classify-sklearn" in cmd:
            # Etapa 1: Sucesso, cria o arquivo
            mock_paths["classified_qza"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["classified_qza"].touch()
            return Mock(returncode=0)
        elif "tools" in cmd and "export" in cmd:
            # Etapa 2: Falha
            raise CalledProcessError(1, "cmd", "export failed")
    
    mock_run_command.side_effect = run_command_side_effect
    
    # --- Execute & Assert ---
    with pytest.raises(RuntimeError, match="Taxonomy export failed."):
        classify.classify_taxonomy(
            rep_seqs_path=mock_paths["rep_seqs_qza"], 
            seq_format='qza',
            classifier_qza=mock_paths["classifier"],
            output_dir=mock_paths["output_dir"],
            threads=4
        )
    
    # Verificação extra: garantir que a etapa 1 foi chamada
    assert mock_run_command.call_count == 2


@patch('pgptracker.qiime.classify.run_command')
@patch('builtins.open', Mock(side_effect=IOError("Permission denied")))
def test_header_fix_fails_with_qza(mock_run_command, mock_paths):
    """Test failure at Step 3 (header fix) due to IO error."""
    def run_command_side_effect(env, cmd, check):
        if "classify-sklearn" in cmd:
            mock_paths["classified_qza"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["classified_qza"].touch()
        elif "tools" in cmd and "export" in cmd:
            # Cria o arquivo E seu diretório pai
            mock_paths["exported_tsv"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["exported_tsv"].write_text("Bad Header\nData")
    mock_run_command.side_effect = run_command_side_effect
    
    with pytest.raises(RuntimeError, match="Header fix failed."):
        classify.classify_taxonomy(
            rep_seqs_path=mock_paths["rep_seqs_qza"], 
            seq_format='qza',
            classifier_qza=mock_paths["classifier"],
            output_dir=mock_paths["output_dir"],
            threads=4
        )