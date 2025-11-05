"""
Unit tests for the classify.py module.
Updated to test both .qza and .fna input paths.
run by: pytest tests/unit/test_classify.py -v
"""

import pytest
from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import patch, Mock

# Import the function to be tested
from pgptracker.qiime.classify import classify_taxonomy

@pytest.fixture
def mock_paths(tmp_path):
    """Create fixture for input/output paths."""
    return {
        "rep_seqs_qza": tmp_path / "rep_seqs.qza",
        "rep_seqs_fna": tmp_path / "rep_seqs.fna", # For new test
        "classifier": tmp_path / "classifier.qza",
        "output_dir": tmp_path / "test_output",
        "imported_qza": tmp_path / "test_output/taxonomy/imported_rep_seqs.qza", # New intermediate
        "classified_qza": tmp_path / "test_output/taxonomy/taxonomy.qza",
        "export_dir": tmp_path / "test_output/taxonomy/exported_taxonomy",
        "exported_tsv": tmp_path / "test_output/taxonomy/exported_taxonomy/taxonomy.tsv",
        "final_tsv": tmp_path / "test_output/taxonomy/taxonomy.tsv"
    }

@patch('pgptracker.qiime.classify.run_command')
def test_classify_happy_path_with_QZA(mock_run_command, mock_paths):
    """
    Test successful execution when input is ALREADY .qza.
    Should only call run_command 2 times (classify, export).
    """
    # --- Setup ---
    def run_command_side_effect(env, cmd, check):
        if "classify-sklearn" in cmd:
            mock_paths["classified_qza"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["classified_qza"].touch()
        elif "export" in cmd:
            mock_paths["export_dir"].mkdir(parents=True, exist_ok=True)
            mock_paths["exported_tsv"].write_text("Feature-ID\ttaxonomy\tconfidence\nASV1\tk__Foo\t0.99")
        return Mock(returncode=0)
    
    mock_run_command.side_effect = run_command_side_effect

    # --- Execute ---
    result_path = classify_taxonomy(
        rep_seqs_path=mock_paths["rep_seqs_qza"], # Input is .qza
        seq_format='qza',                         # Format is 'qza'
        classifier_qza=mock_paths["classifier"],
        output_dir=mock_paths["output_dir"],
        threads=4
    )

    # --- Assert ---
    assert result_path == mock_paths["final_tsv"]
    assert mock_paths["final_tsv"].read_text().startswith("#OTUID\ttaxonomy\tconfidence\n")
    
    # 3. Check run_command calls -
    assert mock_run_command.call_count == 2 # NO import step
    calls = mock_run_command.call_args_list
    
    # Call 1 (classify-sklearn) - should use the *original* .qza
    assert calls[0][0][0] == "qiime"
    assert "classify-sklearn" in calls[0][0][1]
    assert str(mock_paths["rep_seqs_qza"]) in calls[0][0][1] # Uses original
    
    # Call 2 (tools export)
    assert calls[1][0][0] == "qiime"
    assert "export" in calls[1][0][1]

@patch('pgptracker.qiime.classify.run_command')
def test_classify_happy_path_with_FNA(mock_run_command, mock_paths):
     """
     Test successful execution when input is .fna.
     Should call run_command 3 times (import, classify, export).
     """
     # --- Setup ---
     def run_command_side_effect(env, cmd, check):
        # CORREÇÃO: Checa por 'import' (item da lista)
         if "import" in cmd: 
            mock_paths["imported_qza"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["imported_qza"].touch() # CORREÇÃO: Cria o arquivo
         elif "classify-sklearn" in cmd: 
            mock_paths["classified_qza"].parent.mkdir(parents=True, exist_ok=True)
            mock_paths["classified_qza"].touch() # CORREÇÃO: Cria o arquivo
         elif "export" in cmd: 
            mock_paths["export_dir"].mkdir(parents=True, exist_ok=True)
            mock_paths["exported_tsv"].write_text("Bad Header\nData") # CORREÇÃO: Cria o arquivo
         return Mock(returncode=0)
    
     mock_run_command.side_effect = run_command_side_effect

     # --- Execute ---
     result_path = classify_taxonomy(
         rep_seqs_path=mock_paths["rep_seqs_fna"],
         seq_format='fasta',
         classifier_qza=mock_paths["classifier"],
         output_dir=mock_paths["output_dir"],
         threads=4
     )

     # --- Assert ---
     assert result_path == mock_paths["final_tsv"]
     assert mock_run_command.call_count == 3
     calls = mock_run_command.call_args_list
    # CORREÇÃO: Checa por 'import', 'classify-sklearn', 'export'
     assert "import" in calls[0][0][1]
     assert "classify-sklearn" in calls[1][0][1]
     assert str(mock_paths["imported_qza"]) in calls[1][0][1] # Verifica se usou o arquivo importado
     assert "export" in calls[2][0][1]

@patch('pgptracker.qiime.classify.run_command')
def test_classify_fna_import_fails(mock_run_command, mock_paths):
    """Test failure at Step 0 (tools import) when using .fna."""
    # Fail on the *first* call (tools import)
    mock_run_command.side_effect = CalledProcessError(1, "cmd")
    
    with pytest.raises(RuntimeError, match="Failed to import .fna to .qza"):
        classify_taxonomy(
            rep_seqs_path=mock_paths["rep_seqs_fna"],
            seq_format='fasta',
            classifier_qza=mock_paths["classifier"],
            output_dir=mock_paths["output_dir"],
            threads=4
        )
    # Ensure it only tried to run the import command
    mock_run_command.assert_called_once()
    assert "import" in mock_run_command.call_args[0][1]

@patch('pgptracker.qiime.classify.run_command')
def test_classify_fails_with_qza(mock_run_command, mock_paths):
    """Test failure at Step 1 (classify-sklearn) when using .qza."""
    mock_run_command.side_effect = CalledProcessError(1, "cmd")

    with pytest.raises(RuntimeError, match="Taxonomic classification failed."):
        classify_taxonomy(
            rep_seqs_path=mock_paths["rep_seqs_qza"], # Test .qza path
            seq_format='qza',
            classifier_qza=mock_paths["classifier"],
            output_dir=mock_paths["output_dir"],
            threads=4
        )
    # Should only be called once (the classify-sklearn command)
    mock_run_command.assert_called_once()
    assert 'classify-sklearn' in mock_run_command.call_args[0][1]

@patch('pgptracker.qiime.classify.run_command')
def test_export_fails_with_qza(mock_run_command, mock_paths):
    """Test failure at Step 2 (tools export) when using .qza."""
    # Mock to succeed on call 1 (classify), fail on call 2 (export)
    def run_command_side_effect(env, cmd, check):
         if "classify-sklearn" in cmd:
                mock_paths["classified_qza"].parent.mkdir(parents=True, exist_ok=True)
                mock_paths["classified_qza"].touch()
                return Mock(returncode=0)
         elif "export" in cmd:
                raise CalledProcessError(1, "cmd")

    mock_run_command.side_effect = run_command_side_effect
    
    with pytest.raises(RuntimeError, match="Taxonomy export failed."):
        classify_taxonomy(
            rep_seqs_path=mock_paths["rep_seqs_qza"], # Test .qza path
            seq_format='qza',
            classifier_qza=mock_paths["classifier"],
            output_dir=mock_paths["output_dir"],
            threads=4
        )
    assert mock_run_command.call_count == 2

@patch('pgptracker.qiime.classify.run_command')
@patch('builtins.open', Mock(side_effect=IOError("Permission denied")))
def test_header_fix_fails_with_qza(mock_run_command, mock_paths):
    """Test failure at Step 3 (header fix) when using .qza."""
    # Mock run_command to succeed and create the file
    def run_command_side_effect(env, cmd, check):
         if "classify-sklearn" in cmd:
                    mock_paths["classified_qza"].parent.mkdir(parents=True, exist_ok=True)
                    mock_paths["classified_qza"].touch()
         elif "export" in cmd:
                    # CORREÇÃO: Precisa criar o diretório E o arquivo
                    mock_paths["export_dir"].mkdir(parents=True, exist_ok=True)
                    mock_paths["exported_tsv"].write_text("Bad Header\nData")
         return Mock(returncode=0) # CORREÇÃO: Adicionado retorno
    mock_run_command.side_effect = run_command_side_effect
    
    with pytest.raises(RuntimeError, match="Header fix failed."):
        classify_taxonomy(
            rep_seqs_path=mock_paths["rep_seqs_qza"], # Test .qza path
            seq_format='qza',
            classifier_qza=mock_paths["classifier"],
            output_dir=mock_paths["output_dir"],
            threads=4
        )
    assert mock_run_command.call_count == 2