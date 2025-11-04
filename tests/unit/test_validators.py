"""
Unit tests for validators module.

Run with: pytest tests/unit/test_validators.py -v
"""

import pytest
from pathlib import Path
import tempfile
import os
from pgptracker.utils.validator import validate_inputs, ValidationError

@pytest.fixture
def temp_dir():
    """Creates a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_fasta_file(temp_dir):
    """Creates a valid FASTA file for testing."""
    fasta_path = temp_dir / "rep_seqs.fna"
    with open(fasta_path, 'w') as f:
        f.write(">ASV_001\n")
        f.write("ATCGATCGATCG\n")
        f.write(">ASV_002\n")
        f.write("GCTAGCTAGCTA\n")
    return fasta_path


@pytest.fixture
def valid_biom_file(temp_dir):
    """Creates a valid BIOM file (just a non-empty file for testing)."""
    biom_path = temp_dir / "feature_table.biom"
    biom_path.write_text("dummy biom content")
    return biom_path


@pytest.fixture
def valid_qza_sequences(temp_dir):
    """Creates a mock .qza sequences file."""
    qza_path = temp_dir / "rep_seqs.qza"
    qza_path.write_text("dummy qza content")
    return qza_path


@pytest.fixture
def valid_qza_table(temp_dir):
    """Creates a mock .qza feature table file."""
    qza_path = temp_dir / "feature_table.qza"
    qza_path.write_text("dummy qza content")
    return qza_path


@pytest.fixture
def empty_file(temp_dir):
    """Creates an empty file."""
    empty_path = temp_dir / "empty.fna"
    empty_path.touch()
    return empty_path


class TestValidateInputsSuccess:
    """Tests for successful validation scenarios."""
    
    def test_valid_fasta_and_biom(self, valid_fasta_file, valid_biom_file, temp_dir):
        """Test with valid .fna and .biom files."""
        output_dir = temp_dir / "output"
        
        result = validate_inputs(
            str(valid_fasta_file),
            str(valid_biom_file),
            str(output_dir)
        )
        
        assert result['sequences'] == valid_fasta_file
        assert result['table'] == valid_biom_file
        assert result['output'] == output_dir
        assert result['seq_format'] == 'fasta'
        assert result['table_format'] == 'biom'
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_valid_qza_files(self, valid_qza_sequences, valid_qza_table, temp_dir):
        """Test with valid .qza files for both inputs."""
        output_dir = temp_dir / "output"
        
        result = validate_inputs(
            str(valid_qza_sequences),
            str(valid_qza_table),
            str(output_dir)
        )
        
        assert result['sequences'] == valid_qza_sequences
        assert result['table'] == valid_qza_table
        assert result['seq_format'] == 'qza'
        assert result['table_format'] == 'qza'
        assert output_dir.exists()
    
    def test_creates_output_directory(self, valid_fasta_file, valid_biom_file, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        output_dir = temp_dir / "new" / "nested" / "output"
        
        assert not output_dir.exists()
        
        result = validate_inputs(
            str(valid_fasta_file),
            str(valid_biom_file),
            str(output_dir)
        )
        
        assert output_dir.exists()
        assert output_dir.is_dir()
        assert result['output'] == output_dir
    
    def test_existing_output_directory(self, valid_fasta_file, valid_biom_file, temp_dir):
        """Test with existing output directory."""
        output_dir = temp_dir / "existing"
        output_dir.mkdir()
        
        result = validate_inputs(
            str(valid_fasta_file),
            str(valid_biom_file),
            str(output_dir)
        )
        
        assert result['output'] == output_dir
    
    def test_fasta_extension_variants(self, valid_biom_file, temp_dir):
        """Test that all FASTA extensions are accepted (.fna, .fasta, .fa)."""
        output_dir = temp_dir / "output"
        
        for ext in ['.fna', '.fasta', '.fa']:
            seq_file = temp_dir / f"sequences{ext}"
            seq_file.write_text(">seq1\nATCG\n")
            
            result = validate_inputs(
                str(seq_file),
                str(valid_biom_file),
                str(output_dir)
            )
            
            assert result['seq_format'] == 'fasta'


class TestValidateInputsSequenceErrors:
    """Tests for sequence file validation errors."""
    
    def test_sequences_not_found(self, valid_biom_file, temp_dir):
        """Test with non-existent sequences file."""
        output_dir = temp_dir / "output"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                "nonexistent.fna",
                str(valid_biom_file),
                str(output_dir)
            )
        
        assert "not found" in str(exc_info.value).lower()
        assert "nonexistent.fna" in str(exc_info.value)
    
    def test_sequences_empty_file(self, empty_file, valid_biom_file, temp_dir):
        """Test with empty sequences file."""
        output_dir = temp_dir / "output"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(empty_file),
                str(valid_biom_file),
                str(output_dir)
            )
        
        assert "empty" in str(exc_info.value).lower()
        assert "sequences" in str(exc_info.value).lower()
    
    def test_sequences_invalid_extension(self, valid_biom_file, temp_dir):
        """Test with invalid sequences file extension."""
        invalid_file = temp_dir / "sequences.txt"
        invalid_file.write_text("some content")
        output_dir = temp_dir / "output"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(invalid_file),
                str(valid_biom_file),
                str(output_dir)
            )
        
        assert "invalid" in str(exc_info.value).lower()
        assert ".txt" in str(exc_info.value)
    
    def test_sequences_is_directory(self, valid_biom_file, temp_dir):
        """Test when sequences path is a directory, not a file."""
        seq_dir = temp_dir / "sequences_dir"
        seq_dir.mkdir()
        output_dir = temp_dir / "output"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(seq_dir),
                str(valid_biom_file),
                str(output_dir)
            )
        
        assert "not a file" in str(exc_info.value).lower()


class TestValidateInputsTableErrors:
    """Tests for feature table validation errors."""
    
    def test_table_not_found(self, valid_fasta_file, temp_dir):
        """Test with non-existent feature table."""
        output_dir = temp_dir / "output"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(valid_fasta_file),
                "nonexistent.biom",
                str(output_dir)
            )
        
        assert "not found" in str(exc_info.value).lower()
        assert "nonexistent.biom" in str(exc_info.value)
    
    def test_table_invalid_extension(self, valid_fasta_file, temp_dir):
        """Test with invalid feature table extension."""
        invalid_table = temp_dir / "table.txt"
        invalid_table.write_text("some content")
        output_dir = temp_dir / "output"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(valid_fasta_file),
                str(invalid_table),
                str(output_dir)
            )
        
        assert "invalid" in str(exc_info.value).lower()
        assert "table" in str(exc_info.value).lower()
    
    def test_table_empty_file(self, valid_fasta_file, temp_dir):
        """Test with empty feature table file."""
        empty_table = temp_dir / "table.biom"
        empty_table.touch()
        output_dir = temp_dir / "output"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(valid_fasta_file),
                str(empty_table),
                str(output_dir)
            )
        
        assert "empty" in str(exc_info.value).lower()


class TestValidateInputsFormatMismatch:
    """Tests for format compatibility errors."""
    
    def test_qza_sequences_with_biom_table(self, valid_qza_sequences, valid_biom_file, temp_dir):
        """Test format mismatch: .qza sequences with .biom table."""
        output_dir = temp_dir / "output"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(valid_qza_sequences),
                str(valid_biom_file),
                str(output_dir)
            )
        
        error_msg = str(exc_info.value).lower()
        assert "mismatch" in error_msg
        assert ".qza" in str(exc_info.value)
        assert ".biom" in str(exc_info.value)
    
    def test_fasta_sequences_with_qza_table(self, valid_fasta_file, valid_qza_table, temp_dir):
        """Test format mismatch: .fna sequences with .qza table (Gemini's bug catch)."""
        output_dir = temp_dir / "output"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(valid_fasta_file),
                str(valid_qza_table),
                str(output_dir)
            )
        
        error_msg = str(exc_info.value).lower()
        assert "mismatch" in error_msg
        assert ".fna" in str(exc_info.value) or ".fasta" in str(exc_info.value)
        assert ".qza" in str(exc_info.value)


class TestValidateInputsOutputErrors:
    """Tests for output directory validation errors."""
    
    def test_output_is_file_not_directory(self, valid_fasta_file, valid_biom_file, temp_dir):
        """Test when output path exists but is a file, not a directory."""
        output_file = temp_dir / "output.txt"
        output_file.write_text("content")
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(valid_fasta_file),
                str(valid_biom_file),
                str(output_file)
            )
        
        assert "not a directory" in str(exc_info.value).lower()


class TestValidateInputsMultipleErrors:
    """Tests for scenarios with multiple validation errors."""
    
    def test_all_files_invalid(self, temp_dir):
        """Test when all inputs are invalid."""
        output_file = temp_dir / "output.txt"
        output_file.write_text("content")
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                "nonexistent_seqs.fna",
                "nonexistent_table.biom",
                str(output_file)
            )
        
        error_msg = str(exc_info.value).lower()
        # Should contain errors about both files and output
        assert "sequences" in error_msg or "seqs" in error_msg
        assert "table" in error_msg
        assert "output" in error_msg or "directory" in error_msg
    
    def test_combines_all_error_messages(self, empty_file, temp_dir):
        """Test that all errors are reported together."""
        output_file = temp_dir / "output.txt"
        output_file.write_text("content")
        
        with pytest.raises(ValidationError) as exc_info:
            validate_inputs(
                str(empty_file),
                "nonexistent.biom",
                str(output_file)
            )
        
        error_msg = str(exc_info.value)
        # Should contain multiple error messages (bullet points)
        assert error_msg.count('-') >= 2  # At least 2 bullet points
        assert "empty" in error_msg.lower()
        assert "not found" in error_msg.lower()
        assert "not a directory" in error_msg.lower()


class TestValidateInputsReturnValues:
    """Tests for correct return value structure."""
    
    def test_return_structure(self, valid_fasta_file, valid_biom_file, temp_dir):
        """Test that return dictionary has correct structure."""
        output_dir = temp_dir / "output"
        
        result = validate_inputs(
            str(valid_fasta_file),
            str(valid_biom_file),
            str(output_dir)
        )
        
        # Check all expected keys exist
        assert 'sequences' in result
        assert 'table' in result
        assert 'output' in result
        assert 'seq_format' in result
        assert 'table_format' in result
        
        # Check types
        assert isinstance(result['sequences'], Path)
        assert isinstance(result['table'], Path)
        assert isinstance(result['output'], Path)
        assert isinstance(result['seq_format'], str)
        assert isinstance(result['table_format'], str)
        
        # Check format values
        assert result['seq_format'] in ['qza', 'fasta']
        assert result['table_format'] in ['qza', 'biom']