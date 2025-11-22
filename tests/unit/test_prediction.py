"""
Unit tests for stage1_processing/prediction.py module.

Tests functional prediction using R/Castor with adaptive RAM-aware batching.

Run with: pytest tests/unit/test_prediction.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from subprocess import CompletedProcess
import polars as pl
from pgptracker.stage1_processing.prediction import (
    predict_functional_profiles,
    _get_r_script,
    _calculate_optimal_chunk_size,
    _predict_marker_16s,
    _predict_ko_adaptive,
    _process_batch,
    _run_r_script
)


class TestGetRScript:
    """Tests for _get_r_script function."""

    def test_returns_path_object(self):
        """Test that function returns a Path object."""
        with patch('importlib.resources.files') as mock_files:
            mock_resource = MagicMock()
            mock_resource.__truediv__ = MagicMock(return_value="/path/to/script.R")
            mock_files.return_value = mock_resource

            result = _get_r_script("hsp.R")

            assert isinstance(result, Path)

    def test_fallback_to_resources_path(self):
        """Test fallback when files() is not available."""
        with patch('importlib.resources.files', side_effect=AttributeError):
            with patch('importlib.resources.path') as mock_path:
                mock_path.return_value.__enter__ = MagicMock(return_value=Path("/script.R"))
                mock_path.return_value.__exit__ = MagicMock(return_value=None)

                result = _get_r_script("nsti.R")

                assert isinstance(result, Path)


class TestCalculateOptimalChunkSize:
    """Tests for _calculate_optimal_chunk_size function."""

    def test_single_pass_with_sufficient_ram(self, tmp_path):
        """Test single-pass strategy when RAM is sufficient."""
        ko_db = tmp_path / "ko.txt.gz"

        df = pl.DataFrame({
            "genome": ["g1", "g2"],
            **{f"KO{i:04d}": [1.0, 2.0] for i in range(100)}
        })
        df.write_csv(ko_db, separator="\t")

        result = _calculate_optimal_chunk_size(ko_db, available_ram_gb=64.0)

        assert result['strategy'] == 'single_pass'
        assert result['num_batches'] == 1
        assert result['chunk_size'] == 100

    def test_batching_with_insufficient_ram(self, tmp_path):
        """Test batching strategy when RAM is constrained."""
        ko_db = tmp_path / "ko.txt.gz"

        df = pl.DataFrame({
            "genome": ["g1", "g2"],
            **{f"KO{i:04d}": [1.0, 2.0] for i in range(1000)}
        })
        df.write_csv(ko_db, separator="\t")

        result = _calculate_optimal_chunk_size(ko_db, available_ram_gb=2.0)

        assert result['strategy'] == 'batching'
        assert result['num_batches'] > 1
        assert 500 <= result['chunk_size'] <= 5000

    def test_auto_detect_ram(self, tmp_path):
        """Test automatic RAM detection."""
        ko_db = tmp_path / "ko.txt.gz"

        df = pl.DataFrame({
            "genome": ["g1"],
            **{f"KO{i:04d}": [1.0] for i in range(50)}
        })
        df.write_csv(ko_db, separator="\t")

        with patch('pgptracker.stage1_processing.prediction.detect_free_memory', return_value=16.0):
            result = _calculate_optimal_chunk_size(ko_db)

            assert 'chunk_size' in result
            assert 'num_batches' in result
            assert 'strategy' in result


class TestPredictFunctionalProfiles:
    """Tests for predict_functional_profiles function."""

    def test_successful_prediction(self, tmp_path):
        """Test successful functional profile prediction."""
        tree_path = tmp_path / "tree.tre"
        output_dir = tmp_path / "output"
        ref_dir = tmp_path / "ref"

        tree_path.write_text("(seq1:0.5);")
        ref_dir.mkdir()

        (ref_dir / "16S.txt.gz").write_text("genome\t16S\ng1\t4\n")
        (ref_dir / "ko.txt.gz").write_text("genome\tKO0001\ng1\t1.5\n")

        with patch('pgptracker.stage1_processing.prediction._predict_marker_16s') as mock_marker, \
             patch('pgptracker.stage1_processing.prediction._predict_ko_adaptive') as mock_ko:

            mock_marker.return_value = output_dir / "marker.tsv.gz"
            mock_ko.return_value = output_dir / "ko.tsv.gz"

            result = predict_functional_profiles(tree_path, output_dir, ref_dir, threads=2)

            assert result['marker'] == output_dir / "marker.tsv.gz"
            assert result['ko'] == output_dir / "ko.tsv.gz"
            mock_marker.assert_called_once()
            mock_ko.assert_called_once()

    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created if missing."""
        tree_path = tmp_path / "tree.tre"
        output_dir = tmp_path / "nonexistent" / "output"
        ref_dir = tmp_path / "ref"

        tree_path.write_text("(seq1:0.5);")
        ref_dir.mkdir()
        (ref_dir / "16S.txt.gz").write_text("genome\t16S\ng1\t4\n")
        (ref_dir / "ko.txt.gz").write_text("genome\tKO0001\ng1\t1.5\n")

        with patch('pgptracker.stage1_processing.prediction._predict_marker_16s') as mock_marker, \
             patch('pgptracker.stage1_processing.prediction._predict_ko_adaptive') as mock_ko:

            mock_marker.return_value = output_dir / "marker.tsv.gz"
            mock_ko.return_value = output_dir / "ko.tsv.gz"

            predict_functional_profiles(tree_path, output_dir, ref_dir)

            assert output_dir.exists()


class TestPredictMarker16s:
    """Tests for _predict_marker_16s function."""

    def test_successful_marker_prediction(self, tmp_path):
        """Test successful marker gene prediction with NSTI."""
        tree = tmp_path / "tree.tre"
        db_path = tmp_path / "16S.txt.gz"
        output_dir = tmp_path / "output"

        tree.write_text("(seq1:0.5);")
        output_dir.mkdir()

        marker_df = pl.DataFrame({
            "genome": ["g1", "g2"],
            "16S": [4, 5]
        })
        marker_df.write_csv(db_path, separator="\t")

        with patch('pgptracker.stage1_processing.prediction._get_r_script') as mock_get_script, \
             patch('pgptracker.stage1_processing.prediction._run_r_script') as mock_run_r:

            mock_get_script.side_effect = [Path("/hsp.R"), Path("/nsti.R")]

            def create_outputs(*args):
                if "hsp" in str(args[0]):
                    hsp_out = Path(args[1][2])
                    pl.DataFrame({"sequence": ["seq1"], "16S": [4]}).write_csv(hsp_out, separator="\t")
                elif "nsti" in str(args[0]):
                    nsti_out = Path(args[1][2])
                    pl.DataFrame({"sequence": ["seq1"], "metadata_NSTI": [0.05]}).write_csv(nsti_out, separator="\t")

            mock_run_r.side_effect = create_outputs

            result = _predict_marker_16s(tree, db_path, output_dir)

            assert result.exists()
            assert mock_run_r.call_count == 2


class TestPredictKoAdaptive:
    """Tests for _predict_ko_adaptive function."""

    def test_single_pass_mode(self, tmp_path):
        """Test single-pass mode when chunk_size >= total columns."""
        tree = tmp_path / "tree.tre"
        db_path = tmp_path / "ko.txt.gz"
        output_path = tmp_path / "ko_predicted.tsv.gz"

        tree.write_text("(seq1:0.5);")

        ko_df = pl.DataFrame({
            "genome": ["g1", "g2"],
            "KO0001": [1.5, 2.0],
            "KO0002": [0.5, 1.0]
        })
        ko_df.write_csv(db_path, separator="\t")

        with patch('pgptracker.stage1_processing.prediction._get_r_script', return_value=Path("/hsp.R")), \
             patch('pgptracker.stage1_processing.prediction._run_r_script') as mock_run_r:

            def create_output(*args):
                output_file = Path(args[1][2])
                pl.DataFrame({
                    "sequence": ["seq1"],
                    "KO0001": [1.5],
                    "KO0002": [0.5]
                }).write_csv(output_file, separator="\t")

            mock_run_r.side_effect = create_output

            result = _predict_ko_adaptive(tree, db_path, output_path, chunk_size=-1, threads=1)

            assert result == output_path
            assert output_path.exists()
            mock_run_r.assert_called_once()

    def test_batched_mode(self, tmp_path):
        """Test batched mode with chunking."""
        tree = tmp_path / "tree.tre"
        db_path = tmp_path / "ko.txt.gz"
        output_path = tmp_path / "ko_predicted.tsv.gz"

        tree.write_text("(seq1:0.5);")

        ko_df = pl.DataFrame({
            "genome": ["g1"],
            **{f"KO{i:04d}": [float(i)] for i in range(100)}
        })
        ko_df.write_csv(db_path, separator="\t")

        call_count = 0

        def mock_batch_side_effect(*args, **kwargs):
            nonlocal call_count
            # Each batch returns different KO columns
            if call_count == 0:
                # First batch: KO0000-KO0049
                result = pl.DataFrame({
                    "sequence": ["seq1"],
                    **{f"KO{i:04d}": [float(i)] for i in range(50)}
                })
            else:
                # Second batch: KO0050-KO0099
                result = pl.DataFrame({
                    "sequence": ["seq1"],
                    **{f"KO{i:04d}": [float(i)] for i in range(50, 100)}
                })
            call_count += 1
            return result

        with patch('pgptracker.stage1_processing.prediction._get_r_script'), \
             patch('pgptracker.stage1_processing.prediction._process_batch', side_effect=mock_batch_side_effect) as mock_batch:

            result = _predict_ko_adaptive(tree, db_path, output_path, chunk_size=50, threads=1)

            assert result == output_path
            assert mock_batch.call_count >= 2

    def test_auto_detect_chunk_size(self, tmp_path):
        """Test automatic chunk size detection."""
        tree = tmp_path / "tree.tre"
        db_path = tmp_path / "ko.txt.gz"
        output_path = tmp_path / "ko_predicted.tsv.gz"

        tree.write_text("(seq1:0.5);")

        ko_df = pl.DataFrame({
            "genome": ["g1"],
            **{f"KO{i:04d}": [1.0] for i in range(50)}
        })
        ko_df.write_csv(db_path, separator="\t")

        with patch('pgptracker.stage1_processing.prediction._calculate_optimal_chunk_size') as mock_calc, \
             patch('pgptracker.stage1_processing.prediction._get_r_script'), \
             patch('pgptracker.stage1_processing.prediction._run_r_script') as mock_run:

            mock_calc.return_value = {
                'chunk_size': 100,
                'num_batches': 1,
                'strategy': 'single_pass',
                'message': 'RAM sufficient'
            }

            def create_output(*args):
                output_file = Path(args[1][2])
                pl.DataFrame({"sequence": ["seq1"]}).write_csv(output_file, separator="\t")

            mock_run.side_effect = create_output

            result = _predict_ko_adaptive(tree, db_path, output_path, chunk_size=0, threads=1)

            mock_calc.assert_called_once()
            assert result.exists()


class TestProcessBatch:
    """Tests for _process_batch function."""

    def test_successful_batch_processing(self, tmp_path):
        """Test successful processing of a single batch."""
        db_path = tmp_path / "ko.txt.gz"
        tree = tmp_path / "tree.tre"
        hsp_script = Path("/hsp.R")
        temp_dir = tmp_path / "temp"

        tree.write_text("(seq1:0.5);")
        temp_dir.mkdir()

        ko_df = pl.DataFrame({
            "genome": ["g1", "g2"],
            "KO0001": [1.5, 2.0],
            "KO0002": [0.5, 1.0],
            "KO0003": [1.0, 1.5]
        })
        ko_df.write_csv(db_path, separator="\t")

        with patch('pgptracker.stage1_processing.prediction._run_r_script') as mock_run_r:
            def create_output(*args):
                output_file = Path(args[1][2])
                pl.DataFrame({
                    "sequence": ["seq1"],
                    "KO0001": [1.5],
                    "KO0002": [0.5]
                }).write_csv(output_file, separator="\t")

            mock_run_r.side_effect = create_output

            batch_cols = ["KO0001", "KO0002"]
            result = _process_batch(1, batch_cols, db_path, "genome", tree, hsp_script, temp_dir)

            assert isinstance(result, pl.DataFrame)
            assert "sequence" in result.columns
            assert "KO0001" in result.columns
            assert "KO0002" in result.columns


class TestRunRScript:
    """Tests for _run_r_script function."""

    def test_successful_r_execution(self):
        """Test successful R script execution."""
        script_path = Path("/path/to/script.R")
        args = ["arg1", "arg2", "arg3"]

        mock_result = CompletedProcess(args=[], returncode=0, stdout="R output", stderr="")

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            _run_r_script(script_path, args)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "Rscript"
            assert str(script_path) in cmd
            assert all(arg in cmd for arg in args)

    def test_r_script_failure(self):
        """Test handling of R script execution failure."""
        script_path = Path("/path/to/script.R")
        args = ["arg1"]

        mock_result = CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="Error in script: invalid data"
        )

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RuntimeError, match="R script failed: script.R"):
                _run_r_script(script_path, args)

    def test_r_command_structure(self):
        """Test that R command has correct structure."""
        script_path = Path("/scripts/hsp.R")
        args = ["/tree.tre", "/trait.tsv", "/output.tsv", "mp"]

        mock_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            _run_r_script(script_path, args)

            cmd = mock_run.call_args[0][0]
            assert cmd == ["Rscript", str(script_path)] + args
