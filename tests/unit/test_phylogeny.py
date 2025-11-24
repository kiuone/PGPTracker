"""
Unit tests for stage1_processing/phylogeny.py module.

Tests phylogenetic placement using SEPP and GAPPA.

Run with: pytest tests/unit/test_phylogeny.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from subprocess import CompletedProcess
from pgptracker.stage1_processing.b_phylogeny import place_sequences, _run_sepp, _run_gappa


class TestPlaceSequences:
    """Tests for place_sequences function."""

    def test_successful_placement(self, tmp_path):
        """Test successful phylogenetic placement workflow."""
        seqs_path = tmp_path / "input" / "seqs.fna"
        output_dir = tmp_path / "output"
        ref_dir = tmp_path / "ref" / "prokaryotic"

        seqs_path.parent.mkdir(parents=True)
        seqs_path.write_text(">seq1\nACGT\n")

        pro_ref_dir = ref_dir / "pro_ref"
        pro_ref_dir.mkdir(parents=True)
        (pro_ref_dir / "pro_ref.tre").write_text("(ref1:0.1);")
        (pro_ref_dir / "pro_ref.fna").write_text(">ref1\nACGT\n")
        (pro_ref_dir / "pro_ref.raxml_info").write_text("RAxML info")

        with patch('pgptracker.stage1_processing.phylogeny._run_sepp') as mock_sepp, \
             patch('pgptracker.stage1_processing.phylogeny._run_gappa') as mock_gappa:

            expected_jplace = output_dir / "placement_placement.json"
            mock_sepp.return_value = expected_jplace

            result = place_sequences(seqs_path, output_dir, ref_dir, threads=4)

            assert result == output_dir / "placed_seqs.tre"
            mock_sepp.assert_called_once()
            mock_gappa.assert_called_once()

    def test_missing_alignment_file(self, tmp_path):
        """Test error when reference alignment is missing."""
        seqs_path = tmp_path / "seqs.fna"
        output_dir = tmp_path / "output"
        ref_dir = tmp_path / "ref"

        seqs_path.write_text(">seq1\nACGT\n")
        pro_ref_dir = ref_dir / "pro_ref"
        pro_ref_dir.mkdir(parents=True)
        (pro_ref_dir / "pro_ref.tre").write_text("(ref1:0.1);")
        (pro_ref_dir / "pro_ref.raxml_info").write_text("RAxML info")

        with pytest.raises(FileNotFoundError, match="Reference alignment not found"):
            place_sequences(seqs_path, output_dir, ref_dir)

    def test_missing_tree_file(self, tmp_path):
        """Test error when reference tree is missing."""
        seqs_path = tmp_path / "seqs.fna"
        output_dir = tmp_path / "output"
        ref_dir = tmp_path / "ref"

        seqs_path.write_text(">seq1\nACGT\n")
        pro_ref_dir = ref_dir / "pro_ref"
        pro_ref_dir.mkdir(parents=True)
        (pro_ref_dir / "pro_ref.fna").write_text(">ref1\nACGT\n")
        (pro_ref_dir / "pro_ref.raxml_info").write_text("RAxML info")

        with pytest.raises(FileNotFoundError, match="Reference tree not found"):
            place_sequences(seqs_path, output_dir, ref_dir)

    def test_missing_raxml_info_file(self, tmp_path):
        """Test error when RAxML info file is missing."""
        seqs_path = tmp_path / "seqs.fna"
        output_dir = tmp_path / "output"
        ref_dir = tmp_path / "ref"

        seqs_path.write_text(">seq1\nACGT\n")
        pro_ref_dir = ref_dir / "pro_ref"
        pro_ref_dir.mkdir(parents=True)
        (pro_ref_dir / "pro_ref.tre").write_text("(ref1:0.1);")
        (pro_ref_dir / "pro_ref.fna").write_text(">ref1\nACGT\n")

        with pytest.raises(FileNotFoundError, match="RAxML info file not found"):
            place_sequences(seqs_path, output_dir, ref_dir)

    def test_absolute_path_resolution(self, tmp_path):
        """Test that paths are resolved to absolute paths."""
        seqs_path = tmp_path / "seqs.fna"
        output_dir = tmp_path / "output"
        ref_dir = tmp_path / "ref"

        seqs_path.write_text(">seq1\nACGT\n")
        pro_ref_dir = ref_dir / "pro_ref"
        pro_ref_dir.mkdir(parents=True)
        (pro_ref_dir / "pro_ref.tre").write_text("(ref1:0.1);")
        (pro_ref_dir / "pro_ref.fna").write_text(">ref1\nACGT\n")
        (pro_ref_dir / "pro_ref.raxml_info").write_text("RAxML info")

        with patch('pgptracker.stage1_processing.phylogeny._run_sepp') as mock_sepp, \
             patch('pgptracker.stage1_processing.phylogeny._run_gappa') as mock_gappa:

            mock_sepp.return_value = output_dir / "placement_placement.json"

            place_sequences(seqs_path, output_dir, ref_dir)

            call_args = mock_sepp.call_args[0]
            assert call_args[0].is_absolute()
            assert call_args[1].is_absolute()
            assert call_args[2].is_absolute()


class TestRunSepp:
    """Tests for _run_sepp function."""

    def test_successful_sepp_execution(self, tmp_path):
        """Test successful SEPP execution."""
        seqs = tmp_path / "seqs.fna"
        ref_tree = tmp_path / "pro_ref.tre"
        ref_aln = tmp_path / "pro_ref.fna"
        ref_info = tmp_path / "pro_ref.raxml_info"
        output_dir = tmp_path / "output"

        seqs.write_text(">seq1\nACGT\n")
        ref_tree.write_text("(ref1:0.1);")
        ref_aln.write_text(">ref1\nACGT\n")
        ref_info.write_text("RAxML info")
        output_dir.mkdir()

        mock_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        expected_json = output_dir / "placement_placement.json"

        def create_output(*args, **kwargs):
            # Create the output file when subprocess.run is called
            expected_json.write_text('{"placements": []}')
            return mock_result

        with patch('subprocess.run', side_effect=create_output) as mock_run:
            result = _run_sepp(seqs, ref_tree, ref_aln, ref_info, output_dir, threads=4)

            assert result == expected_json
            mock_run.assert_called_once()

            cmd = mock_run.call_args[0][0]
            assert "run_sepp.py" in cmd
            assert "-t" in cmd
            assert "-a" in cmd
            assert "-r" in cmd
            assert "-f" in cmd
            assert "-x" in cmd
            assert "4" in cmd

    def test_sepp_command_structure(self, tmp_path):
        """Test that SEPP command has correct flag mapping."""
        seqs = tmp_path / "seqs.fna"
        ref_tree = tmp_path / "pro_ref.tre"
        ref_aln = tmp_path / "pro_ref.fna"
        ref_info = tmp_path / "pro_ref.raxml_info"
        output_dir = tmp_path / "output"

        seqs.write_text(">seq1\nACGT\n")
        ref_tree.write_text("tree")
        ref_aln.write_text("alignment")
        ref_info.write_text("info")
        output_dir.mkdir()

        mock_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        output_json = output_dir / "placement_placement.json"

        def create_output(*args, **kwargs):
            output_json.write_text('{}')
            return mock_result

        with patch('subprocess.run', side_effect=create_output) as mock_run:
            _run_sepp(seqs, ref_tree, ref_aln, ref_info, output_dir, threads=2)

            cmd = mock_run.call_args[0][0]

            t_idx = cmd.index("-t")
            assert str(ref_tree) == cmd[t_idx + 1]

            a_idx = cmd.index("-a")
            assert str(ref_aln) == cmd[a_idx + 1]

            r_idx = cmd.index("-r")
            assert str(ref_info) == cmd[r_idx + 1]

    def test_sepp_failure(self, tmp_path):
        """Test handling of SEPP execution failure."""
        seqs = tmp_path / "seqs.fna"
        ref_tree = tmp_path / "pro_ref.tre"
        ref_aln = tmp_path / "pro_ref.fna"
        ref_info = tmp_path / "pro_ref.raxml_info"
        output_dir = tmp_path / "output"

        seqs.write_text(">seq1\nACGT\n")
        ref_tree.write_text("tree")
        ref_aln.write_text("alignment")
        ref_info.write_text("info")
        output_dir.mkdir()

        mock_result = CompletedProcess(
            args=[],
            returncode=1,
            stdout="SEPP stdout",
            stderr="SEPP failed: alignment error"
        )

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RuntimeError, match="SEPP failed with exit code 1"):
                _run_sepp(seqs, ref_tree, ref_aln, ref_info, output_dir, threads=1)

    def test_missing_output_file(self, tmp_path):
        """Test error when SEPP doesn't create expected output."""
        seqs = tmp_path / "seqs.fna"
        ref_tree = tmp_path / "pro_ref.tre"
        ref_aln = tmp_path / "pro_ref.fna"
        ref_info = tmp_path / "pro_ref.raxml_info"
        output_dir = tmp_path / "output"

        seqs.write_text(">seq1\nACGT\n")
        ref_tree.write_text("tree")
        ref_aln.write_text("alignment")
        ref_info.write_text("info")
        output_dir.mkdir()

        mock_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(FileNotFoundError, match="SEPP finished but did not create"):
                _run_sepp(seqs, ref_tree, ref_aln, ref_info, output_dir, threads=1)


class TestRunGappa:
    """Tests for _run_gappa function."""

    def test_successful_gappa_execution(self, tmp_path):
        """Test successful GAPPA tree conversion."""
        jplace_file = tmp_path / "placement.json"
        output_tree = tmp_path / "placed_seqs.tre"

        jplace_file.write_text('{"placements": []}')

        mock_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        def create_newick(*args, **kwargs):
            # GAPPA creates {file_prefix}{jplace_basename}.newick
            # placed_seqs + placement.json → placed_seqsplacement.newick
            newick_file = tmp_path / "placed_seqsplacement.newick"
            newick_file.write_text("(seq1:0.5);")
            return mock_result

        with patch('subprocess.run', side_effect=create_newick) as mock_run:
            _run_gappa(jplace_file, output_tree)

            # Verify file was renamed to .tre
            assert output_tree.exists()
            assert not (tmp_path / "placed_seqsplacement.newick").exists()

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "gappa" in cmd
            assert "examine" in cmd
            assert "graft" in cmd
            assert "--jplace-path" in cmd
            assert "--out-dir" in cmd
            assert "--file-prefix" in cmd

    def test_gappa_failure(self, tmp_path):
        """Test handling of GAPPA execution failure."""
        jplace_file = tmp_path / "placement.json"
        output_tree = tmp_path / "placed_seqs.tre"

        jplace_file.write_text('{}')

        mock_result = CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="GAPPA failed: invalid jplace format"
        )

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RuntimeError, match="GAPPA failed with exit code 1"):
                _run_gappa(jplace_file, output_tree)

    def test_gappa_command_structure(self, tmp_path):
        """Test that GAPPA command has correct structure."""
        jplace_file = tmp_path / "input" / "placement.json"
        output_tree = tmp_path / "output" / "tree.tre"

        jplace_file.parent.mkdir(parents=True)
        jplace_file.write_text('{}')

        mock_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        def create_newick(*args, **kwargs):
            # Create output directory and newick file
            output_tree.parent.mkdir(parents=True, exist_ok=True)
            # GAPPA creates {file_prefix}{jplace_basename}.newick
            # tree + placement.json → treeplacement.newick
            newick_file = output_tree.parent / f"{output_tree.stem}{jplace_file.stem}.newick"
            newick_file.write_text("(seq1:0.5);")
            return mock_result

        with patch('subprocess.run', side_effect=create_newick) as mock_run:
            _run_gappa(jplace_file, output_tree)

            cmd = mock_run.call_args[0][0]

            jplace_idx = cmd.index("--jplace-path")
            assert str(jplace_file) == cmd[jplace_idx + 1]

            out_dir_idx = cmd.index("--out-dir")
            assert str(output_tree.parent) == cmd[out_dir_idx + 1]

            prefix_idx = cmd.index("--file-prefix")
            assert output_tree.stem == cmd[prefix_idx + 1]
