"""
Phylogenetic placement wrapper for PGPTracker.

Replaces PICRUSt2 place_seqs.py with direct SEPP/GAPPA calls.
"""

import subprocess
import tempfile
from pathlib import Path


def place_sequences(
    seqs_path: Path,
    output_dir: Path,
    ref_dir: Path,
    threads: int = 1
) -> Path:
    """
    Place sequences into reference phylogeny using SEPP.

    Args:
        seqs_path: Path to representative sequences (.fna/.fasta)
        output_dir: Directory for output files
        ref_dir: Path to prokaryotic reference database
        threads: Number of parallel processes

    Returns:
        Path to final Newick tree file
    """
    ref_tree = ref_dir / "pro_ref.tre"
    ref_aln = ref_dir / "pro_ref.fna"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running SEPP phylogenetic placement (threads={threads})")
    jplace_file = _run_sepp(seqs_path, ref_tree, ref_aln, output_dir, threads)

    print("Converting placement to Newick tree using GAPPA")
    output_tree = output_dir / "placed_seqs.tre"
    _run_gappa(jplace_file, output_tree)

    print(f"Phylogenetic placement completed: {output_tree}")
    return output_tree


def _run_sepp(
    seqs: Path,
    ref_tree: Path,
    ref_aln: Path,
    output_dir: Path,
    threads: int
) -> Path:
    """Execute SEPP via run_sepp.py subprocess."""
    with tempfile.TemporaryDirectory(prefix="sepp_") as temp_dir:
        jplace_file = output_dir / "placement.jplace"

        cmd = [
            "run_sepp.py",
            "-t", str(ref_tree),
            "-r", str(ref_aln),
            "-f", str(seqs),
            "-o", str(jplace_file.stem),
            "-d", temp_dir,
            "-x", str(threads)
        ]

        result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"SEPP failed with exit code {result.returncode}\n"
                f"STDERR: {result.stderr}"
            )

    return jplace_file


def _run_gappa(jplace_file: Path, output_tree: Path) -> None:
    """Convert .jplace to Newick tree using GAPPA."""
    cmd = [
        "gappa", "examine", "graft",
        "--jplace-path", str(jplace_file),
        "--out-dir", str(output_tree.parent),
        "--file-prefix", output_tree.stem
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"GAPPA failed with exit code {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )
