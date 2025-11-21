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

    Raises:
        FileNotFoundError: If inputs don't exist
        RuntimeError: If SEPP or GAPPA fails
    """
    # Validate inputs
    if not seqs_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {seqs_path}")
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")

    # Reference files
    ref_tree = ref_dir / "pro_ref.tre"
    ref_aln = ref_dir / "pro_ref.fna"
    ref_info = ref_dir / "pro_ref_info.tsv"

    for ref_file in [ref_tree, ref_aln]:
        if not ref_file.exists():
            raise FileNotFoundError(f"Reference file missing: {ref_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run SEPP to generate placement file
    print(f"Running SEPP phylogenetic placement (threads={threads})")
    jplace_file = _run_sepp(
        seqs=seqs_path,
        ref_tree=ref_tree,
        ref_aln=ref_aln,
        output_dir=output_dir,
        threads=threads
    )

    # Step 2: Convert .json placement to .tre using GAPPA
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
    """
    Execute SEPP via run_sepp.py subprocess.

    Args:
        seqs: Query sequences
        ref_tree: Reference tree
        ref_aln: Reference alignment
        output_dir: Output directory
        threads: Thread count

    Returns:
        Path to .jplace file

    Raises:
        RuntimeError: If SEPP fails
    """
    # Use managed temp directory to prevent /tmp clutter
    with tempfile.TemporaryDirectory(prefix="sepp_") as temp_dir:
        jplace_file = output_dir / "placement.jplace"

        cmd = [
            "run_sepp.py",
            "-t", str(ref_tree),
            "-r", str(ref_aln),
            "-f", str(seqs),
            "-o", str(jplace_file.stem),
            "-d", temp_dir,  # Explicit temp directory
            "-x", str(threads)
        ]

        result = subprocess.run(
            cmd,
            cwd=output_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"SEPP failed with exit code {result.returncode}\n"
                f"STDERR: {result.stderr}"
            )

    if not jplace_file.exists():
        raise RuntimeError(f"SEPP did not generate expected output: {jplace_file}")

    return jplace_file


def _run_gappa(jplace_file: Path, output_tree: Path) -> None:
    """
    Convert .jplace to Newick tree using GAPPA.

    Args:
        jplace_file: SEPP placement file (.jplace)
        output_tree: Output Newick tree path

    Raises:
        RuntimeError: If GAPPA fails
    """
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

    if not output_tree.exists():
        raise RuntimeError(f"GAPPA did not generate expected output: {output_tree}")
