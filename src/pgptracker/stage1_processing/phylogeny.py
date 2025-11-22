"""
Phylogenetic placement wrapper for PGPTracker.

Replaces PICRUSt2 place_seqs.py with direct SEPP/GAPPA calls.
"""

import subprocess
import tempfile
from pathlib import Path
from pgptracker.utils.profiling_tools.profiler import profile_memory


@profile_memory
def place_sequences(
    seqs_path: Path,
    output_dir: Path,
    ref_dir: Path,
    threads: int = 1
) -> Path:
    """
    Insert unknown ASV sequences into reference phylogeny using SEPP algorithm.

    SEPP places query sequences into existing reference tree without rebuilding it.
    This is faster and more accurate than de novo tree construction for large references.

    Workflow:
    1. SEPP: Align ASVs to reference, find best insertion points in tree
    2. GAPPA: Convert JSON output to standard Newick format

    Why needed: Phylogenetic proximity predicts functional similarity
    - Close relatives in tree likely have similar gene content
    - Tree topology used by Castor/R to infer gene copy numbers (HSP algorithm)

    Args:
        seqs_path: FASTA with ASV sequences (e.g., rep_seqs.fna from QIIME2)
        output_dir: Directory for intermediate files (placement JSON, Newick tree)
        ref_dir: Path to prokaryotic reference database
        threads: CPU cores for SEPP parallelization

    Returns:
        Path to placed_seqs.tre (Newick tree with ASVs grafted onto reference)

    Example:
        Input: 150 ASV sequences
        Reference: ~20,000 prokaryotic genomes
        Output: Newick tree with 20,150 tips (20k ref + 150 ASVs)
        Runtime: ~5-7 minutes on 8 cores
    """
    seqs_path = seqs_path.resolve()
    output_dir = output_dir.resolve()
    ref_dir = ref_dir.resolve()

    ref_tree = ref_dir / "pro_ref" / "pro_ref.tre"
    ref_aln = ref_dir / "pro_ref" / "pro_ref.fna"
    ref_info = ref_dir / "pro_ref" / "pro_ref.raxml_info"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running SEPP phylogenetic placement (threads={threads})")
    jplace_file = _run_sepp(seqs_path, ref_tree, ref_aln, ref_info, output_dir, threads)

    print("Converting placement to Newick tree using GAPPA")
    output_tree = output_dir / "placed_seqs.tre"
    _run_gappa(jplace_file, output_tree)

    print(f"Phylogenetic placement completed: {output_tree}")
    return output_tree


@profile_memory
def _run_sepp(
    seqs: Path,
    ref_tree: Path,
    ref_aln: Path,
    ref_info: Path,
    output_dir: Path,
    threads: int
) -> Path:
    """
    Execute SEPP algorithm to place query sequences into reference tree.

    SEPP (SATe-enabled Phylogenetic Placement) uses divide-and-conquer strategy:
    1. Fragments query sequences into overlapping chunks
    2. Each chunk aligned to reference subset
    3. Maximum likelihood determines best insertion point
    4. Ensemble method combines results

    Args:
        seqs: Query sequences to place (FASTA format)
        ref_tree: Reference phylogeny (Newick format, ~20k tips)
        ref_aln: Multiple sequence alignment of reference genomes
        ref_info: RAxML parameters file (substitution model, branch lengths)
        output_dir: Where to write placement JSON
        threads: Parallel alignment jobs

    Returns:
        Path to placement_placement.json (jplace format)

    Note: SEPP refuses to overwrite existing files. We clean old outputs first.
    """
    with tempfile.TemporaryDirectory(prefix="sepp_") as temp_dir:
        output_prefix = "placement"

        # Clean old SEPP output files to avoid overwrite errors
        for old_file in output_dir.glob(f"{output_prefix}_*"):
            old_file.unlink()

        cmd = [
            "run_sepp.py",
            "-t", str(ref_tree),
            "-a", str(ref_aln),
            "-r", str(ref_info),
            "-f", str(seqs),
            "-o", output_prefix,
            "-d", str(output_dir),
            "-x", str(threads),
            "--tempdir", temp_dir
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"SEPP failed with exit code {result.returncode}\n"
                f"STDERR: {result.stderr}\n"
                f"STDOUT: {result.stdout}"
            )

        expected_json = output_dir / f"{output_prefix}_placement.json"

        if not expected_json.exists():
            raise FileNotFoundError(
                f"SEPP finished but did not create {expected_json}\n"
                f"Files in output directory: {list(output_dir.glob('*'))}"
            )

    return expected_json


def _run_gappa(jplace_file: Path, output_tree: Path) -> None:
    """
    Convert jplace format (SEPP output) to Newick tree using GAPPA.

    GAPPA (Genesis Applications for Phylogenetic Placement Analysis) extracts
    the phylogeny from jplace JSON and grafts query sequences at their ML positions.

    Args:
        jplace_file: Placement result from SEPP (placement_placement.json)
        output_tree: Desired output path (e.g., placed_seqs.tre)

    Output file naming quirk:
        GAPPA concatenates: file_prefix + jplace_basename + ".newick"
        Example: "placed_seqs" + "placement_placement" + ".newick"
                 → placed_seqsplacement_placement.newick
        We rename this to output_tree path (.tre extension)

    Why Newick needed: R/Castor HSP algorithm requires standard tree format
    """
    cmd = [
        "gappa", "examine", "graft",
        "--jplace-path", str(jplace_file),
        "--out-dir", str(output_tree.parent),
        "--file-prefix", output_tree.stem,
        "--allow-file-overwriting"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"GAPPA failed with exit code {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )

    # GAPPA creates {file_prefix}{jplace_basename}.newick
    # Example: placed_seqs + placement_placement.json → placed_seqsplacement_placement.newick
    gappa_output = output_tree.parent / f"{output_tree.stem}{jplace_file.stem}.newick"
    if not gappa_output.exists():
        raise FileNotFoundError(
            f"GAPPA finished but did not create {gappa_output}\n"
            f"Files in output directory: {list(output_tree.parent.glob('*'))}"
        )

    # Rename to .tre extension as expected by prediction.py
    gappa_output.rename(output_tree)
