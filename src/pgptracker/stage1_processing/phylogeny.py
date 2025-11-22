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

    Workflow:
    1. SEPP aligns ASVs to reference alignment and places them in reference tree
    2. GAPPA converts placement JSON to standard Newick tree format

    Why phylogenetic placement needed:
    - Unknown ASVs (user's sequences) need evolutionary context
    - Tree topology determines which reference genomes are "nearby"
    - Proximity in tree = similar gene content (basis for HSP predictions)

    SEPP algorithm:
    - Ensemble-based maximum likelihood placement
    - Fragments query sequence and finds best insertion point in tree
    - More accurate than de novo tree building for large reference trees

    Input sequences from: pipeline_st1.py → export_qza_files() → rep_seqs.fna
    Reference tree: bundled database (src/pgptracker/databases/prokaryotic/pro_ref/pro_ref.tre) (~20k genomes)

    Returns:
        Path to placed_seqs.tre (Newick format with ASVs inserted)
    """
    seqs_path = seqs_path.resolve()
    output_dir = output_dir.resolve()
    ref_dir = ref_dir.resolve()

    ref_tree = ref_dir / "pro_ref" / "pro_ref.tre"
    ref_aln = ref_dir / "pro_ref" / "pro_ref.fna"
    ref_info = ref_dir / "pro_ref" / "pro_ref.raxml_info"

    if not ref_aln.exists():
        raise FileNotFoundError(f"Reference alignment not found at: {ref_aln}")
    if not ref_tree.exists():
        raise FileNotFoundError(f"Reference tree not found at: {ref_tree}")
    if not ref_info.exists():
        raise FileNotFoundError(f"RAxML info file not found at: {ref_info}")

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
    Execute SEPP algorithm via run_sepp.py (must be in PATH from conda env).

    SEPP flags:
    -t: tree file (pro_ref.tre)
    -a: alignment file (pro_ref.fna)
    -r: raxml info file (pro_ref.raxml_info)
    -f: fragment/query sequences
    -o: output prefix
    -d: output directory for placement files
    -x: threads
    --tempdir: temporary working directory for SEPP internals

    Output: placement_placement.json (SEPP appends _placement to the prefix)
    """
    with tempfile.TemporaryDirectory(prefix="sepp_") as temp_dir:
        output_prefix = "placement"

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
    Convert SEPP's JSON placement format to Newick tree using GAPPA.

    GAPPA grafts placed sequences onto reference tree at their insertion points.
    Output used by R/Castor for Hidden State Prediction (needs Newick format).
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
