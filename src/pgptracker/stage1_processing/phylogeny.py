"""
Phylogenetic placement wrapper for PGPTracker.

Replaces PICRUSt2 place_seqs.py with direct SEPP/GAPPA calls.
"""

import subprocess
import tempfile
from pathlib import Path
from pgptracker.utils.profiling_tools.profiler import profile_memory
from pgptracker.utils.env_manager import detect_free_memory


def _calculate_safe_threads(requested_threads: int) -> int:
    """
    Calculate safe thread count based on available RAM to prevent thrashing.

    SEPP (and the underlying pplacer/hmmalign tools) can be memory-intensive depending
    on the size of the reference tree (~20k tips) and the number of query sequences.
    
    If we blindly launch 8 or 16 threads on a machine with limited RAM, each thread
    allocates its own memory for the HMM models and alignment chunks. This can lead
    to "Out of Memory" errors or disk swapping (thrashing), making the process
    extremely slow or crashing the system.

    Heuristic:
    - Base memory for Reference Tree + Alignment: ~5 GB (fixed cost)
    - Per-thread overhead estimation: ~1 GB per active worker
    """
    available_ram_gb = detect_free_memory()
    
    # Reserve 5GB for the base process and system stability
    usable_ram_for_threads = max(0, available_ram_gb - 5.0)
    
    # Estimate max threads that fit in the remaining RAM (1GB per thread conservative estimate)
    max_threads_by_ram = int(usable_ram_for_threads / 1.0)
    
    # Ensure we always run at least 1 thread
    max_threads_by_ram = max(1, max_threads_by_ram)

    # Use the lower limit: either what the user asked for, or what the RAM permits
    safe_threads = min(requested_threads, max_threads_by_ram)

    if safe_threads < requested_threads:
        print(f"  [Adaptive Resource] Limiting threads from {requested_threads} to {safe_threads} "
              f"to avoid RAM overflow (Available: {available_ram_gb:.1f}GB).")
    
    return safe_threads


@profile_memory
def place_sequences(
    seqs_path: Path,
    output_dir: Path,
    ref_dir: Path,
    threads: int = 1
) -> Path:
    """
    Insert unknown ASV sequences into reference phylogeny using SEPP algorithm.

    This is the first critical step of functional prediction. Before we can guess what
    genes a bacteria has, we must identify exactly where it fits in the tree of life.

    Why SEPP and not standard tree building?
    - De novo tree building with 20,000+ genomes is computationally impossible for routine use.
    - SEPP (SATe-enabled Phylogenetic Placement) inserts new sequences into a *fixed* reference tree without rebuilding the whole structure.
    - It is highly accurate because it uses an ensemble of HMMs (Hidden Markov Models) 
      to align sequences only to the most relevant parts of the tree.

    Workflow:
    1. Input Validation: Check for Reference Tree, Alignment (.fna), and RAxML Info.
    2. SEPP Execution: Aligns ASVs and calculates Maximum Likelihood insertion branches.
    3. GAPPA Execution: Converts the complex JSON placement output into a standard 
       Newick tree file that can be read by downstream tools (like R/Castor).

    Args:
        seqs_path: FASTA file with ASV sequences (e.g., rep_seqs.fna).
        output_dir: Directory to store intermediate files and the final tree.
        ref_dir: Path to the bundled prokaryotic reference database.
        threads: Number of CPU cores to use for parallel alignment.

    Returns:
        Path to the final 'placed_seqs.tre' (Newick format).
    """
    # Ensure absolute paths to prevent "File not found" errors when subprocess changes cwd
    seqs_path = seqs_path.resolve()
    output_dir = output_dir.resolve()
    ref_dir = ref_dir.resolve()

    # Adaptive resource management: Don't crash the user's PC
    safe_threads = _calculate_safe_threads(threads)

    # Locate reference files (handle loose files or 'pro_ref' subfolder structure)
    possible_tree = ref_dir / "pro_ref.tre"
    if not possible_tree.exists():
        ref_dir = ref_dir / "pro_ref"
    
    ref_tree = ref_dir / "pro_ref.tre"
    ref_aln = ref_dir / "pro_ref.fna"
    ref_info = ref_dir / "pro_ref.raxml_info"

    # Validate reference files existence
    if not ref_aln.exists():
        raise FileNotFoundError(f"Reference alignment not found at: {ref_aln}")
    if not ref_info.exists():
        raise FileNotFoundError(f"RAxML info file not found at: {ref_info}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running SEPP phylogenetic placement (threads={safe_threads})")
    jplace_file = _run_sepp(seqs_path, ref_tree, ref_aln, ref_info, output_dir, safe_threads)

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
    Execute SEPP algorithm via the `run_sepp.py` command line tool.
    
    This helper manages the subprocess call, ensuring correct flags are mapped:
    - -t: Reference Tree
    - -a: Reference Alignment (FASTA)
    - -r: RAxML Info File (Model parameters)
    - -f: Fragment File (User's ASVs)
    
    Uses a temporary directory for SEPP's internal files to keep the output folder clean.
    CRITICAL: Cleans up old output files before running, as SEPP fails if files exist.
    """
    with tempfile.TemporaryDirectory(prefix="sepp_") as temp_dir:
        output_prefix = "placement"
        
        # FIX: Clean old SEPP output files to avoid overwrite errors
        # SEPP creates files like 'placement_placement.json', 'placement_placement.relabelled.tree', etc.
        for old_file in output_dir.glob(f"{output_prefix}_*"):
            try:
                old_file.unlink()
            except OSError:
                pass

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

        # Capture stdout/stderr for debugging if it fails
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"SEPP failed with exit code {result.returncode}\n"
                f"STDERR: {result.stderr}\n"
                f"STDOUT: {result.stdout}"
            )
        
        # SEPP output naming is quirky: prefix + "_" + prefix + ".json" usually
        expected_json = output_dir / f"{output_prefix}_placement.json"
        
        if not expected_json.exists():
             raise FileNotFoundError(f"SEPP finished but did not create {expected_json}")

        return expected_json


def _run_gappa(jplace_file: Path, output_tree: Path) -> None:
    """
    Convert SEPP's JSON placement format (.jplace) to Newick tree (.tre) using GAPPA.
    
    Why:
    SEPP outputs a JSON file containing placement probabilities for every edge.
    Downstream tools (like the Castor R package used in prediction) require a standard
    Newick tree format where the sequences are grafted as tips onto the best edge.
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

    # GAPPA creates: {file_prefix}{jplace_basename}.newick
    # Example: placed_seqs + placement_placement.json → placed_seqsplacement_placement.newick
    gappa_output = output_tree.parent / f"{output_tree.stem}{jplace_file.stem}.newick"

    if not gappa_output.exists():
        raise FileNotFoundError(
            f"GAPPA finished but did not create {gappa_output}\n"
            f"Files in output directory: {list(output_tree.parent.glob('*'))}"
        )

    # Rename .newick to .tre for downstream compatibility
    gappa_output.rename(output_tree)