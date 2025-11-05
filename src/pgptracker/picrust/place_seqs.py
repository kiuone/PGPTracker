"""
Phylogenetic tree builder for PGPTracker.

This module wraps PICRUSt2 place_seqs.py (Douglas et al., 2020)
to place study sequences into reference phylogeny.

File originally named phylo.py
"""

from pathlib import Path
from pgptracker.utils.env_manager import run_command
import subprocess # Keep subprocess for CalledProcessError

def _validate_output(expected_path: Path, tool_name: str) -> None:
    """
    Validates that expected output file exists and is not empty.
    (Local helper function to avoid extra utils file)
    
    Args:
        expected_path: Path to expected output file
        tool_name: Name of tool for error messages
        
    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If file is empty
    """
    if not expected_path.exists():
        raise FileNotFoundError(
            f"{tool_name} did not create expected output: {expected_path}"
        )
    
    if expected_path.stat().st_size == 0:
        raise RuntimeError(
            f"{tool_name} created empty output file: {expected_path}"
        )

def build_phylogenetic_tree(
    sequences_path: Path,
    output_dir: Path,
    threads: int = 1
) -> Path:
    """
    Builds phylogenetic tree by placing sequences into reference tree.
    
    Wraps PICRUSt2 place_seqs.py using SEPP algorithm.
    
    Args:
        sequences_path: Path to representative sequences (.fna/.fasta)
        output_dir: Directory for output files
        threads: Number of parallel processes (default: 1)
        
    Returns:
        Path to output tree file (placed_seqs.tre)
        
    Raises:
        FileNotFoundError: If sequences file doesn't exist
        subprocess.CalledProcessError: If PICRUSt2 fails
        RuntimeError: If output validation fails or file is empty
    """
    # 1. Validate input
    if not sequences_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
    
    if sequences_path.stat().st_size == 0:
        raise RuntimeError(f"Sequences file is empty: {sequences_path}")
    
    # 2. Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Define output path
    output_tree = output_dir / "placed_seqs.tre"
    
    # 4. Build command
    cmd = [
        "place_seqs.py",
        "-s", str(sequences_path),
        "-o", str(output_tree),
        "-p", str(threads),
        "-t", "sepp",
    ]
    
    # 5. Run command with attribution
    print("Running PICRUSt2 place_seqs.py (Douglas et al., 2020)")
    print(f"  Input: {sequences_path}")
    print(f"  Output: {output_tree}")
    print(f"  Threads: {threads}")
    print(f"  Algorithm: SEPP")
    
    # Using "Picrust2" (capitalized) as requested
    run_command("Picrust2", cmd, check=True)
    
    # 6. Validate output
    _validate_output(output_tree, "place_seqs.py")
    
    print(f"Phylogenetic tree completed: {output_tree}")
    
    return output_tree