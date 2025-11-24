"""
Gene content prediction for PGPTracker.

This module wraps PICRUSt2 hsp.py (Douglas et al., 2020)
to predict marker gene copy numbers and KO abundances.

File originally named predict.py
"""

from pathlib import Path
from typing import Dict
from pgptracker.utils.env_manager import run_command
import subprocess # Keep subprocess for CalledProcessError
from pgptracker.utils.validator import validate_output_file as _validate_output

def _run_hsp_prediction(
    prediction_type: str,
    tree_path: Path,
    output_path: Path,
    threads: int,
    chunk_size: int,
    calculate_nsti: bool
) -> Path:
    """
    Runs single hsp.py prediction.
    (Local helper function)
    """
    # 1. Build command
    cmd = [
        "hsp.py",
        "-i", prediction_type,
        "-r", "bacteria",  # Using bacteria reference as per pipeline.txt
        "-m", "pic",       # Using pic method as per pipeline.txt
        "-t", str(tree_path),
        "-o", str(output_path),
        "-p", str(threads),
        "--chunk_size", str(chunk_size)
    ]
    
    # Add NSTI flag if needed
    if calculate_nsti:
        cmd.append("-n")
    
    # 2. Print execution details
    print(f"  Type: {prediction_type}")
    print(f"  Chunk size: {chunk_size}")
    
    # 3. Run command
    run_command("Picrust2", cmd, check=True)
    
    # 4. Validate output (inline validation)
    _validate_output(output_path, "hsp.py", f"{prediction_type} prediction")
    
    print(f"  Prediction completed: {output_path}")
    
    return output_path

def predict_gene_content(
    tree_path: Path,
    output_dir: Path,
    threads: int = 1,
    chunk_size: int = 1000
) -> Dict[str, Path]:
    """
    Predicts gene content using hidden state prediction.
    
    Runs PICRUSt2 hsp.py twice:
    1. Marker genes (16S) with NSTI calculation
    2. KO (KEGG Ortholog) predictions
    
    Args:
        tree_path: Path to phylogenetic tree (.tre)
        output_dir: Directory for output files
        threads: Number of parallel processes (default: 1)
        chunk_size: Number of gene families per chunk (default: 1000)
        
    Returns:
        Dictionary with paths to prediction files:
            - 'marker': marker_nsti_predicted.tsv.gz
            - 'ko': KO_predicted.tsv.gz
            
    Raises:
        FileNotFoundError: If tree file doesn't exist
        subprocess.CalledProcessError: If PICRUSt2 fails
        RuntimeError: If output validation fails or file is empty
    """
    # 1. Validate input
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    
    if tree_path.stat().st_size == 0:
        raise RuntimeError(f"Tree file is empty: {tree_path}")
    
    # 2. Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Run marker prediction (16S with NSTI)
    print("\nRunning PICRUSt2 hsp.py for marker genes (Douglas et al., 2020)")
    marker_path = _run_hsp_prediction(
        prediction_type="16S",
        tree_path=tree_path,
        output_path=output_dir / "marker_nsti_predicted.tsv.gz",
        threads=threads,
        chunk_size=chunk_size,
        calculate_nsti=True
    )
    
    # 4. Run KO prediction
    print("\nRunning PICRUSt2 hsp.py for KO predictions (Douglas et al., 2020)")
    ko_path = _run_hsp_prediction(
        prediction_type="KO",
        tree_path=tree_path,
        output_path=output_dir / "KO_predicted.tsv.gz",
        threads=threads,
        chunk_size=chunk_size,
        calculate_nsti=False
    )
    
    return {
        'marker': marker_path,
        'ko': ko_path
    }