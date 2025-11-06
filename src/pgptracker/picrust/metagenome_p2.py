"""
Metagenome pipeline runner for PGPTracker.

This module wraps PICRUSt2 metagenome_pipeline.py (Douglas et al., 2020)
to normalize sequence abundances and generate unstratified metagenome predictions.

File originally named normalize.py
"""

from pathlib import Path
from typing import Dict
from pgptracker.utils.env_manager import run_command
import subprocess # Keep subprocess for CalledProcessError
from pgptracker.utils.validator import validate_output_file as _validate_output


def run_metagenome_pipeline(
    table_path: Path,
    marker_path: Path,
    ko_predicted_path: Path,
    output_dir: Path,
    max_nsti: float = 1.7
) -> Dict[str, Path]:
    """
    Normalizes abundances and generates unstratified metagenome predictions.
    
    Wraps PICRUSt2 metagenome_pipeline.py to:
    1. Normalize ASV abundances by 16S copy number
    2. Generate unstratified KO abundances per sample
    
    This function ALWAYS runs unstratified mode.
    
    Args:
        table_path: Path to feature table (.biom)
        marker_path: Path to marker predictions (marker_nsti_predicted.tsv.gz)
        ko_predicted_path: Path to KO predictions (KO_predicted.tsv.gz)
        output_dir: Base directory for PICRUSt2 outputs
        max_nsti: Maximum NSTI threshold for filtering (default: 1.7)
        
    Returns:
        Dictionary with paths to output files:
            - 'seqtab_norm': Normalized feature table (seqtab_norm.tsv.gz)
            - 'pred_metagenome_unstrat': Unstratified KO abundances (pred_metagenome_unstrat.tsv.gz)
            
    Raises:
        FileNotFoundError: If any input file doesn't exist
        subprocess.CalledProcessError: If PICRUSt2 fails
        RuntimeError: If output validation fails or file is empty
    """
    # 1. Validate inputs
    for path, name in [
        (table_path, "Feature table"),
        (marker_path, "Marker predictions"),
        (ko_predicted_path, "KO predictions")
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        
        if path.stat().st_size == 0:
            raise RuntimeError(f"{name} file is empty: {path}")
    
    # 2. Ensure parent output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Define output subdirectory (PICRUSt2 creates this)
    metagenome_out_dir = output_dir / "KO_metagenome_out"
    
    # 4. Build command (without stratified flags)
    cmd = [
        "metagenome_pipeline.py",
        "-i", str(table_path),
        "-m", str(marker_path),
        "-f", str(ko_predicted_path),
        "-o", str(metagenome_out_dir),
        "--max_nsti", str(max_nsti)
    ]
    
    # 5. Print execution details
    print("\nRunning PICRUSt2 metagenome_pipeline.py (Douglas et al., 2020)")
    print(f"  Feature table: {table_path}")
    print(f"  Marker predictions: {marker_path}")
    print(f"  KO predictions: {ko_predicted_path}")
    print(f"  Output directory: {metagenome_out_dir}")
    print(f"  Max NSTI: {max_nsti}")
    print(f"  Mode: Unstratified only")
    
    # 6. Run command
    # Using "Picrust2" (capitalized) as requested
    run_command("Picrust2", cmd, check=True)
    
    # 7. Define expected output paths
    seqtab_norm = metagenome_out_dir / "seqtab_norm.tsv.gz"
    pred_unstrat = metagenome_out_dir / "pred_metagenome_unstrat.tsv.gz"
    
    # 8. Validate outputs (inline validation)
    _validate_output(seqtab_norm, "metagenome_pipeline.py", "normalized sequence table")
    _validate_output(pred_unstrat, "metagenome_pipeline.py", "unstratified metagenome predictions")
    
    print(f"Metagenome pipeline completed:")
    print(f"  Normalized table: {seqtab_norm}")
    print(f"  Unstratified predictions: {pred_unstrat}")
    
    return {
        'seqtab_norm': seqtab_norm,
        'pred_metagenome_unstrat': pred_unstrat
    }