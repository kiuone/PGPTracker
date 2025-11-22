"""
Functional prediction wrapper for PGPTracker.

This module orchestrates the prediction of functional gene content (KOs) based on the
phylogenetic tree. It acts as a bridge between Python (data handling) and R (statistical algorithms).

Optimization Strategy:
1. High-RAM Systems (>10GB free):
   - Uses "Single-Pass Mode".
   - Loads the entire database into memory using Polars (fast).
   - Writes a single temporary input file.
   - Launches ONE R process to handle all KOs.
   - Why? Loading the phylogenetic tree in R is computationally expensive. 
     Doing it once is faster than splitting the job and reloading the tree 8+ times.

2. Low-RAM Systems (<10GB free):
   - Uses "Batched Mode".
   - Reads columns from the disk in small chunks.
   - Processes them serially to ensure memory never exceeds limits.
   - Slower due to I/O and overhead, but prevents system crashes.
"""

import gc
import subprocess
import tempfile
import gzip
import os
import math
from pathlib import Path
from typing import Dict, Optional, List

import polars as pl
from pgptracker.utils.env_manager import detect_free_memory
from pgptracker.utils.profiling_tools.profiler import profile_memory


def _get_r_script(script_name: str) -> Path:
    """
    Resolve absolute path to bundled R script using importlib.resources.

    Args:
        script_name: Filename (e.g., "hsp.R").

    Returns:
        Absolute path to the script.
    """
    from importlib import resources
    try:
        r_scripts = resources.files("pgptracker.resources.r_scripts")
        return Path(str(r_scripts / script_name))
    except AttributeError:
        with resources.path("pgptracker.resources.r_scripts", script_name) as p:
            return Path(p)


def _calculate_processing_strategy(
    ko_db_path: Path, 
    available_ram_gb: Optional[float] = None
) -> dict:
    """
    Determine processing strategy based on File Size vs Available RAM.

    Logic:
    - Estimate the uncompressed size of the KO table.
    - If the system can hold the Full Table + R Overhead (approx 2x table size),
      we choose 'single_pass'. This is the fastest method.
    - Otherwise, we calculate a safe chunk size for 'serial_batched' processing.

    Args:
        ko_db_path: Path to the compressed KO database.
        available_ram_gb: Optional override for free memory.

    Returns:
        Dict containing 'strategy', 'chunk_size', and a descriptive 'message'.
    """
    if available_ram_gb is None:
        available_ram_gb = detect_free_memory()

    # Scan header to count columns (fast operation, no data loading)
    # infer_schema_length=0 is critical to avoid scanning the whole file for types
    try:
        ko_lazy = pl.scan_csv(ko_db_path, separator="\t", infer_schema_length=0)
        num_columns = len(ko_lazy.collect_schema().names()) - 1 # Subtract ID column
    except Exception:
        # Fallback if schema collection fails
        num_columns = 10543 # Default size of PICRUSt2 KO table

    # Estimate Sizes
    # Physical size on disk (compressed)
    compressed_size_gb = ko_db_path.stat().st_size / (1024**3)
    
    # Estimated uncompressed size in RAM (Polars is efficient ~1x)
    # Gzip text compression is usually 4-6x. Using 6x as conservative estimate.
    estimated_db_ram_gb = compressed_size_gb * 6.0
    
    # Safety buffer for R process overhead (loading tree + matrix duplication)
    required_ram_single_pass = estimated_db_ram_gb * 2.5
    
    # --- Decision Logic ---
    
    # Strategy A: High Performance (Single Pass)
    if available_ram_gb >= required_ram_single_pass:
        return {
            'strategy': 'single_pass',
            'chunk_size': num_columns,
            'message': (f"🚀 High RAM Mode: {available_ram_gb:.1f}GB available. "
                        f"Processing all {num_columns} KOs in one pass (Fastest).")
        }

    # Strategy B: Low Memory (Serial Batches)
    else:
        # Target usage: 40% of available RAM per batch to be safe
        safe_ram_gb = available_ram_gb * 0.4
        fraction = safe_ram_gb / estimated_db_ram_gb
        chunk_size = int(num_columns * fraction)
        chunk_size = max(500, chunk_size) # Minimum floor
        num_batches = math.ceil(num_columns / chunk_size)
        
        return {
            'strategy': 'serial_batched',
            'chunk_size': chunk_size,
            'message': (f"⚠️ Low RAM Mode: {available_ram_gb:.1f}GB available. "
                        f"Processing in {num_batches} chunks to prevent crash.")
        }


@profile_memory
def predict_functional_profiles(
    tree_path: Path,
    output_dir: Path,
    ref_dir: Path,
    threads: int = 1,
    chunk_size: Optional[int] = 0
) -> Dict[str, Path]:
    """
    Main entry point for functional prediction (ASV -> KO).

    This function coordinates the prediction of two types of data:
    1. Marker Genes (16S): Used for normalization. Calculates copy numbers and NSTI.
    2. Functional Genes (KOs): The actual functional profile.

    Args:
        tree_path: Path to the phylogenetic tree (Newick format).
        output_dir: Directory to save prediction results.
        ref_dir: Path containing reference databases (16S.txt.gz, ko.txt.gz).
        threads: Number of threads (Not used in Single-Pass mode to avoid R overhead).
        chunk_size: (Optional) Manual override for batch size.

    Returns:
        Dict containing paths to 'marker' and 'ko' prediction files.
    """
    ref_dir = ref_dir.resolve()
    output_dir = output_dir.resolve()
    tree_path = tree_path.resolve()

    marker_db = ref_dir / "16S.txt.gz"
    ko_db = ref_dir / "ko.txt.gz"

    if not marker_db.exists():
        raise FileNotFoundError(f"Marker DB not found: {marker_db}")
    if not ko_db.exists():
        raise FileNotFoundError(f"KO DB not found: {ko_db}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Predicting marker genes (16S) and NSTI...")
    marker_path = _predict_marker_16s(tree_path, marker_db, output_dir)

    print("Predicting KO abundances...")
    ko_path = _predict_ko_adaptive(
        tree_path, ko_db, output_dir / "KO_predicted.tsv.gz", chunk_size
    )

    return {'marker': marker_path, 'ko': ko_path}


@profile_memory
def _predict_marker_16s(tree: Path, db_path: Path, output_dir: Path) -> Path:
    """
    Predict 16S rRNA copy numbers and calculate phylogenetic quality metric (NSTI).

    Workflow:
    1. Load 16S reference table (genome_id → copy_count).
    2. Call `hsp.R`: Uses Castor (Max Parsimony) to predict copy counts for unknown tips.
    3. Call `nsti.R`: Calculates phylogenetic distance to the nearest reference genome.
    4. Join predictions with NSTI values into a single table.
    """
    hsp_script = _get_r_script("hsp.R")
    nsti_script = _get_r_script("nsti.R")

    # Force string reading to prevent compute errors on mixed-type ID columns
    marker_df = pl.read_csv(db_path, separator="\t", infer_schema_length=0)
    genome_col = marker_df.columns[0]

    with tempfile.TemporaryDirectory(prefix="marker_") as temp_dir:
        temp_dir = Path(temp_dir)

        # Write input for R (HSP)
        trait_file = temp_dir / "16S_trait.tsv"
        marker_df.write_csv(trait_file, separator="\t")

        # Run HSP
        hsp_output = temp_dir / "16S_predicted.tsv"
        _run_r_script(hsp_script, [str(tree), str(trait_file), str(hsp_output), "mp"])

        # Write input for R (NSTI)
        known_tips_file = temp_dir / "known_tips.txt"
        marker_df.select(genome_col).write_csv(known_tips_file, include_header=False)

        # Run NSTI
        nsti_output = temp_dir / "nsti.tsv"
        _run_r_script(nsti_script, [str(tree), str(known_tips_file), str(nsti_output)])

        # Join results
        pred_16s = pl.read_csv(hsp_output, separator="\t")
        nsti_df = pl.read_csv(nsti_output, separator="\t")
        result = pred_16s.join(nsti_df, on="sequence", how="left")

    output_path = output_dir / "marker_nsti_predicted.tsv.gz"
    with gzip.open(output_path, 'wb') as f:
        result.write_csv(f, separator="\t") # type: ignore

    return output_path


def _ensure_ko_prefix(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensures all KO columns start with 'ko:'.
    
    The raw PICRUSt2 database usually has columns like 'K00001'.
    Downstream steps in PGPTracker expect 'ko:K00001'.
    """
    new_cols = []
    for col in df.columns:
        if col == "sequence" or col.startswith("metadata_") or col == "closest_reference_genome":
            new_cols.append(col)
        elif col.startswith("ko:"):
            new_cols.append(col)
        else:
            new_cols.append(f"ko:{col}")
    
    df.columns = new_cols
    return df


@profile_memory
def _predict_ko_adaptive(
    tree: Path,
    db_path: Path,
    output_path: Path,
    chunk_size: Optional[int]
) -> Path:
    """
    Predict KO gene copy numbers using an adaptive strategy.

    Logic:
    - Analyzes available RAM.
    - If High RAM: Loads full DB, runs single R process (Fastest).
    - If Low RAM: Loads chunks, runs multiple R processes sequentially (Safe).
    """
    hsp_script = _get_r_script("hsp.R")

    # 1. Determine Strategy
    config = _calculate_processing_strategy(db_path)
    
    # Manual Override
    if chunk_size and chunk_size > 0:
        final_chunk_size = chunk_size
        strategy = 'manual_batched'
        print(f"  [Mode] Manual Override: Batch size {final_chunk_size}.")
    elif chunk_size == -1:
        strategy = 'single_pass'
        final_chunk_size = 9999999
        print("  [Mode] Manual Override: Single-pass forced.")
    else:
        final_chunk_size = int(config['chunk_size'])
        strategy = config['strategy']
        print(f"  {config['message']}")

    # 2. Execution
    
    # --- STRATEGY A: SINGLE PASS (Fastest) ---
    if strategy == 'single_pass':
        print("  Loading full database into RAM...")
        # Force string reading
        full_ko_df = pl.read_csv(db_path, separator="\t", infer_schema_length=0)
        
        with tempfile.TemporaryDirectory(prefix="ko_pass_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            trait_file = temp_dir / "full_ko_trait.tsv"
            temp_output = temp_dir / "full_ko_predicted.tsv"
            
            # Write uncompressed temp file for R (Fast I/O)
            full_ko_df.write_csv(trait_file, separator="\t")
            
            # Free RAM before calling R
            del full_ko_df
            gc.collect()
            
            # Run R
            _run_r_script(hsp_script, [str(tree), str(trait_file), str(temp_output), "mp"])
            
            # Read Result
            final_result = pl.read_csv(temp_output, separator="\t")

    # --- STRATEGY B: SERIAL BATCHES (Low RAM) ---
    else:
        print("  Streaming batches from disk...")
        # Get column names
        ko_lazy = pl.scan_csv(db_path, separator="\t", infer_schema_length=0)
        all_columns = ko_lazy.collect_schema().names()
        genome_col = all_columns[0]
        ko_columns = all_columns[1:]
        
        batches = [ko_columns[i:i+final_chunk_size] for i in range(0, len(ko_columns), final_chunk_size)]
        batch_results = []
        
        with tempfile.TemporaryDirectory(prefix="ko_serial_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            for i, batch_cols in enumerate(batches):
                # Read ONLY needed columns from disk
                batch_df = (pl.scan_csv(db_path, separator="\t", infer_schema_length=0)
                            .select([genome_col] + batch_cols)
                            .collect(engine='streaming'))
                
                batch_trait = temp_dir / f"batch_{i+1}_trait.tsv"
                batch_df.write_csv(batch_trait, separator="\t")
                batch_output = temp_dir / f"batch_{i+1}_predicted.tsv"
                
                _run_r_script(hsp_script, [str(tree), str(batch_trait), str(batch_output), "mp"])
                
                batch_results.append(pl.read_csv(batch_output, separator="\t"))
                
                del batch_df
                gc.collect()
        
        # Merge
        final_result = batch_results[0]
        for batch in batch_results[1:]:
            batch = batch.drop("sequence")
            final_result = pl.concat([final_result, batch], how="horizontal")

    # 3. Finalize and Save
    final_result = _ensure_ko_prefix(final_result)

    with gzip.open(output_path, 'wb') as f:
        # type: ignore
        final_result.write_csv(f, separator="\t") # type: ignore
        
    return output_path


def _run_r_script(script_path: Path, args: list) -> None:
    """
    Execute R script via subprocess.
    """
    cmd = ["Rscript", str(script_path)] + args
    
    # Environment optimization: prevent R from using multiple threads for BLAS/LAPACK
    # This ensures CPU usage is controlled by our python script logic
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        raise RuntimeError(
            f"R script failed: {script_path.name}\n"
            f"Exit code: {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )