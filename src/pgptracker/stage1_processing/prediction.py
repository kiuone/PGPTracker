"""
Functional prediction wrapper for PGPTracker.

Predicts gene copy numbers using Hidden State Prediction (Castor/R) on phylogenetic tree.

Performance Strategy (matching PICRUSt2 architecture):

The R scripts process KO columns sequentially (not parallelized).
Parallelization happens at the Python level by chunking KOs and processing chunks
concurrently using ProcessPoolExecutor.

1. High-RAM Mode (30GB+ available):
   - Chunk size: 1000-2000 KOs per chunk
   - Parallel workers: All available threads (e.g., 8 cores = 8 chunks running simultaneously)
   - Each R process handles one chunk single-threaded
   - Target: <10 min on 8 cores

2. Low-RAM Mode (10-30GB available):
   - Chunk size: 500-1000 KOs per chunk (smaller chunks for RAM safety)
   - Parallel workers: All available threads
   - Target: <60 min max

Minimum Requirements: 16GB total RAM, 4 CPU cores
"""

import gc
import os
import subprocess
import tempfile
import gzip
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional

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
    available_ram_gb: Optional[float] = None,
    threads: int = 1
) -> dict:
    """
    Calculate chunk size and worker count based on available RAM.

    Strategy (like PICRUSt2):
    - Always use chunked parallel processing
    - High-RAM: larger chunks (1000-2000 KOs), more parallel workers
    - Low-RAM: smaller chunks (500-1000 KOs), RAM-safe batching

    Args:
        ko_db_path: Compressed KO reference database
        available_ram_gb: Override for free memory detection
        threads: Available CPU cores

    Returns:
        {'chunk_size': int, 'workers': int, 'message': str}
    """
    if available_ram_gb is None:
        available_ram_gb = detect_free_memory()

    # Count columns without loading data
    try:
        ko_lazy = pl.scan_csv(ko_db_path, separator="\t", infer_schema_length=0)
        num_columns = len(ko_lazy.collect_schema().names()) - 1
    except Exception:
        num_columns = 10543  # PICRUSt2 default

    # Check minimum RAM requirement
    if available_ram_gb < 10.0:
        raise RuntimeError(
            f"Insufficient RAM: {available_ram_gb:.1f}GB available.\n"
            f"Minimum 10GB required. Current system has <16GB total RAM.\n"
            f"PGPTracker requires at least 16GB RAM and 4 CPU cores."
        )

    # High-RAM Mode: 30GB+ available
    # Use larger chunks and all available workers
    if available_ram_gb >= 30.0:
        chunk_size = 1000  # PICRUSt2-style chunking (500-2000 range)
        workers = threads  # Use all available CPUs
        num_batches = (num_columns + chunk_size - 1) // chunk_size

        return {
            'chunk_size': chunk_size,
            'workers': workers,
            'message': (f"High-RAM Mode: {available_ram_gb:.1f}GB available. "
                        f"Processing {num_columns} KOs in {num_batches} chunks "
                        f"({chunk_size} KOs/chunk, {workers} parallel workers, target <10min)")
        }

    # Low-RAM Mode: 10-30GB available
    # Use smaller chunks for RAM safety
    else:
        chunk_size = 500  # Smaller chunks to avoid OOM
        workers = min(threads, 4)  # Limit concurrent workers to avoid memory spikes
        num_batches = (num_columns + chunk_size - 1) // chunk_size

        return {
            'chunk_size': chunk_size,
            'workers': workers,
            'message': (f"Low-RAM Mode: {available_ram_gb:.1f}GB available. "
                        f"Processing {num_columns} KOs in {num_batches} chunks "
                        f"({chunk_size} KOs/chunk, {workers} parallel workers, target <60min)")
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
    Predict gene copy numbers for ASVs using phylogenetic tree.

    Workflow:
    1. Marker prediction (16S): Single-threaded, fast (<1 min)
    2. KO prediction: Adaptive strategy based on RAM
       - High-RAM (30GB+): Single R call, multi-threaded, <10 min
       - Low-RAM (10-30GB): Parallel batches, <60 min

    Args:
        tree_path: Newick tree from phylogeny.place_sequences()
        output_dir: Output directory for predictions
        ref_dir: Reference database directory
        threads: Available CPU cores
        chunk_size: Manual override (0=auto, -1=force high-RAM)

    Returns:
        {'marker': marker_nsti_predicted.tsv.gz, 'ko': KO_predicted.tsv.gz}
    """
    ref_dir = ref_dir.resolve()
    output_dir = output_dir.resolve()
    tree_path = tree_path.resolve()

    marker_db = ref_dir / "16S.txt.gz"
    ko_db = ref_dir / "ko.txt.gz"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Predicting marker genes (16S) and NSTI...")
    marker_path = _predict_marker_16s(tree_path, marker_db, output_dir)

    print("Predicting KO abundances...")
    ko_path = _predict_ko_adaptive(
        tree_path, ko_db, output_dir / "KO_predicted.tsv.gz", chunk_size, threads
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
    chunk_size: Optional[int],
    threads: int
) -> Path:
    """
    Predict KO gene copy numbers using chunked parallel processing.

    Architecture (matching PICRUSt2):
    - Splits KO columns into chunks (500-2000 KOs per chunk)
    - Processes chunks in parallel using ProcessPoolExecutor
    - Each worker: load subset → R prediction (single-threaded) → return result
    - Parallelization at Python level, not R level

    Args:
        tree: Phylogenetic tree from place_sequences()
        db_path: Compressed KO reference database (ko.txt.gz)
        output_path: Output path for predictions
        chunk_size: Manual override (0=auto, >0=custom chunk size)
        threads: Available CPU cores (determines max_workers)

    Returns:
        Path to KO_predicted.tsv.gz
    """
    hsp_script = _get_r_script("hsp.R")

    # 1. Determine Strategy
    config = _calculate_processing_strategy(db_path, threads=threads)

    # Manual Override
    if chunk_size and chunk_size > 0:
        final_chunk_size = chunk_size
        final_workers = config['workers']
        print(f"  Manual Override: Chunk size {final_chunk_size}, {final_workers} workers")
    else:
        final_chunk_size = int(config['chunk_size'])
        final_workers = int(config['workers'])
        print(f"  {config['message']}")

    # 2. Prepare Batches
    ko_lazy = pl.scan_csv(db_path, separator="\t", infer_schema_length=0)
    all_columns = ko_lazy.collect_schema().names()
    genome_col = all_columns[0]
    ko_columns = all_columns[1:]

    batches = [ko_columns[i:i+final_chunk_size] for i in range(0, len(ko_columns), final_chunk_size)]
    print(f"  Processing {len(ko_columns)} KOs in {len(batches)} chunks using {final_workers} parallel workers...")

    # 3. Parallel Batch Processing (PICRUSt2-style)
    with ProcessPoolExecutor(max_workers=final_workers) as executor:
        futures = []
        for i, batch_cols in enumerate(batches):
            future = executor.submit(
                _process_ko_batch,
                db_path, tree, hsp_script, genome_col, batch_cols, i
            )
            futures.append(future)

        # Collect results as they complete
        batch_results = [future.result() for future in futures]

    # 4. Merge Results Horizontally
    final_result = batch_results[0]
    for batch in batch_results[1:]:
        batch = batch.drop("sequence")
        final_result = pl.concat([final_result, batch], how="horizontal")

    # 5. Finalize and Save
    final_result = _ensure_ko_prefix(final_result)

    with gzip.open(output_path, 'wb') as f:
        final_result.write_csv(f, separator="\t")  # type: ignore

    return output_path


def _process_ko_batch(
    db_path: Path,
    tree: Path,
    hsp_script: Path,
    genome_col: str,
    batch_cols: list,
    batch_index: int
) -> pl.DataFrame:
    """
    Worker function for parallel KO batch processing.

    This function is executed in a separate process by ProcessPoolExecutor.
    Each worker loads a subset of KO columns, runs R prediction, and returns results.

    Args:
        db_path: Path to compressed KO database
        tree: Path to phylogenetic tree
        hsp_script: Path to hsp.R script
        genome_col: Name of genome ID column
        batch_cols: List of KO column names for this batch
        batch_index: Batch number for logging

    Returns:
        DataFrame with predictions for this batch's KO columns
    """
    import gc

    # Load only this batch's columns from disk
    batch_df = (
        pl.scan_csv(db_path, separator="\t", infer_schema_length=0)
        .select([genome_col] + batch_cols)
        .collect(engine='streaming')
    )

    with tempfile.TemporaryDirectory(prefix=f"ko_batch_{batch_index}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        batch_trait = temp_dir / "trait.tsv"
        batch_output = temp_dir / "predicted.tsv"

        batch_df.write_csv(batch_trait, separator="\t")

        # Free RAM before R call
        del batch_df
        gc.collect()

        # Run R (single-threaded per worker, parallelism managed by ProcessPoolExecutor)
        _run_r_script(hsp_script, [str(tree), str(batch_trait), str(batch_output), "mp"])

        result = pl.read_csv(batch_output, separator="\t")

    return result


def _run_r_script(script_path: Path, args: list, threads: int = 1) -> None:
    """
    Execute R script via subprocess.

    Args:
        script_path: Path to R script
        args: Command line arguments for the script
        threads: Number of OpenMP threads (always 1 for chunked parallelization)

    Note:
    - R scripts process KO columns sequentially (not parallelized)
    - Setting OMP_NUM_THREADS=1 prevents R from using multiple threads
    - Parallelization happens at Python level via ProcessPoolExecutor
    """
    cmd = ["Rscript", str(script_path)] + args

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"  # Single-threaded R, Python-level parallelization

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        raise RuntimeError(
            f"R script failed: {script_path.name}\n"
            f"Exit code: {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )