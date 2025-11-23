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
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
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
    # Pre-load works, now use parallel chunked processing for speed
    if available_ram_gb >= 30.0:
        chunk_size = 1000  # Large chunks (fast per-chunk processing)
        workers = threads  # Use ALL available cores for parallelization
        num_batches = (num_columns + chunk_size - 1) // chunk_size

        return {
            'chunk_size': chunk_size,
            'workers': workers,
            'message': (f"High-RAM Mode: {available_ram_gb:.1f}GB available. "
                        f"Processing {num_columns} KOs in {num_batches} chunks "
                        f"({chunk_size} KOs/chunk, {workers} parallel workers, target <10min)")
        }

    # Low-RAM Mode: 10-30GB available
    # Use chunked parallel processing to avoid OOM
    else:
        chunk_size = 500  # Smaller chunks to avoid OOM
        workers = min(threads, 4)  # Limit concurrent workers to avoid memory spikes
        num_batches = (num_columns + chunk_size - 1) // chunk_size

        return {
            'chunk_size': chunk_size,
            'workers': workers,
            'message': (f"Low-RAM Mode: {available_ram_gb:.1f}GB available. "
                        f"Processing {num_columns} KOs in {num_batches} chunks "
                        f"({chunk_size} KOs/chunk, {workers} parallel workers, target <60min)")}


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
        tree_path, ko_db, output_dir / "KO_predicted.tsv.gz", chunk_size, threads)

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

@profile_memory
def _predict_ko_adaptive(
    tree: Path,
    db_path: Path,
    output_path: Path,
    chunk_size: Optional[int],
    threads: int
) -> Path:
    """
    Predict KO gene copy numbers using EXACT PICRUSt2 architecture.

    Uses pandas + joblib (same as PICRUSt2) instead of polars + ProcessPoolExecutor.

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

    # 2. PRE-LOAD using PANDAS (exactly like PICRUSt2)
    print("  Loading full KO table into RAM (pandas)...")
    trait_tab = pd.read_csv(db_path, sep="\t", index_col=0, low_memory=False)

    num_cols = trait_tab.shape[1]
    print(f"  Loaded {num_cols} KO columns ({trait_tab.shape[0]} genomes)")

    # 3. Calculate number of chunks (PICRUSt2 formula)
    num_chunks = int(num_cols / (final_chunk_size + 1))
    if num_chunks == 0:
        num_chunks = 1

    print(f"  Creating {num_chunks} chunk files...")

    with tempfile.TemporaryDirectory(prefix="ko_chunks_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        file_subsets = []

        # 4. Create chunks using iloc (EXACTLY like PICRUSt2)
        for i in range(num_chunks):
            start_col = i * final_chunk_size
            end_col = (i + 1) * final_chunk_size

            subset_tab = trait_tab.iloc[:, start_col:end_col]
            subset_file = temp_dir / f"subset_tab_{i}.tsv"
            subset_tab.to_csv(subset_file, sep="\t")
            file_subsets.append(subset_file)

        # Free RAM
        del trait_tab
        gc.collect()

        # 5. Parallel processing using joblib (EXACTLY like PICRUSt2)
        print(f"  Processing {len(file_subsets)} chunks using {final_workers} parallel workers (joblib)...")

        batch_results = Parallel(n_jobs=final_workers)(
            delayed(_process_ko_chunk_file)(tree, chunk_file, hsp_script, i)
            for i, chunk_file in enumerate(file_subsets))

    # 6. Merge with pandas
    final_df = pd.concat(batch_results, axis=1)

    # 7. Ensure ko: prefix
    final_df = _ensure_ko_prefix_pandas(final_df)

    # 8. Save
    final_df.to_csv(output_path, sep="\t", compression='gzip')

    return output_path

def _process_ko_chunk_file(
    tree: Path,
    chunk_file: Path,
    hsp_script: Path,
    chunk_index: int
) -> pd.DataFrame:
    """
    Worker function for parallel KO chunk processing (PICRUSt2 architecture).

    Uses pandas (like PICRUSt2) for compatibility with joblib.Parallel.

    Args:
        tree: Path to phylogenetic tree
        chunk_file: Path to PRE-CREATED chunk file (uncompressed .tsv)
        hsp_script: Path to hsp.R script
        chunk_index: Chunk number for logging

    Returns:
        pandas DataFrame with predictions for this chunk's KO columns
    """
    with tempfile.TemporaryDirectory(prefix=f"ko_worker_{chunk_index}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        output_file = temp_dir / "predicted.tsv"

        # Run R with PRE-CREATED chunk file (no .gz reading!)
        _run_r_script(hsp_script, [str(tree), str(chunk_file), str(output_file), "mp"])

        # Read result with pandas (PICRUSt2 style)
        result = pd.read_csv(output_file, sep="\t", index_col=0)

    return result

def _ensure_ko_prefix_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all KO columns have 'ko:' prefix (pandas version).

    Args:
        df: DataFrame with KO predictions

    Returns:
        DataFrame with renamed columns
    """
    new_cols = []
    for col in df.columns:
        if col.startswith("K") and not col.startswith("ko:"):
            new_cols.append(f"ko:{col}")
        else:
            new_cols.append(col)

    df.columns = new_cols
    return df


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
            f"STDERR: {result.stderr}")