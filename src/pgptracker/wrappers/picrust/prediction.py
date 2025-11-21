"""
Functional prediction wrapper for PGPTracker.

Replaces PICRUSt2 hsp.py with direct R/Castor calls using adaptive batching for RAM optimization.
"""

import gc
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import polars as pl
from pgptracker.utils.env_manager import detect_free_memory


def _get_r_script(script_name: str) -> Path:
    """
    Locate R script in installed package resources.

    Uses importlib.resources to find scripts bundled in pgptracker.resources.r_scripts/
    regardless of installation method (pip, conda, editable install).
    """
    from importlib import resources
    try:
        r_scripts = resources.files("pgptracker.resources.r_scripts")
        return r_scripts / script_name
    except AttributeError:
        with resources.path("pgptracker.resources.r_scripts", script_name) as p:
            return p


def _calculate_optimal_chunk_size(ko_db_path: Path, available_ram_gb: Optional[float] = None) -> dict:
    """
    Determine whether to process KO table in single pass or batches based on RAM.

    Logic:
    1. Estimates KO table size: ~5MB per column × 10,000 columns = ~50GB
    2. Compares against 50% of available RAM (safety margin)
    3. If sufficient RAM: returns full column count (single-pass)
    4. If insufficient RAM: calculates batch size that fits in memory

    Example:
        System with 16GB RAM, KO table needs 50GB:
        → usable_ram = 8GB → chunk_size ≈ 1600 columns → 7 batches

        System with 64GB RAM, KO table needs 50GB:
        → usable_ram = 32GB → chunk_size = 10,000 (all columns) → 1 batch
    """
    if available_ram_gb is None:
        available_ram_gb = detect_free_memory()

    ko_lazy = pl.scan_csv(ko_db_path, separator="\t")
    num_columns = len(ko_lazy.collect_schema().names()) - 1

    full_table_size_gb = (num_columns * 5) / 1024
    usable_ram_gb = available_ram_gb * 0.5

    if usable_ram_gb >= full_table_size_gb:
        return {
            'chunk_size': num_columns,
            'num_batches': 1,
            'strategy': 'single_pass',
            'message': f"RAM: {available_ram_gb:.1f}GB available, {full_table_size_gb:.1f}GB needed (single pass)"
        }
    else:
        columns_per_gb = num_columns / full_table_size_gb
        chunk_size = int(columns_per_gb * usable_ram_gb)
        chunk_size = max(500, min(chunk_size, 5000))
        num_batches = (num_columns + chunk_size - 1) // chunk_size

        return {
            'chunk_size': chunk_size,
            'num_batches': num_batches,
            'strategy': 'batching',
            'message': f"RAM: {available_ram_gb:.1f}GB available, {full_table_size_gb:.1f}GB needed ({num_batches} batches)"
        }


def predict_functional_profiles(
    tree_path: Path,
    output_dir: Path,
    ref_dir: Path,
    threads: int = 1,
    chunk_size: Optional[int] = 0
) -> Dict[str, Path]:
    """
    Predict gene copy numbers using Hidden State Prediction (Castor/R implementation).

    Uses phylogenetic tree (from SEPP placement) to infer gene content in unknown ASVs
    by analyzing patterns in reference genomes. Two predictions made:

    1. Marker genes (16S rRNA copy number + NSTI quality metric)
       - Single column, processed in one batch
       - NSTI = phylogenetic distance to nearest reference genome

    2. KO genes (10,000+ KEGG Ortholog functional genes)
       - Processed adaptively: single-pass if RAM sufficient, batched if constrained
       - Each batch calls R/Castor independently, results merged horizontally

    Input tree comes from: pipeline_st1.py → place_sequences() → SEPP algorithm
    Reference data from: ~/.pgptracker/db/prokaryotic/ (downloaded by setup command)

    Args:
        chunk_size: 0=auto-detect, -1=force single-pass, >0=manual batch size

    Returns:
        {'marker': marker_nsti_predicted.tsv.gz, 'ko': KO_predicted.tsv.gz}
    """
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


def _predict_marker_16s(tree: Path, db_path: Path, output_dir: Path) -> Path:
    """
    Predict 16S rRNA copy numbers and calculate phylogenetic quality metric (NSTI).

    Workflow:
    1. Load 16S reference table (genome_id → copy_count)
    2. Call hsp.R: uses Castor max parsimony to predict copy counts for unknown tips
    3. Call nsti.R: calculates phylogenetic distance to nearest reference genome
    4. Join predictions with NSTI values

    NSTI (Nearest Sequenced Taxon Index): Lower = better prediction quality
    - NSTI < 0.06: High confidence (close reference exists)
    - NSTI > 2.0: Low confidence (distant from all references)

    Output: marker_nsti_predicted.tsv.gz (columns: sequence, 16S, metadata_NSTI)
    """
    hsp_script = _get_r_script("hsp.R")
    nsti_script = _get_r_script("nsti.R")

    marker_df = pl.read_csv(db_path, separator="\t")
    genome_col = marker_df.columns[0]

    with tempfile.TemporaryDirectory(prefix="marker_") as temp_dir:
        temp_dir = Path(temp_dir)

        trait_file = temp_dir / "16S_trait.tsv"
        marker_df.write_csv(trait_file, separator="\t")

        hsp_output = temp_dir / "16S_predicted.tsv"
        _run_r_script(hsp_script, [str(tree), str(trait_file), str(hsp_output), "mp"])

        known_tips_file = temp_dir / "known_tips.txt"
        marker_df.select(genome_col).write_csv(known_tips_file, has_header=False)

        nsti_output = temp_dir / "nsti.tsv"
        _run_r_script(nsti_script, [str(tree), str(known_tips_file), str(nsti_output)])

        pred_16s = pl.read_csv(hsp_output, separator="\t")
        nsti_df = pl.read_csv(nsti_output, separator="\t")
        result = pred_16s.join(nsti_df, on="sequence", how="left")

    output_path = output_dir / "marker_nsti_predicted.tsv.gz"
    result.write_csv(output_path, separator="\t")

    return output_path


def _predict_ko_adaptive(
    tree: Path,
    db_path: Path,
    output_path: Path,
    chunk_size: Optional[int],
    threads: int
) -> Path:
    """
    Predict KO gene copy numbers with RAM-aware batching strategy.

    Challenge: KO reference table contains 10,000+ columns (genes) × 20,000+ rows (genomes)
    Loading entire table = ~50GB RAM. Solution: process columns in batches.

    Single-pass mode (chunk_size >= total columns):
    - Loads all KO columns at once using Polars streaming engine
    - Single R/Castor call processes entire table
    - Fastest (no merging overhead)

    Batched mode (chunk_size < total columns):
    - Splits columns into N batches (e.g., 2000 columns each)
    - Each batch: load → R/Castor → save result
    - Parallel processing if threads > 1 (ProcessPoolExecutor)
    - Merges batches horizontally (concatenate columns)

    Output: KO_predicted.tsv.gz (ASV_ID → 10,000+ KO gene predictions)
    """
    hsp_script = _get_r_script("hsp.R")

    ko_lazy = pl.scan_csv(db_path, separator="\t")
    all_columns = ko_lazy.collect_schema().names()
    genome_col = all_columns[0]
    ko_columns = all_columns[1:]

    if chunk_size == 0 or chunk_size is None:
        config = _calculate_optimal_chunk_size(db_path)
        chunk_size = config['chunk_size']
        print(f"  {config['message']}")
    elif chunk_size == -1:
        chunk_size = len(ko_columns)
        print("  Single-pass mode (no chunking)")

    if chunk_size >= len(ko_columns):
        ko_table = pl.scan_csv(db_path, separator="\t").collect(engine='streaming')

        with tempfile.TemporaryDirectory(prefix="ko_") as temp_dir:
            temp_dir = Path(temp_dir)
            trait_file = temp_dir / "ko_trait.tsv"
            ko_table.write_csv(trait_file, separator="\t")

            temp_output = temp_dir / "ko_predicted.tsv"
            _run_r_script(hsp_script, [str(tree), str(trait_file), str(temp_output), "mp"])

            result = pl.read_csv(temp_output, separator="\t")

        result.write_csv(output_path, separator="\t")
        return output_path

    num_batches = (len(ko_columns) + chunk_size - 1) // chunk_size
    print(f"  Processing {len(ko_columns)} columns in {num_batches} batches")

    batches = [ko_columns[i:i+chunk_size] for i in range(0, len(ko_columns), chunk_size)]

    with tempfile.TemporaryDirectory(prefix="ko_batch_") as temp_dir:
        temp_dir = Path(temp_dir)

        if threads > 1 and num_batches > 1:
            with ProcessPoolExecutor(max_workers=min(threads, num_batches)) as executor:
                batch_results = list(executor.map(
                    lambda args: _process_batch(*args),
                    [(i+1, batch, db_path, genome_col, tree, hsp_script, temp_dir)
                     for i, batch in enumerate(batches)]
                ))
        else:
            batch_results = []
            for i, batch_cols in enumerate(batches, 1):
                result = _process_batch(i, batch_cols, db_path, genome_col, tree, hsp_script, temp_dir)
                batch_results.append(result)
                del result
                gc.collect()

    final_result = batch_results[0]
    for batch in batch_results[1:]:
        batch = batch.drop("sequence")
        final_result = pl.concat([final_result, batch], how="horizontal")

    final_result.write_csv(output_path, separator="\t")
    return output_path


def _process_batch(
    batch_num: int,
    batch_cols: list,
    db_path: Path,
    genome_col: str,
    tree: Path,
    hsp_script: Path,
    temp_dir: Path
) -> pl.DataFrame:
    """
    Execute single KO batch: select columns → write temp file → call R → return result.

    Uses Polars streaming engine to load only specified columns (memory optimization).
    Temp files needed because R scripts expect file paths, not stdin pipes.
    """
    batch_df = (pl.scan_csv(db_path, separator="\t")
                .select([genome_col] + batch_cols)
                .collect(engine='streaming'))

    batch_trait = temp_dir / f"batch_{batch_num}_trait.tsv"
    batch_df.write_csv(batch_trait, separator="\t")

    batch_output = temp_dir / f"batch_{batch_num}_predicted.tsv"
    _run_r_script(hsp_script, [str(tree), str(batch_trait), str(batch_output), "mp"])

    result = pl.read_csv(batch_output, separator="\t")
    return result


def _run_r_script(script_path: Path, args: list) -> None:
    """
    Execute R script via subprocess (Rscript binary must be in PATH).

    Why shell out to R instead of using Python:
    - Castor library (phylogenetic algorithms) only available in R
    - R implementation is reference standard for Hidden State Prediction
    - Avoids reimplementing complex ancestral state reconstruction algorithms
    """
    cmd = ["Rscript", str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"R script failed: {script_path.name}\n"
            f"Exit code: {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )
