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


def _get_r_script(script_name: str) -> Path:
    """Get path to R script in package resources."""
    from importlib import resources
    try:
        r_scripts = resources.files("pgptracker.resources.r_scripts")
        return r_scripts / script_name
    except AttributeError:
        with resources.path("pgptracker.resources.r_scripts", script_name) as p:
            return p


def _detect_available_ram() -> float:
    """
    Detect available system RAM in GB.

    Returns:
        Available RAM in GB (fallback: 8.0 if detection fails)
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except:
        pass
    return 8.0


def _calculate_optimal_chunk_size(ko_db_path: Path, available_ram_gb: Optional[float] = None) -> dict:
    """
    Calculate optimal chunk size based on available RAM.

    Args:
        ko_db_path: Path to KO database
        available_ram_gb: Override RAM detection (None = auto-detect)

    Returns:
        dict: {chunk_size, num_batches, strategy, message}
    """
    if available_ram_gb is None:
        available_ram_gb = _detect_available_ram()

    ko_lazy = pl.scan_csv(ko_db_path, separator="\t")
    num_columns = len(ko_lazy.collect_schema().names()) - 1

    # Rough estimate: 5MB per column for ~10k genomes
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
    Predict functional profiles using Hidden State Prediction.

    Args:
        tree_path: Path to phylogenetic tree (.tre)
        output_dir: Directory for output files
        ref_dir: Path to prokaryotic reference database
        threads: Number of threads for parallel processing
        chunk_size: KO columns per batch (0=auto, -1=no chunking, >0=manual)

    Returns:
        Dictionary: {'marker': marker_path, 'ko': ko_path}

    Raises:
        FileNotFoundError: If inputs don't exist
        RuntimeError: If R scripts fail
    """
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")

    marker_db = ref_dir / "16S.txt.gz"
    ko_db = ref_dir / "ko.txt.gz"

    for db_file in [marker_db, ko_db]:
        if not db_file.exists():
            raise FileNotFoundError(f"Reference database missing: {db_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Predicting marker genes (16S) and NSTI...")
    marker_path = _predict_marker_16s(
        tree=tree_path,
        db_path=marker_db,
        output_dir=output_dir,
        ref_dir=ref_dir
    )

    print("Predicting KO abundances...")
    ko_path = _predict_ko_adaptive(
        tree=tree_path,
        db_path=ko_db,
        output_path=output_dir / "KO_predicted.tsv.gz",
        chunk_size=chunk_size,
        threads=threads
    )

    return {'marker': marker_path, 'ko': ko_path}


def _predict_marker_16s(
    tree: Path,
    db_path: Path,
    output_dir: Path,
    ref_dir: Path
) -> Path:
    """Predict 16S copy numbers and calculate NSTI."""
    hsp_script = _get_r_script("hsp.R")
    nsti_script = _get_r_script("nsti.R")

    marker_df = pl.read_csv(db_path, separator="\t")
    genome_col = marker_df.columns[0]

    with tempfile.TemporaryDirectory(prefix="marker_") as temp_dir:
        temp_dir = Path(temp_dir)

        trait_file = temp_dir / "16S_trait.tsv"
        marker_df.write_csv(trait_file, separator="\t")

        hsp_output = temp_dir / "16S_predicted.tsv"
        _run_r_hsp(hsp_script, tree, trait_file, hsp_output, method="mp")

        known_tips_file = temp_dir / "known_tips.txt"
        marker_df.select(genome_col).write_csv(known_tips_file, has_header=False)

        nsti_output = temp_dir / "nsti.tsv"
        _run_r_nsti(nsti_script, tree, known_tips_file, nsti_output)

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
    Adaptive KO prediction with automatic RAM optimization.

    Args:
        tree: Phylogenetic tree
        db_path: KO database path
        output_path: Output file path
        chunk_size: 0=auto, -1=force single pass, >0=manual
        threads: Number of parallel threads
    """
    hsp_script = _get_r_script("hsp.R")

    ko_lazy = pl.scan_csv(db_path, separator="\t")
    all_columns = ko_lazy.collect_schema().names()
    genome_col = all_columns[0]
    ko_columns = all_columns[1:]

    # Determine chunk size
    if chunk_size == 0 or chunk_size is None:
        config = _calculate_optimal_chunk_size(db_path)
        chunk_size = config['chunk_size']
        print(f"  {config['message']}")
    elif chunk_size == -1:
        chunk_size = len(ko_columns)
        print(f"  Single-pass mode (no chunking)")

    # Single-pass execution
    if chunk_size >= len(ko_columns):
        ko_table = pl.scan_csv(db_path, separator="\t").collect(engine='streaming')

        with tempfile.TemporaryDirectory(prefix="ko_") as temp_dir:
            temp_dir = Path(temp_dir)
            trait_file = temp_dir / "ko_trait.tsv"
            ko_table.write_csv(trait_file, separator="\t")

            temp_output = temp_dir / "ko_predicted.tsv"
            _run_r_hsp(hsp_script, tree, trait_file, temp_output, method="mp")

            result = pl.read_csv(temp_output, separator="\t")

        result.write_csv(output_path, separator="\t")
        return output_path

    # Batched execution
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

    # Merge batches
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
    """Process single batch of KO columns."""
    batch_df = (pl.scan_csv(db_path, separator="\t")
                .select([genome_col] + batch_cols)
                .collect(engine='streaming'))

    batch_trait = temp_dir / f"batch_{batch_num}_trait.tsv"
    batch_df.write_csv(batch_trait, separator="\t")

    batch_output = temp_dir / f"batch_{batch_num}_predicted.tsv"
    _run_r_hsp(hsp_script, tree, batch_trait, batch_output, method="mp")

    result = pl.read_csv(batch_output, separator="\t")
    return result


def _run_r_hsp(
    script_path: Path,
    tree: Path,
    trait_file: Path,
    output_file: Path,
    method: str = "mp"
) -> None:
    """Execute R HSP script."""
    cmd = [
        "Rscript",
        str(script_path),
        str(tree),
        str(trait_file),
        str(output_file),
        method
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"R HSP failed with exit code {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )


def _run_r_nsti(
    script_path: Path,
    tree: Path,
    known_tips_file: Path,
    output_file: Path
) -> None:
    """Execute R NSTI script."""
    cmd = [
        "Rscript",
        str(script_path),
        str(tree),
        str(known_tips_file),
        str(output_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"R NSTI failed with exit code {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )
