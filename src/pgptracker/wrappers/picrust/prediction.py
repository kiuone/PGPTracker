"""
Functional prediction wrapper for PGPTracker.

Replaces PICRUSt2 hsp.py with direct R/Castor calls using Hybrid Batching for RAM optimization.
"""

import gc
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import polars as pl


# Locate R scripts in package resources
def _get_r_script(script_name: str) -> Path:
    """Get path to R script in package resources."""
    from importlib import resources
    try:
        # Python 3.9+
        r_scripts = resources.files("pgptracker.resources.r_scripts")
        return r_scripts / script_name
    except AttributeError:
        # Python 3.8 fallback
        with resources.path("pgptracker.resources.r_scripts", script_name) as p:
            return p


def predict_functional_profiles(
    tree_path: Path,
    output_dir: Path,
    ref_dir: Path,
    threads: int = 1,
    chunk_size: int = 2000
) -> Dict[str, Path]:
    """
    Predict functional profiles using Hidden State Prediction with Hybrid Batching.

    Processes marker genes (16S + NSTI) in single batch.
    Processes KO table in chunks to minimize RAM usage.

    Args:
        tree_path: Path to phylogenetic tree (.tre)
        output_dir: Directory for output files
        ref_dir: Path to prokaryotic reference database
        threads: Number of R threads (not used for batching)
        chunk_size: Number of KO columns per batch (default: 2000)

    Returns:
        Dictionary with paths:
            - 'marker': marker_nsti_predicted.tsv.gz
            - 'ko': KO_predicted.tsv.gz

    Raises:
        FileNotFoundError: If inputs don't exist
        RuntimeError: If R scripts fail
    """
    # Validate inputs
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")

    # Reference files
    marker_db = ref_dir / "16S.txt.gz"
    ko_db = ref_dir / "ko.txt.gz"

    for db_file in [marker_db, ko_db]:
        if not db_file.exists():
            raise FileNotFoundError(f"Reference database missing: {db_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Predict marker genes (16S) with NSTI
    print("\nPredicting marker genes (16S) and calculating NSTI")
    marker_path = _predict_marker_16s(
        tree=tree_path,
        db_path=marker_db,
        output_dir=output_dir,
        ref_dir=ref_dir
    )

    # Step 2: Predict KO with Hybrid Batching
    print(f"\nPredicting KO abundances using Hybrid Batching (chunk_size={chunk_size})")
    ko_path = _predict_ko_hybrid(
        tree=tree_path,
        db_path=ko_db,
        output_path=output_dir / "KO_predicted.tsv.gz",
        chunk_size=chunk_size
    )

    return {'marker': marker_path, 'ko': ko_path}


def _predict_marker_16s(
    tree: Path,
    db_path: Path,
    output_dir: Path,
    ref_dir: Path
) -> Path:
    """
    Predict 16S copy numbers and calculate NSTI (single batch).

    Args:
        tree: Phylogenetic tree
        db_path: Path to 16S reference database
        output_dir: Output directory
        ref_dir: Reference directory

    Returns:
        Path to marker_nsti_predicted.tsv.gz
    """
    hsp_script = _get_r_script("hsp.R")
    nsti_script = _get_r_script("nsti.R")

    # Load 16S database (small, single column)
    marker_df = pl.read_csv(db_path, separator="\t")
    genome_col = marker_df.columns[0]

    with tempfile.TemporaryDirectory(prefix="marker_") as temp_dir:
        temp_dir = Path(temp_dir)

        # Write trait table for R
        trait_file = temp_dir / "16S_trait.tsv"
        marker_df.write_csv(trait_file, separator="\t")

        # Run HSP for 16S
        hsp_output = temp_dir / "16S_predicted.tsv"
        _run_r_hsp(hsp_script, tree, trait_file, hsp_output, method="mp")

        # Calculate NSTI
        # Extract reference genome IDs
        known_tips_file = temp_dir / "known_tips.txt"
        marker_df.select(genome_col).write_csv(known_tips_file, has_header=False)

        nsti_output = temp_dir / "nsti.tsv"
        _run_r_nsti(nsti_script, tree, known_tips_file, nsti_output)

        # Join 16S predictions with NSTI
        pred_16s = pl.read_csv(hsp_output, separator="\t")
        nsti_df = pl.read_csv(nsti_output, separator="\t")

        result = pred_16s.join(nsti_df, on="sequence", how="left")

    # Write final output
    output_path = output_dir / "marker_nsti_predicted.tsv.gz"
    result.write_csv(output_path, separator="\t")
    print(f"  Marker prediction completed: {output_path}")

    return output_path


def _predict_ko_hybrid(
    tree: Path,
    db_path: Path,
    output_path: Path,
    chunk_size: int
) -> Path:
    """
    Predict KO abundances using Hybrid Batching for RAM optimization.

    Strategy:
    1. Lazy-load column names from KO database
    2. Process columns in chunks (e.g., 2000 at a time)
    3. For each chunk:
       - Load only those columns (~50MB)
       - Run HSP in R
       - Store result as LazyFrame
    4. Concatenate all chunks horizontally
    5. Write final output

    Args:
        tree: Phylogenetic tree
        db_path: Path to KO database (10,000+ columns)
        output_path: Path for final output
        chunk_size: Number of KO columns per batch

    Returns:
        Path to KO_predicted.tsv.gz
    """
    hsp_script = _get_r_script("hsp.R")

    # Lazy-load to get column names without loading entire file
    ko_lazy = pl.scan_csv(db_path, separator="\t")
    all_columns = ko_lazy.collect_schema().names()
    genome_col = all_columns[0]
    ko_columns = all_columns[1:]  # All KO columns

    print(f"  Total KO columns: {len(ko_columns)}")
    print(f"  Processing in {(len(ko_columns) + chunk_size - 1) // chunk_size} batches")

    batch_results = []

    with tempfile.TemporaryDirectory(prefix="ko_batch_") as temp_dir:
        temp_dir = Path(temp_dir)

        # Process KO columns in chunks
        for i in range(0, len(ko_columns), chunk_size):
            batch_cols = ko_columns[i:i + chunk_size]
            batch_num = i // chunk_size + 1
            total_batches = (len(ko_columns) + chunk_size - 1) // chunk_size

            print(f"    Batch {batch_num}/{total_batches}: Processing {len(batch_cols)} KO columns")

            # Load only this batch of columns
            batch_df = pl.read_csv(
                db_path,
                separator="\t",
                columns=[genome_col] + batch_cols
            )

            # Write batch trait table
            batch_trait = temp_dir / f"batch_{batch_num}_trait.tsv"
            batch_df.write_csv(batch_trait, separator="\t")

            # Run HSP on batch
            batch_output = temp_dir / f"batch_{batch_num}_predicted.tsv"
            _run_r_hsp(hsp_script, tree, batch_trait, batch_output, method="mp")

            # Read result and store
            batch_result = pl.read_csv(batch_output, separator="\t")
            batch_results.append(batch_result)

            # Free memory
            del batch_df, batch_result
            gc.collect()

    # Concatenate all batches horizontally
    print("  Merging all batches")

    # First batch has 'sequence' column, subsequent batches need it removed
    final_result = batch_results[0]
    for batch in batch_results[1:]:
        # Drop 'sequence' column from subsequent batches (already in first)
        batch = batch.drop("sequence")
        # Horizontal concat
        final_result = pl.concat([final_result, batch], how="horizontal")

    # Write final output
    final_result.write_csv(output_path, separator="\t")
    print(f"  KO prediction completed: {output_path}")

    return output_path


def _run_r_hsp(
    script_path: Path,
    tree: Path,
    trait_file: Path,
    output_file: Path,
    method: str = "mp"
) -> None:
    """
    Execute R HSP script.

    Args:
        script_path: Path to hsp.R
        tree: Tree file
        trait_file: Trait table
        output_file: Output path
        method: HSP method ('mp' or 'emp_prob')

    Raises:
        RuntimeError: If R script fails
    """
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
    """
    Execute R NSTI script.

    Args:
        script_path: Path to nsti.R
        tree: Tree file
        known_tips_file: File with reference sequence IDs
        output_file: Output path

    Raises:
        RuntimeError: If R script fails
    """
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
