"""
QIIME2 Taxonomic Classification Wrapper for PGPTracker.

This module wraps the QIIME2 feature-classifier (Bokulich et al., 2018)
to assign taxonomy to representative sequences.
"""

import subprocess
from pathlib import Path
import requests
import appdirs
import sys
import shutil
from typing import Optional
from tqdm import tqdm
from pgptracker.utils.env_manager import run_command
from pgptracker.utils.validator import ValidationError

CLASSIFIER_URL = "https://ftp.microbio.me/greengenes_release/2024.09/2024.09.backbone.v4.nb.qza"
CLASSIFIER_FILENAME = "2024.09.backbone.v4.nb.qza"
APP_NAME = "PGPTracker"
APP_AUTHOR = "PGPTracker"

# --- Private Helpers ---

def _get_cache_dir() -> Path:
    """Finds the user-specific cache directory for PGPTracker."""
    cache_dir = Path(appdirs.user_cache_dir(APP_NAME, APP_AUTHOR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def _fix_qiime_header(exported_tsv: Path, final_tsv: Path) -> None:
    """
    Rewrites the QIIME2 taxonomy export header to be BIOM-compatible.
    """
    with open(exported_tsv, 'r') as f_in, open(final_tsv, 'w') as f_out:
        # Read and discard the original header (e.g., "Feature-ID\ttaxonomy...")
        f_in.readline()

        # Write the new, BIOM-compatible header
        f_out.write("#OTU/ASV_ID\ttaxonomy\tconfidence\n")

        # Copy the rest of the file data efficiently
        shutil.copyfileobj(f_in, f_out)

def _get_default_classifier() -> Path:
    """
    Gets the path to the default classifier, downloading it if it doesn't exist.
    """
    cache_dir = _get_cache_dir()
    classifier_path = cache_dir / CLASSIFIER_FILENAME
    
    if classifier_path.exists():
        print(f"  -> Found default classifier in cache: {classifier_path}")
        return classifier_path

    print(f"  -> Default classifier not found. Downloading to cache:\n     {classifier_path}")
    
    with requests.get(CLASSIFIER_URL, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        progress_bar = tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True,
            desc="Downloading Greengenes")
        
        with open(classifier_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()

    # Basic integrity check
    if total_size > 0 and classifier_path.stat().st_size != total_size:
        classifier_path.unlink(missing_ok=True) # Clean up incomplete file
        raise IOError(f"Download incomplete. Expected {total_size} bytes, got {classifier_path.stat().st_size}.")

    print("  -> Download complete.")
    return classifier_path

# --- Main Function ---

def classify_taxonomy(
    rep_seqs_path: Path,
    seq_format: str,
    classifier_qza_path: Optional[Path], 
    output_dir: Path,
    threads: int
) -> Path:
    """
    Runs QIIME2 'classify-sklearn', exports, and fixes the header.

    Args:
        rep_seqs_path: Path to representative sequences (.qza or .fasta).
        seq_format: The format of the sequences ('qza' or 'fasta').
        classifier_qza_path: Path to a custom classifier. If None, downloads default.
        output_dir: The main output directory for this run.
        threads: Number of threads to use.

    Returns:
        Path: The path to the final, header-fixed 'taxonomy.tsv' file.
        
    Raises:
        ValidationError: If any step fails to produce the expected output.
    """
    # 1. Resolve classifier path
    classifier_to_use = None

    if classifier_qza_path:
        # User provided a custom classifier (Path)
        print(f"  -> Using custom classifier from: {classifier_qza_path}")
        if not classifier_qza_path.exists():
            raise FileNotFoundError(f"Custom classifier not found: {classifier_qza_path}")
        classifier_to_use = classifier_qza_path
    else:
        # Get default (download or cache)
        classifier_to_use = _get_default_classifier()

    # 2. Define pathways and ensure output directory exists
    classify_dir = output_dir / "taxonomy"
    classify_dir.mkdir(parents=True, exist_ok=True)
    classified_qza = classify_dir / "taxonomy.qza"
    export_dir = classify_dir / "exported_taxonomy"
    final_taxonomy_tsv = classify_dir / "taxonomy.tsv"

    # 3. Import sequences to .qza if needed
    rep_seqs_qza_to_use = rep_seqs_path
    if seq_format != 'qza':
        # seq_format is 'fasta', needs to import
        imported_qza = classify_dir / "imported_rep_seqs.qza"

        cmd_import = [
            "qiime", "tools", "import",
            "--type", "FeatureData[Sequence]",
            "--input-path", str(rep_seqs_path),
            "--output-path", str(imported_qza)]

        run_command("qiime", cmd_import, check=True, capture_output=True)
        rep_seqs_qza_to_use = imported_qza
        print(f" -> Successfully imported to {imported_qza}")

    # 4. Run classify-sklearn
    print(" -> Running QIIME2 classify-sklearn (Bokulich et al., 2018)...")

    cmd_classify = [
        "qiime", "feature-classifier", "classify-sklearn",
        "--i-reads", str(rep_seqs_qza_to_use),
        "--i-classifier", str(classifier_to_use),
        "--o-classification", str(classified_qza),
        "--p-n-jobs", str(threads)
    ]

    run_command("qiime", cmd_classify, check=True, capture_output=True)

    # 5. Export .qza to .tsv
    cmd_export = [
        "qiime", "tools", "export",
        "--input-path", str(classified_qza),
        "--output-path", str(export_dir)
    ]

    try:
        run_command("qiime", cmd_export, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f" [ERROR] QIIME2 tools export failed: {e.stderr}", file=sys.stderr)
        raise RuntimeError("Taxonomy export failed.") from e

    exported_tsv = export_dir / "taxonomy.tsv"
    if not exported_tsv.exists():
        raise ValidationError(f"QIIME2 export failed to create {exported_tsv}")

    # 6. Fix the header of the exported taxonomy file
    _fix_qiime_header(exported_tsv, final_taxonomy_tsv)

    return final_taxonomy_tsv