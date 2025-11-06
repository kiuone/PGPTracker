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
from typing import Optional
from tqdm import tqdm
from pgptracker.utils.env_manager import run_command
from pgptracker.utils.validator import ValidationError

CLASSIFIER_URL = "https://ftp.microbio.me/greengenes_release/2024.09/2024.09.backbone.v4.nb.qza"
CLASSIFIER_FILENAME = "2024.09.backbone.v4.nb.qza"
APP_NAME = "PGPTracker"
APP_AUTHOR = "PGPTracker"

def _get_cache_dir() -> Path:
    """Finds the user-specific cache directory for PGPTracker."""
    cache_dir = Path(appdirs.user_cache_dir(APP_NAME, APP_AUTHOR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def _get_default_classifier() -> Path:
    """
    Gets the path to the default classifier, downloading it if it doesn't exist.
    """
    cache_dir = _get_cache_dir()
    classifier_path = cache_dir / CLASSIFIER_FILENAME
    
    if classifier_path.exists():
        print(f"  -> Found default classifier in cache: {classifier_path}")
        return classifier_path

    print(f"  -> Default classifier not found. Downloading to cache:\n     {classifier_path}")
    
    try:
        with requests.get(CLASSIFIER_URL, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            progress_bar = tqdm(
                total=total_size, 
                unit='iB', 
                unit_scale=True,
                desc="Downloading Greengenes"
            )
            
            with open(classifier_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            raise IOError("Download incomplete.")

        print("  -> Download complete.")
        return classifier_path

    except (requests.RequestException, IOError, OSError) as e:
        print(f"[ERROR] Failed to download classifier: {e}", file=sys.stderr)
        if classifier_path.exists():
            classifier_path.unlink() # Remove incomplete file
        raise RuntimeError(f"Failed to download default classifier from {CLASSIFIER_URL}") from e

def classify_taxonomy(
rep_seqs_path: Path,
seq_format: str,
classifier_qza_path: Optional[Path], # Can be Path or Traversable
output_dir: Path,
threads: int
) -> Path:

    """
Runs QIIME2 'classify-sklearn', exports, and fixes the header.

Args:
    rep_seqs_path: Path to representative sequences (.qza) from the user.
    seq_format: The format of the sequences ('qza' or 'fasta').
    classifier_qza_path: Path to a custom classifier. If None, downloads default.
    output_dir: The main output directory for this run.
    threads: Number of threads to use.

Returns:
    Path: The path to the final, header-fixed 'taxonomy.tsv' file.
    
Raises:
    ValidationError: If any step fails to produce the expected output.
    """
    # Resolve classifier path
    classifier_to_use = None
    is_packaged_classifier = False

    try:
        if classifier_qza_path and isinstance(classifier_qza_path, Path):
                # 1. User provided a custom classifier (Path)
                print(f"   -> Using custom classifier from: {classifier_qza_path}")
                if not classifier_qza_path.exists():
                    raise FileNotFoundError(f"Custom classifier not found: {classifier_qza_path}")
                classifier_obj_to_use = classifier_qza_path
                is_packaged_classifier = False
        elif classifier_qza_path:
                # 2. Default classifier object (Traversable)
                print("    -> Using default bundled Greengenes (2024.09) classifier.")
                classifier_obj_to_use = classifier_qza_path
                is_packaged_classifier = True
        else:
                # 3. Fallback: Download/cache (if cli.py logic failed)
                print("    -> No classifier provided. Checking for default classifier...")
                classifier_obj_to_use = _get_default_classifier()
                is_packaged_classifier = False

    except (RuntimeError, IOError, FileNotFoundError) as e:
        print(f"\n[CLASSIFIER ERROR] Failed to get classifier: {e}", file=sys.stderr)
        raise RuntimeError("Failed to resolve classifier path.") from e

    # Import sequences to .qza if needed
    rep_seqs_qza_to_use = None
    if seq_format == 'qza':
        print("  -> Sequence format is .qza, proceeding.")
        rep_seqs_qza_to_use = rep_seqs_path
    else:
        # seq_format is'fasta', needs to import
        print(f"     -> Input is .{seq_format}, importing to .qza for QIIME2...")

        # Defines a new path for the imported file
        classify_dir_temp = output_dir / "taxonomy" # Temp directory
        classify_dir_temp.mkdir(parents=True, exist_ok=True)
        imported_qza = classify_dir_temp / "imported_rep_seqs.qza"

        cmd_import = [
                "qiime", "tools", "import",
                "--type", "FeatureData[Sequence]",
                "--input-path", str(rep_seqs_path),
                "--output-path", str(imported_qza)
        ]

        try:
                run_command("qiime", cmd_import, check=True)
        except subprocess.CalledProcessError as e:
                print(f"   [ERROR] QIIME2 tools import failed: {e.stderr}", file=sys.stderr)
                raise RuntimeError("Failed to import .fna to .qza") from e

        if not imported_qza.exists():
                raise ValidationError("QIIME2 import created no output file.")

        rep_seqs_qza_to_use = imported_qza
        print(f"     -> Successfully imported to {imported_qza}")

    # Define pathways
    classify_dir = output_dir / "taxonomy"
    classify_dir.mkdir(parents=True, exist_ok=True)

    classified_qza = classify_dir / "taxonomy.qza"
    export_dir = classify_dir / "exported_taxonomy"
    final_taxonomy_tsv = classify_dir / "taxonomy.tsv"

    # step 1: run classify-sklearn
    print("    -> Running QIIME2 classify-sklearn (Bokulich et al., 2018)...")

    try:
        # Handle both Path and Traversable objects
        if is_packaged_classifier:
            # It's a Traversable, use as_file context
            with importlib.resources.as_file(classifier_obj_to_use) as classifier_real_path:
                cmd_classify = [
                        "qiime", "feature-classifier", "classify-sklearn",
                        "--i-reads", str(rep_seqs_qza_to_use),
                        "--i-classifier", str(classifier_real_path),
                        "--o-classification", str(classified_qza),
                        "--p-n-jobs", str(threads)
                    ]
                run_command("qiime", cmd_classify, check=True)
        else:
                # It's just a regular Path, use it directly
                cmd_classify = [
                    "qiime", "feature-classifier", "classify-sklearn",
                    "--i-reads", str(rep_seqs_qza_to_use),
                    "--i-classifier", str(classifier_obj_to_use),
                    "--o-classification", str(classified_qza),
                    "--p-n-jobs", str(threads)
                ]
                run_command("qiime", cmd_classify, check=True)

    except subprocess.CalledProcessError as e:
        print(f"     [ERROR] QIIME2 classify-sklearn failed: {e.stderr}", file=sys.stderr)
        raise RuntimeError("Taxonomic classification failed.") from e

    if not classified_qza.exists():
        raise ValidationError(f"QIIME2 failed to create {classified_qza}")

    # step 2: exports .qza to .tsv
    print(f"   -> Exporting taxonomy to {export_dir}...")
    cmd_export = [
        "qiime", "tools", "export",
        "--input-path", str(classified_qza),
        "--output-path", str(export_dir)
    ]

    try:
        run_command("qiime", cmd_export, check=True)
    except subprocess.CalledProcessError as e:
        print(f"     [ERROR] QIIME2 tools export failed: {e.stderr}", file=sys.stderr)
        raise RuntimeError("Taxonomy export failed.") from e

    exported_tsv = export_dir / "taxonomy.tsv"
    if not exported_tsv.exists():
        raise ValidationError(f"QIIME2 export failed to create {exported_tsv}")

    # step 3: fix the header of the exported taxonomy file
    try:
        with open(exported_tsv, 'r') as f_in:
                lines = f_in.readlines()

        # Substitues the first line in the header
        lines[0] = "#OTUID\ttaxonomy\tconfidence\n"

        with open(final_taxonomy_tsv, 'w') as f_out:
                f_out.writelines(lines)

    except (OSError, IOError) as e:
        print(f"     [ERROR] Failed to fix taxonomy header: {e}", file=sys.stderr)
        raise RuntimeError("Header fix failed.") from e

    print(f"   -> Taxonomy file ready: {final_taxonomy_tsv}")
    return final_taxonomy_tsv