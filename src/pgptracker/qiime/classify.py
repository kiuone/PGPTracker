"""
QIIME2 Taxonomic Classification Wrapper for PGPTracker.

This module wraps the QIIME2 feature-classifier (Bokulich et al., 2018)
to assign taxonomy to representative sequences.
"""

import subprocess
from pathlib import Path
import importlib.resources
from pgptracker.utils.env_manager import run_command
from pgptracker.utils.validator import ValidationError # Usaremos para validação de output

def classify_taxonomy(
    rep_seqs_path: Path,
    seq_format: str,
    classifier_qza: Path,
    output_dir: Path,
    threads: int
) -> Path:
    
    """
    Runs QIIME2 'classify-sklearn', exports, and fixes the header.

    Args:
        rep_seqs_path: Path to representative sequences (.qza) from the user.
        seq_format: The format of the sequences ('qza' or 'fasta').
        classifier_qza: Path to the QIIME2 classifier artifact (.qza).
        output_dir: The main output directory for this run.
        threads: Number of threads to use.

    Returns:
        Path: The path to the final, header-fixed 'taxonomy.tsv' file.
        
    Raises:
        ValidationError: If any step fails to produce the expected output.
    """

    # Import sequences to .qza if needed
    rep_seqs_qza_to_use = None
    if seq_format == 'qza':
        print("  -> Sequence format is .qza, proceeding.")
        rep_seqs_qza_to_use = rep_seqs_path
    else:
        # seq_format is'fasta', needs to import 
        print(f"  -> Input is .{seq_format}, importing to .qza for QIIME2...")

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
            print(f"  [ERROR] QIIME2 tools import failed: {e.stderr}")
            raise RuntimeError("Failed to import .fna to .qza") from e

        if not imported_qza.exists():
            raise ValidationError("QIIME2 import created no output file.")

        rep_seqs_qza_to_use = imported_qza
        print(f"  -> Successfully imported to {imported_qza}")

    # Define pathways
    classify_dir = output_dir / "taxonomy"
    classify_dir.mkdir(parents=True, exist_ok=True)
    
    classified_qza = classify_dir / "taxonomy.qza"
    export_dir = classify_dir / "exported_taxonomy"
    final_taxonomy_tsv = classify_dir / "taxonomy.tsv" 

    # step 1: run classify-sklearn
    print("  -> Running QIIME2 classify-sklearn (Bokulich et al., 2018)...")
    cmd_classify = [
        "qiime", "feature-classifier", "classify-sklearn",
        "--i-reads", str(rep_seqs_qza_to_use),
        "--i-classifier", str(classifier_qza),
        "--o-classification", str(classified_qza),
        "--p-n-jobs", str(threads)
    ]
    
    try:
        run_command("qiime", cmd_classify, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] QIIME2 classify-sklearn failed: {e.stderr}")
        raise RuntimeError("Taxonomic classification failed.") from e

    if not classified_qza.exists():
        raise ValidationError(f"QIIME2 failed to create {classified_qza}")

    # step 2: exports .qza to .tsv
    print(f"  -> Exporting taxonomy to {export_dir}...")
    cmd_export = [
        "qiime", "tools", "export",
        "--input-path", str(classified_qza),
        "--output-path", str(export_dir)
    ]
    
    try:
        run_command("qiime", cmd_export, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] QIIME2 tools export failed: {e.stderr}")
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
            
    except Exception as e:
        print(f"  [ERROR] Failed to fix taxonomy header: {e}")
        raise RuntimeError("Header fix failed.") from e

    print(f"  -> Taxonomy file ready: {final_taxonomy_tsv}")
    return final_taxonomy_tsv