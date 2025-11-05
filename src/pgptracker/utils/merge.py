"""
Table merging module for PGPTracker.

Handles the complex chain of unzipping, BIOM conversion, and metadata merging
"""
import subprocess
import gzip
import shutil
from pathlib import Path
from pgptracker.utils.env_manager import run_command
from pgptracker.utils.validator import ValidationError

def merge_taxonomy_to_table(
    seqtab_norm_gz: Path,
    taxonomy_tsv: Path,
    output_dir: Path,
    save_intermediates: bool = False
) -> Path:
    
    """
    Runs the BIOM conversion and merging pipeline.

    1. Unzips 'seqtab_norm.tsv.gz'
    2. Converts 'seqtab_norm.tsv' -> 'seqtab_norm.biom'
    3. Merges 'taxonomy.tsv' -> 'feature_table_with_taxonomy.biom'
    4. Converts '...with_taxonomy.biom' -> 'feature_table_with_taxonomy.tsv'
    
    Args:
        seqtab_norm_gz: Path to the normalized table from PICRUSt2.
        taxonomy_tsv: Path to the processed taxonomy file from QIIME2.
        output_dir: The main output directory for this run.
        save_intermediates: If True, keeps the 'merge_work' directory.

    Returns:
        Path: The path to the final 'feature_table_with_taxonomy.tsv'.
    
    Raises:
        RuntimeError: If any biom command fails.
        ValidationError: If intermediate files are not created.
    """
    
    # Define caminhos
    work_dir = output_dir / "merge_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    seqtab_tsv = work_dir / "seqtab_norm.tsv"
    seqtab_biom = work_dir / "seqtab_norm.biom"
    merged_biom = work_dir / "feature_table_with_taxonomy.biom"
    final_tsv = output_dir / "feature_table_with_taxonomy.tsv" # Final file

    try:
        # Step 1: Unzip 'seqtab_norm.tsv.gz' (o 'gunzip -c')
        print("  -> Unzipping normalized table...")
        with gzip.open(seqtab_norm_gz, 'rb') as f_in:
            with open(seqtab_tsv, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        if not seqtab_tsv.exists() or seqtab_tsv.stat().st_size == 0:
            raise ValidationError("Failed to unzip seqtab_norm.tsv.gz")

        # Step 2: Converts TSV -> BIOM (o 'biom convert') 
        print("  -> Converting normalized table to BIOM format...")
        cmd_convert1 = [
            "biom", "convert",
            "-i", str(seqtab_tsv),
            "-o", str(seqtab_biom),
            "--table-type=OTU table",
            "--to-hdf5"
        ]
        # 'biom' is in 'pgptracker' environment
        run_command("PGPTracker", cmd_convert1, check=True)
        if not seqtab_biom.exists():
            raise ValidationError("biom convert (step 1) failed")

        # Step 3: Adds metadata (o 'biom add-metadata') ---
        print("  -> Merging taxonomy into BIOM table...")
        cmd_merge = [
            "biom", "add-metadata",
            "-i", str(seqtab_biom),
            "-o", str(merged_biom),
            "--observation-metadata-fp", str(taxonomy_tsv),
            "--sc-separated", "taxonomy"
        ]
        run_command("PGPTracker", cmd_merge, check=True)
        if not merged_biom.exists():
            raise ValidationError("biom add-metadata failed")

        # Step 4: Converts BIOM -> TSV (o 'biom convert' final) ---
        print("  -> Converting final BIOM table to TSV...")
        cmd_convert2 = [
            "biom", "convert",
            "-i", str(merged_biom),
            "-o", str(final_tsv),
            "--to-tsv"
        ]
        run_command("PGPTracker", cmd_convert2, check=True)
        if not final_tsv.exists() or final_tsv.stat().st_size == 0:
            raise ValidationError("biom convert (step 2) failed")

    except (subprocess.CalledProcessError, ValidationError, gzip.BadGzipFile) as e:
        print(f"  [ERROR] BIOM merging pipeline failed: {e}")
        raise RuntimeError("BIOM merging pipeline failed.") from e
    
    # Cleanup
    if not save_intermediates:
        print("  -> Cleaning up intermediate merge files...")
        shutil.rmtree(work_dir)

    print(f"  -> Final merged table ready: {final_tsv}")
    return final_tsv