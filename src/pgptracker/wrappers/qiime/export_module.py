"""
QIIME2 export functions for PGPTracker.

This module handles exporting .qza files to .fna (FASTA) and .biom formats
using QIIME2 tools export command.
"""

from pathlib import Path
from typing import Dict, Any
import shutil
import subprocess 
from pgptracker.utils.env_manager import run_command


def export_qza_files(inputs: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    """
    Exports .qza files to .fna and .biom formats if needed.
    
    If inputs are already in .fna/.biom format, copies them to export directory
    for consistent pipeline structure.
    ...
    """
    # Create exports subdirectory
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Exporting input files...")
    
    # Export or copy sequences
    if inputs['seq_format'] == 'qza':
        print("  -> Exporting sequences from .qza...")
        sequences_path = _run_export(
            qza_path=inputs['sequences'],
            export_dir=exports_dir,
            temp_subdir_name="exported_sequences",
            expected_filename="dna-sequences.fasta", 
            final_filename="dna-sequences.fna"    
        )
    else:
        print("  -> Copying sequences (.fna already in correct format)...")
        sequences_path = _copy_file(
            inputs['sequences'],
            exports_dir / "dna-sequences.fna"
        )
    
    # Export or copy feature table
    if inputs['table_format'] == 'qza':
        print("  -> Exporting feature table from .qza...")
        table_path = _run_export(
            qza_path=inputs['table'],
            export_dir=exports_dir,
            temp_subdir_name="exported_table",
            expected_filename="feature-table.biom", 
            final_filename="feature-table.biom"    
        )
    else:
        print("  -> Copying feature table (.biom already in correct format)...")
        table_path = _copy_file(
            inputs['table'],
            exports_dir / "feature-table.biom"
        )
    
    print("Export completed successfully")
    print(f"  Sequences: {sequences_path}")
    print(f"  Table: {table_path}")
    
    return {
        'sequences': sequences_path,
        'table': table_path
    }


def _run_export(
    qza_path: Path,
    export_dir: Path,
    temp_subdir_name: str,
    expected_filename: str,
    final_filename: str
) -> Path:
    """
    GENERIC HELPER: Runs 'qiime tools export', checks, and renames the file.
    
    Args:
        qza_path: Path to the input .qza file
        export_dir: The PARENT 'exports' directory
        temp_subdir_name: Name of the temporary dir to create (e.g., "exported_seqs")
        expected_filename: The file QIIME2 creates (e.g., "dna-sequences.fasta")
        final_filename: The name you want the file to have (e.g., "dna-sequences.fna")
        
    Returns:
        Path to the final, renamed file.
        
    Raises:
        subprocess.CalledProcessError: If the 'qiime tools export' command fails.
        FileNotFoundError: If QIIME2 runs but does not create the expected file.
    """
    # QIIME2 exports to a subdirectory
    temp_export_path = export_dir / temp_subdir_name
    
    # Clean up any existing temp export dir
    if temp_export_path.exists():
        shutil.rmtree(temp_export_path)
    temp_export_path.mkdir(parents=True, exist_ok=True)
    
    # Run QIIME2 export
    cmd = [
        "qiime", "tools", "export",
        "--input-path", str(qza_path),
        "--output-path", str(temp_export_path)
    ]

    run_command("qiime", cmd, check=True)
    
    # Verifies if the expected file exists
    exported_file = temp_export_path / expected_filename
    
    if not exported_file.exists():
        raise FileNotFoundError(
            f"Expected exported file not found: {exported_file}\n"
            f"QIIME2 export may have failed silently"
        )
    
    # Move and rename to final location
    final_path = export_dir / final_filename
    shutil.move(str(exported_file), str(final_path))
    
    # Cleans up temporary export directory
    shutil.rmtree(temp_export_path)
    
    return final_path


def _copy_file(src: Path, dst: Path) -> Path:
    """
    Copies a file to destination.
    ...
    Raises:
        RuntimeError: If copy fails
    """
    try:
        shutil.copy2(str(src), str(dst))
        return dst
    except (shutil.Error, IOError) as e: 
        raise RuntimeError(
            f"Failed to copy file from {src} to {dst}: {e}"
        )