"""
QIIME2 export functions for PGPTracker.

This module handles exporting .qza files to .fna (FASTA) and .biom formats
using QIIME2 tools export command.
"""

from pathlib import Path
from typing import Dict
import shutil

from pgptracker.utils.env_manager import run_command


def export_qza_files(inputs: Dict[str, any], output_dir: Path) -> Dict[str, Path]:
    """
    Exports .qza files to .fna and .biom formats if needed.
    
    If inputs are already in .fna/.biom format, copies them to export directory
    for consistent pipeline structure.
    
    Args:
        inputs: Dictionary from validate_inputs() containing:
            - 'sequences': Path to sequences (.qza or .fna)
            - 'table': Path to feature table (.qza or .biom)
            - 'seq_format': 'qza' or 'fasta'
            - 'table_format': 'qza' or 'biom'
        output_dir: Base output directory (will create exports/ subdirectory)
        
    Returns:
        dict: Dictionary with exported file paths:
            {
                'sequences': Path to .fna file,
                'table': Path to .biom file
            }
            
    Raises:
        RuntimeError: If QIIME2 export fails.
        FileNotFoundError: If exported files are not found.
    """
    # Create exports subdirectory
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Exporting input files...")
    
    # Export or copy sequences
    if inputs['seq_format'] == 'qza':
        print("  -> Exporting sequences from .qza...")
        sequences_path = _export_sequences_qza(
            inputs['sequences'],
            exports_dir
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
        table_path = _export_table_qza(
            inputs['table'],
            exports_dir
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


def _export_sequences_qza(qza_path: Path, output_dir: Path) -> Path:
    """
    Exports representative sequences from .qza to .fna format.
    
    Args:
        qza_path: Path to rep-seqs.qza file
        output_dir: Directory where to export
        
    Returns:
        Path: Path to exported .fna file
        
    Raises:
        RuntimeError: If export fails
        FileNotFoundError: If exported file not found
    """
    # QIIME2 exports to a subdirectory with fixed name
    export_subdir = output_dir / "exported_sequences"
    export_subdir.mkdir(parents=True, exist_ok=True)
    
    # Run QIIME2 export
    cmd = [
        "qiime", "tools", "export",
        "--input-path", str(qza_path),
        "--output-path", str(export_subdir)
    ]
    
    try:
        run_command("qiime", cmd, check=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to export sequences from .qza: {e}"
        )
    
    # QIIME2 exports sequences as dna-sequences.fasta
    exported_file = export_subdir / "dna-sequences.fasta"
    
    if not exported_file.exists():
        raise FileNotFoundError(
            f"Expected exported file not found: {exported_file}\n"
            f"QIIME2 export may have failed silently"
        )
    
    # Move to consistent naming
    final_path = output_dir / "dna-sequences.fna"
    shutil.move(str(exported_file), str(final_path))
    
    # Clean up export subdirectory
    shutil.rmtree(export_subdir)
    
    return final_path


def _export_table_qza(qza_path: Path, output_dir: Path) -> Path:
    """
    Exports feature table from .qza to .biom format.
    
    Args:
        qza_path: Path to feature-table.qza file
        output_dir: Directory where to export
        
    Returns:
        Path: Path to exported .biom file
        
    Raises:
        RuntimeError: If export fails
        FileNotFoundError: If exported file not found
    """
    # QIIME2 exports to a subdirectory with fixed name
    export_subdir = output_dir / "exported_table"
    export_subdir.mkdir(parents=True, exist_ok=True)
    
    # Run QIIME2 export
    cmd = [
        "qiime", "tools", "export",
        "--input-path", str(qza_path),
        "--output-path", str(export_subdir)
    ]
    
    try:
        run_command("qiime", cmd, check=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to export feature table from .qza: {e}"
        )
    
    # QIIME2 exports table as feature-table.biom
    exported_file = export_subdir / "feature-table.biom"
    
    if not exported_file.exists():
        raise FileNotFoundError(
            f"Expected exported file not found: {exported_file}\n"
            f"QIIME2 export may have failed silently"
        )
    
    # Move to parent directory
    final_path = output_dir / "feature-table.biom"
    shutil.move(str(exported_file), str(final_path))
    
    # Clean up export subdirectory
    shutil.rmtree(export_subdir)
    
    return final_path


def _copy_file(src: Path, dst: Path) -> Path:
    """
    Copies a file to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        Path: Destination path
        
    Raises:
        RuntimeError: If copy fails
    """
    try:
        shutil.copy2(str(src), str(dst))
        return dst
    except Exception as e:
        raise RuntimeError(
            f"Failed to copy file from {src} to {dst}: {e}"
        )