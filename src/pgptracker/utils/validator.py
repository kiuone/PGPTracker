"""
Input file validators for PGPTracker CLI.

This module provides centralized validation logic for input files,
checking existence, format compatibility, and output verification.
"""

from pathlib import Path
from typing import Dict, List, Any, Union
import polars as pl

class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

def _validate_file(path: Path, file_type: str, valid_extensions: List[str]) -> List[str]:
    """
    Helper function to validate a single file's existence and extension.
    """
    errors = []
    
    if not path.exists():
        errors.append(f"{file_type} file not found: {path}")
    elif not path.is_file():
        errors.append(f"{file_type} path is not a file: {path}")
    elif path.stat().st_size == 0:
        errors.append(f"{file_type} file is empty: {path}")
    elif path.suffix not in valid_extensions:
        ext_list = ", ".join(valid_extensions)
        errors.append(
            f"Invalid {file_type.lower()} format: {path.suffix}\n"
            f" Expected: {ext_list}"
        )
    
    return errors

def validate_output_file(
    path: Path,
    tool_name: str,
    file_description: str
) -> None:
    """
    Validates that a tool's output file exists and is not empty.
    
    Args:
        path: Path to the output file.
        tool_name: Name of the tool (e.g., "PICRUSt2").
        file_description: Description (e.g., "phylogenetic tree").
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        RuntimeError: If file is empty.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{tool_name} did not create {file_description}: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"{tool_name} created empty {file_description}: {path}"
        )

def validate_inputs(
    rep_seqs: str,
    feature_table: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Validates all pipeline input files and prepares the output directory.
    
    Checks:
    - Existence and non-empty status.
    - Valid extensions (.qza, .fna, .biom).
    - Format compatibility (sequences and table must match ecosystem).
    
    Returns:
        dict: Validated paths and detected formats.
    """
    errors = []

    seq_path = Path(rep_seqs)
    table_path = Path(feature_table)
    out_path = Path(output_dir)
    
    # 1. Individual File Validation
    errors.extend(_validate_file(
        seq_path,
        "Sequences",
        ['.qza', '.fna', '.fasta', '.fa']
    ))
    
    errors.extend(_validate_file(
        table_path,
        "Feature table",
        ['.qza', '.biom']
    ))
    
    # 2. Cross-Validation (Format Compatibility)
    if seq_path.exists() and table_path.exists():
        seq_is_qza = seq_path.suffix == '.qza'
        table_is_qza = table_path.suffix == '.qza'
        
        # Rule: Both must be .qza OR both must be standard formats
        if seq_is_qza != table_is_qza:
            errors.append(
                "Format mismatch: inputs must be consistent.\n"
                f"  Sequences: {seq_path.suffix}\n"
                f"  Table: {table_path.suffix}\n"
                "  Valid pairs: (.qza + .qza) OR (.fna/.fasta + .biom)"
            )
    
    if errors:
        error_msg = "Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValidationError(error_msg)
    
    # 3. Output Preparation
    out_path.mkdir(parents=True, exist_ok=True)

    return {
        'sequences': seq_path,
        'table': table_path,
        'output': out_path,
        'seq_format': 'qza' if seq_path.suffix == '.qza' else 'fasta',
        'table_format': 'qza' if table_path.suffix == '.qza' else 'biom'
    }

def find_asv_column(df: Union[pl.DataFrame, pl.LazyFrame]) -> str:
    # Candidates for ASV ID column names (common in QIIME2/PICRUSt2)
    ASV_ID_CANDIDATES = ['OTU/ASV_ID', 'ASV_ID', 'OTU_ID', '#OTU ID', 'sequence', 'feature-id', 'Feature ID']
    
    # Handle both DataFrame and LazyFrame
    if isinstance(df, pl.LazyFrame):
        cols = df.collect_schema().names()
    else:
        cols = df.columns

    asv_col = next((c for c in ASV_ID_CANDIDATES if c in cols), None)
    
    if asv_col is None:
        raise ValueError(f"ASV column not found. Expected one of: {ASV_ID_CANDIDATES}")
        
    return asv_col