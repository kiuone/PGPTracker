"""
Input file validators for PGPTracker CLI.

This module provides a single validation function that checks all inputs
at once and provides clear error messages.
"""

from pathlib import Path
from typing import Dict, List


def _validate_file(filepath: str, file_type: str, valid_extensions: List[str]) -> List[str]:
    """
    Helper function to validate a single file.
    
    Args:
        filepath: Path to file
        file_type: Description for error messages (e.g., "Sequences")
        valid_extensions: List of valid extensions
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    path = Path(filepath)
    
    if not path.exists():
        errors.append(f"{file_type} file not found: {filepath}")
    elif not path.is_file():
        errors.append(f"{file_type} path is not a file: {filepath}")
    elif path.stat().st_size == 0:
        errors.append(f"{file_type} file is empty: {filepath}")
    elif path.suffix not in valid_extensions:
        ext_list = ", ".join(valid_extensions)
        errors.append(
            f"Invalid {file_type.lower()} format: {path.suffix}\n"
            f"  Expected: {ext_list}"
        )
    
    return errors


def validate_inputs(
    rep_seqs: str,
    feature_table: str,
    output_dir: str
) -> Dict[str, any]:
    """
    Validates all input files and output directory.
    
    Checks:
    - Files exist and are not empty
    - File extensions are valid
    - Format compatibility between sequences and table (.qza + .qza OR .fasta + .biom)
    - Output directory can be created
    
    Args:
        rep_seqs: Path to representative sequences (.qza or .fna/.fasta/.fa)
        feature_table: Path to feature table (.qza or .biom)
        output_dir: Path to output directory
        
    Returns:
        dict: Dictionary containing validated paths and detected formats:
            {
                'sequences': Path object,
                'table': Path object,
                'output': Path object,
                'seq_format': 'qza' or 'fasta',
                'table_format': 'qza' or 'biom'
            }
            
    Raises:
        ValueError: If any validation fails, with all error messages combined.
    """
    errors = []
    
    # Validate sequences file
    errors.extend(_validate_file(
        rep_seqs,
        "Sequences",
        ['.qza', '.fna', '.fasta', '.fa']
    ))
    
    # Validate feature table file
    errors.extend(_validate_file(
        feature_table,
        "Feature table",
        ['.qza', '.biom']
    ))
    
    # Check format compatibility (only if both files exist)
    seq_path = Path(rep_seqs)
    table_path = Path(feature_table)
    
    if seq_path.exists() and table_path.exists():
        seq_is_qza = seq_path.suffix == '.qza'
        table_is_qza = table_path.suffix == '.qza'
        
        # Both must be .qza OR both must NOT be .qza
        if seq_is_qza != table_is_qza:
            errors.append(
                "Format mismatch: sequences and table must both be .qza OR both be non-.qza\n"
                f"  Sequences: {seq_path.suffix}\n"
                f"  Table: {table_path.suffix}\n"
                f"  Valid combinations: (.qza + .qza) OR (.fna/.fasta + .biom)"
            )
    
    # Validate output directory
    out_path = Path(output_dir)
    if out_path.exists() and not out_path.is_dir():
        errors.append(
            f"Output path exists but is not a directory: {output_dir}"
        )
    
    # Raise all errors at once
    if errors:
        error_msg = "Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    # Create output directory if it doesn't exist
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory: {output_dir}")
    
    # Return validated inputs with detected formats
    return {
        'sequences': seq_path,
        'table': table_path,
        'output': out_path,
        'seq_format': 'qza' if seq_path.suffix == '.qza' else 'fasta',
        'table_format': 'qza' if table_path.suffix == '.qza' else 'biom'
    }