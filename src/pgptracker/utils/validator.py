"""
Input file validators for PGPTracker CLI.

This module provides a single validation function that checks all inputs
at once and provides clear error messages.
"""

class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

from pathlib import Path
from typing import Dict, List, Any

def _validate_file(path: Path, file_type: str, valid_extensions: List[str]) -> List[str]:
    """
    Helper function to validate a single file.
    
    Args:
        path: Path object to the file
        file_type: Description for error messages (e.g., "Sequences")
        valid_extensions: List of valid extensions
        
    Returns:
        List of error messages (empty if valid)
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
    path: Path to the output file
    tool_name: Name of the tool that created the file (for error messages)
    file_description: Description of the file (e.g., "phylogenetic tree")

    Raises:
    FileNotFoundError: If file doesn't exist
    RuntimeError: If file is empty
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

    seq_path = Path(rep_seqs)
    table_path = Path(feature_table)
    out_path = Path(output_dir)
    
    # Validate sequences file
    errors.extend(_validate_file(
        seq_path,
        "Sequences",
        ['.qza', '.fna', '.fasta', '.fa']
    ))
    
    # Validate feature table file
    errors.extend(_validate_file(
        table_path,
        "Feature table",
        ['.qza', '.biom']
    ))
    
    # Check format compatibility (only if both files exist)
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
    
    # Raise all errors at once
    if errors:
        error_msg = "Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValidationError(error_msg)
    
    # Create output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)

    # Return validated inputs with detected formats
    return {
        'sequences': seq_path,
        'table': table_path,
        'output': out_path,
        'seq_format': 'qza' if seq_path.suffix == '.qza' else 'fasta',
        'table_format': 'qza' if table_path.suffix == '.qza' else 'biom'
    }

def find_asv_column(df):
    ASV_ID_CANDIDATES = ['OTU/ASV_ID', 'ASV_ID', 'OTU_ID', '#OTU ID', 'sequence']
    df_cols = df.collect_schema().names()
    asv_col = next((c for c in ASV_ID_CANDIDATES if c in df_cols), None)
    if asv_col is None:
        raise ValueError(f"ASV column not found. Expected: {ASV_ID_CANDIDATES}")
    return asv_col