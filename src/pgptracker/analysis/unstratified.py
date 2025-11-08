#!/usr/bin/env python3
"""
Generates unstratified PGPT tables (PGPT x Sample) and provides
database loading utilities for the PGPTracker pipeline.

Author: Vivian Mello
"""

import polars as pl
from pathlib import Path
import importlib.resources

def load_pathways_db(pgpt_level: str) -> pl.DataFrame:
    """
    Loads the bundled PLaBA pathways database from within the package
    and extracts the KO-to-PGPT mapping based on the specified level.
    
    Args:
        pgpt_level: The hierarchical level to use (e.g., 'Lv3', 'Lv4').
    
    Returns:
        A Polars DataFrame with ['KO', 'PGPT_ID'] columns.
    """
    db_filename = "pathways_plabase.txt"
    
    try:
        # 1. Find the bundled file path
        with importlib.resources.as_file(
            importlib.resources.files("pgptracker.databases").joinpath(db_filename)
        ) as p:
            db_path = p
        
        if not db_path.exists():
            raise FileNotFoundError 
            
    except Exception as e:
        print(f"  [ERROR] Critical: Bundled database file '{db_filename}' not found.")
        print("          Ensure 'src/pgptracker/databases/' contains the file and 'setup.py' includes it.")
        raise RuntimeError(f"Failed to load bundled database: {db_filename}") from e

    # 2. Load and process the file using Polars
    df = pl.read_csv(db_path, separator='\t', has_header=True)

    if pgpt_level not in df.columns:
        valid_levels = [c for c in df.columns if c.startswith("Lv")]
        raise ValueError(f"PGPT level '{pgpt_level}' not found in database. Available: {valid_levels}")
    
    # Extract KO using RegEx (KXXXXX at the end of the string)
    df = df.with_columns(
        pl.col('PGPT_ID').str.extract(r"-(K\d{5})$", 1).alias('KO_raw')
    )
    
    # Add 'ko:' prefix
    df = df.with_columns(
        pl.when(pl.col('KO_raw').is_not_null())
        .then(pl.lit('ko:') + pl.col('KO_raw'))
        .otherwise(None)
        .alias('KO')
    ).drop('KO_raw')
    
    # Filter null KOs and select final columns
    # df = df.filter(pl.col('KO').is_not_null()).select(['KO', 'PGPT_ID']).unique()

    df = df.filter(
        pl.col('KO').is_not_null() & pl.col(pgpt_level).is_not_null()
    ).select(['KO', pgpt_level]).unique()
    
    print(f"  -> Found {len(df)} unique KO-to-PGPT mappings at level '{pgpt_level}'.")
    return df

def generate_unstratified_pgpt(
    unstrat_ko_path: Path,
    output_dir: Path,
    pgpt_level: str
) -> Path:
    """
    Generates the unstratified PGPT abundance table (PGPT x Sample).
    This is the "black box" analysis.
    
    Args:
        unstrat_ko_path: Path to 'pred_metagenome_unstrat.tsv.gz'
        output_dir: Directory to save the output file.
        pgpt_level: The hierarchical level to use (e.g., 'Lv3').
        
    Returns:
        Path to the generated unstratified abundance table.
    """
    if not unstrat_ko_path.exists():
        raise FileNotFoundError(f"Unstratified KO file not found: {unstrat_ko_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load KOs (KO x Sample)
    df_ko = pl.read_csv(
        unstrat_ko_path,
        separator='\t',
        has_header=True,
        comment_prefix='#'
    )
    # Rename ID column (usually 'function' or '#OTU ID')
    df_ko = df_ko.rename({df_ko.columns[0]: 'KO'})
    
    # 2. Load Pathways (KO -> PGPT)
    df_pathways = load_pathways_db(pgpt_level=pgpt_level)
    
    # 3. Melt KO table (Wide -> Long)
    sample_cols = [c for c in df_ko.columns if c != 'KO']
    ko_long = df_ko.unpivot(
        index='KO',
        on=sample_cols,
        variable_name='Sample',
        value_name='Abundance'
    )
    ko_long = ko_long.filter(pl.col('Abundance') > 0)
    
    # 4. Join (KO x Sample) with (KO -> PGPT)
    joined = ko_long.join(df_pathways, on='KO', how='inner')
    
    # 5. Aggregate (PGPT x Sample)
    pgpt_abun = joined.group_by([pgpt_level, 'Sample']).agg(
        pl.col('Abundance').sum().alias('Total_Abundance')
    )
    
    # 6. Pivot (Long -> Wide) for final table
    pgpt_wide = pgpt_abun.pivot(
        values='Total_Abundance',
        index=pgpt_level,
        on='Sample',
        aggregate_function='sum'
    ).fill_null(0.0)

    output_path = output_dir / f"unstratified_pgpt_{pgpt_level}_abundances.tsv"

    try:
        snippet_df = pl.read_csv(output_path, separator='\t', n_rows=3)
        
        print("\n--- Output Preview: First 3 rows ---")
        with pl.Config(
            set_fmt_str_lengths=25,
            tbl_width_chars=160,
            tbl_rows=3,
            tbl_cols=4,
            tbl_hide_dataframe_shape=True,
            tbl_hide_column_data_types=True
        ):
            print(snippet_df)
    except Exception as e:
        print(f"  [Warning] Could not display output preview: {e}")
    
    # 7. Save
    pgpt_wide.write_csv(output_path, separator='\t')
    
    return output_path