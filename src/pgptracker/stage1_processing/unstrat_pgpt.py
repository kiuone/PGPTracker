#!/usr/bin/env python3
"""
Generates unstratified PGPT tables (PGPT x Sample) and provides
database loading utilities for the PGPTracker pipeline.

Author: Vivian Mello
"""

import polars as pl
from pathlib import Path
import time
import importlib.resources
from pgptracker.utils.profiling_tools.profiler import profile_memory


@profile_memory
def load_pathways_db(pgpt_level: str) -> pl.DataFrame:
    """
Loads and processes the bundled PLaBA pathways database.

    This function finds the 'pathways_plabase.txt' file bundled inside the 
    package, loads it, and performs critical cleanup and transformation:
    
    1.  Uses RegEx to extract the 'KXXXXX' ID from the 
        end of the 'PGPT_ID' string.
    2.  Harmonizes the ID by adding the 'ko:' prefix (e.g., 'K02584' -> 'ko:K02584')
        to match the KO prediction file format.
    3.  Filters out any rows that failed the RegEx (KO is null) or have 
        no PGPT assigned at the specified level (pgpt_level is null).
    4.  Selects only the 'KO' column and the specific 'pgpt_level' requested.
    5.  Removes duplicates to create a clean, 2-column (KO -> PGPT) lookup map.

    Args:
        pgpt_level: The hierarchical level to use (e.g., 'Lv3', 'Lv4').

    Returns:
        A Polars DataFrame (lookup map) with two columns: ['KO', pgpt_level].
    
    Example:
        If `pgpt_level='Lv3'`, this function transforms the raw file:

        | PGPT_ID                 | Lv3                    | ... (other columns) |
        | :---                    | :---                   | :--- |
        | ...-K02584              | NITROGEN_ACQUISITION   | ... |
        | ...-K02585              | NITROGEN_ACQUISITION   | ... |
        | ...-K02586              | NITROGEN_ACQUISITION   | ... |
        | ...-K02586              | NITROGEN_ACQUISITION   | ... |
        | PGPT_NO_KO_ID           | SOME_PGPT              | ... |
        | ...-K00000              | Null                   | ... |
        
        Into this final, clean DataFrame:
        
        | KO        | Lv3                    |
        | :---      | :---                   |
        | ko:K02584 | NITROGEN_ACQUISITION   |
        | ko:K02585 | NITROGEN_ACQUISITION   |
        | ko:K02586 | NITROGEN_ACQUISITION   |
    """
    db_filename = "pathways_plabase.txt"

    # 1. Find the bundled file path
    with importlib.resources.as_file(
        importlib.resources.files("pgptracker.databases").joinpath(db_filename)
    ) as p:
        db_path = p

    # 2. Load and process the file using Polars
    df = pl.read_csv(db_path, separator='\t', has_header=True)

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
    df = df.filter(
        pl.col('KO').is_not_null() & pl.col(pgpt_level).is_not_null()
    ).select(['KO', pgpt_level]).unique()
    
    print(f"  \n  -> Found {len(df)} unique KO-to-PGPT mappings at level '{pgpt_level}'.")
    return df


@profile_memory
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
    start_time = time.time()

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
    
    # Sort by pgpt_level before saving
    pgpt_wide = pgpt_wide.sort(pgpt_level)

    # 7.Update running totals for the final log message
    total_rows = pgpt_wide.height
    total_columns = pgpt_wide.width
    
    # 8. Save
    output_path = output_dir / f"unstratified_pgpt_{pgpt_level}_abundances.tsv"
    pgpt_wide.write_csv(output_path, separator='\t')

    # Log completion
    elapsed = time.time() - start_time
    print(f"  -> Export complete: {total_rows:,} rows × {total_columns} columns processed in: ({elapsed:.1f}s)")

    # 8. Print out a pretty table in the terminal for the user
    snippet_df = pl.read_csv(output_path, separator='\t', n_rows=3).sort(pgpt_level).head(3)
    
    print("\n--- Output Preview: First 3 rows ---")
    with pl.Config(
        set_fmt_str_lengths=25,
        tbl_width_chars=180,
        tbl_cols=4,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True
    ):
        print(snippet_df)
    
    return output_path