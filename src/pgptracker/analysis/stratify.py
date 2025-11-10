#!/usr/bin/env python3
"""
Generates taxonomically-stratified functional analysis for PGPTracker.

This module implements the stratified logic:
1. Aggregates ASV abundances by a user-defined taxonomic level and sample.
2. Aggregates KO copy numbers by the same taxonomic level.
3. Joins the two aggregated tables in batches to calculate functional contribution.
4. Maps KOs to PGPTs and aggregates to the final stratified profile.

Author: Vivian Mello
"""

import polars as pl
import gzip
import gc
import io
import time
from pathlib import Path
from typing import Optional, List, Tuple
from pgptracker.analysis.unstratified import load_pathways_db
from pgptracker.utils.profiler import profile_memory
from pgptracker.utils.validator import find_asv_column

TAXONOMY_COLS = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

def identify_sample_columns(
    ftable: pl.DataFrame,
    sample_prefix: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Identifies sample abundance columns in a feature table.
    
    IMPORTANT: Feature table should contain ONLY:
      - ASV identifier column (e.g., 'OTU/ASV_ID', 'ASV_ID')
      - Taxonomy columns (Kingdom, Phylum, ..., Species)
      - Sample abundance columns
    
    Args:
        ftable: Feature table DataFrame
        sample_prefix: If provided, only columns starting with this are samples
        exclude_cols: If provided, these columns are NOT samples and NOT taxonomy or asv_col
    
    Returns:
        List of sample column names
        Default: assume everything is a sample, except TAXONOMY_COLS and asv_col
    
    Raises:
        ValueError: If no sample columns found
    
    Example:
        Input columns: ['ASV_ID', 'Genus', 'Sample_A', 'Sample_B', 'depth']
        
        With sample_prefix='Sample_' -> ['Sample_A', 'Sample_B']
        With exclude_cols=['depth']  -> ['Sample_A', 'Sample_B']
        With neither -> ['Sample_A', 'Sample_B', 'depth'] (warns user)
    """

    asv_col = find_asv_column(ftable)

    # Define known taxonomy and asv_id columns
    asv_and_tax_cols = [c for c in TAXONOMY_COLS if c in ftable.columns] + [asv_col]
    
    # Identify sample columns based on user input
    if sample_prefix:
        sample_cols = [c for c in ftable.columns if c.startswith(sample_prefix)]
    
    elif exclude_cols:
        sample_cols = [c for c in ftable.columns if c not in asv_and_tax_cols and c not in exclude_cols]
    
    else:
        # Default: assume everything else is a sample
        sample_cols = [c for c in ftable.columns if c not in asv_and_tax_cols]
        if sample_cols:
            print(f"  -> [Warning] Assuming all non-taxonomy columns are samples ({len(sample_cols)} found)")
            print(f"     If incorrect, use --sample-prefix or --exclude-cols")
        else:
            raise ValueError("No sample columns identified. Check your feature table.")
    
    return sample_cols

@profile_memory
def aggregate_by_tax_level_sample(
    ftable: pl.DataFrame, 
    tax_level: str,
    sample_cols: List[str],
    keep_unclassified_samples: bool = False
    ) -> pl.DataFrame:
    """
    Aggregates ASV abundances by taxonomic level and sample.
    
    This is Step 1 of the stratified approach: reduce the feature table
    from individual ASVs to taxonomic-level summaries before joining.
    
    Args:
        ftable: Feature table with ASV abundances
        tax_level: Taxonomic column to aggregate by (e.g., 'Genus')
        sample_cols: List of sample column names
        keep_unclassified_samples: If True, rows with 'Null' taxonomy
            are renamed to 'Unclassified_[tax_level]'. 
            If False (default), these rows are filtered out.
    
    Returns:
        DataFrame with columns [tax_level, Sample, Total_Tax_Abundance]
    
    Example:
        Input (ftable):
            OTU/ASV_ID  Genus         Sample_A  Sample_B
            ASV_001     Pseudomonas   10        5
            ASV_002     Pseudomonas   15        0
            ASV_003     Bacillus      0         20
        
        Output (tax_level='Genus'):
            Genus         Sample    Total_Tax_Abundance
            Pseudomonas   Sample_A  25
            Pseudomonas   Sample_B  5
            Bacillus      Sample_B  20
    """
    asv_col = find_asv_column(ftable)
    
    # Transform: Wide (ASV × Sample) -> Long (ASV-Sample pairs)
    ftable_long = ftable.unpivot(
        index=[asv_col, tax_level],
        on=sample_cols,
        variable_name='Sample',
        value_name='Abundance'
    )
    
    # Filter out zeros (reduce data volume)
    ftable_long = ftable_long.filter(pl.col('Abundance') > 0)
    
    # Aggregate: Sum by tax_level and sample
    tax_abun = ftable_long.group_by([tax_level, 'Sample']).agg(
        pl.col('Abundance').sum().alias('Total_Tax_Abundance')
    )

    # Polars see 'Null' as a tax_level type
    # Pseudomonas, Null and Bacillus would all be in 'Genus', for example.
    if keep_unclassified_samples:
        # If True: Rename Nulls to 'unclassified_...'
        tax_abun = tax_abun.with_columns(
            pl.col(tax_level).fill_null(f'Unclassified_{tax_level}'))
    else:
        # If False (default): Delete Nulls.
        tax_abun = tax_abun.filter(pl.col(tax_level).is_not_null())
    
    print(f"  -> Step 1 Result: {len(tax_abun)} '{tax_level}'-Sample pairs")
    return tax_abun

@profile_memory
def aggregate_by_tax_level_ko(
    ko_predicted: pl.DataFrame,
    tax_abun: pl.DataFrame,
    tax_level: str
) -> pl.DataFrame:
    """
        Aggregates KO copy numbers by taxonomic level.
    
    This is Step 2 of the Santo Graal approach: calculate average KO copy
    numbers per taxonomic group before joining with abundance data.
    
    Args:
        ko_predicted: KO predictions per ASV
        tax_abun: Feature table (used only to get taxonomy mapping)
        tax_level: Taxonomic column to aggregate by
    
    Returns:
        DataFrame with columns [tax_level, KO, Avg_Copy_Number]
    
    Example:
        Input (ko_predicted):
            OTU/ASV_ID  ko:K00001  ko:K00002
            ASV_001     2.5        1.0
            ASV_002     1.5        0.0
            ASV_003     0.0        3.0
        
        Input (tax_abun, for taxonomy):
            OTU/ASV_ID  Genus
            ASV_001     Pseudomonas
            ASV_002     Pseudomonas
            ASV_003     Bacillus
        
        Output (tax_level='Genus'):
            OTU/ASV_ID      Genus         KO         Avg_Copy_Number
            ASV_001         Pseudomonas   ko:K00001  2.0
            ASV_002         Pseudomonas   ko:K00002  1.0
            ASV_003         Bacillus      ko:K00002  3.0
    """
    asv_col = find_asv_column(ko_predicted)
    
    # 1. Get taxonomy map (ASV_ID -> tax_level)
    tax_map = tax_abun.select([asv_col, tax_level]).unique()
    
    # Get KO columns
    ko_cols = [c for c in ko_predicted.columns if c.startswith('ko:')]
    
    # 2. Transform: Wide (ASV × KO) -> Long (ASV-KO pairs)
    ko_long = ko_predicted.unpivot(
        index=asv_col,
        on=ko_cols,
        variable_name='KO',
        value_name='Copy_Number'
    )
    
    # Filter out zeros
    ko_long = ko_long.filter(pl.col('Copy_Number') > 0)
    
    # 3. Join with taxonomy map
    ko_with_tax = ko_long.join(tax_map, on=asv_col, how='inner')
    
    # 4.Aggregate: Mean copy number by (Taxon, KO)
    tax_ko = ko_with_tax.group_by([tax_level, 'KO']).agg(
        pl.col('Copy_Number').mean().alias('Avg_Copy_Number')
    )

    print(f"  -> Step 2 Result: {len(tax_ko)} '{tax_level}'-KO pairs")
    return tax_ko

@profile_memory
def join_and_calculate_batched(
    tax_abun: pl.DataFrame,
    tax_ko: pl.DataFrame,
    pathways: pl.DataFrame,
    output_path: Path,
    tax_level: str,
    pgpt_level:str,
    # batch_size: int 
) -> None:
    """
    Joins aggregated tables and calculates functional abundances by iterating
    over taxonomic groups to minimize memory usage.
    
    This is Step 3 of the stratified approach:
    1. For each taxon (e.g., each Genus):
       - Join small abundance table with small KO table
       - Calculate functional abundance = Abundance × Copy_Number
       - Map KOs to PGPTs and aggregate
    2. Write results incrementally to compressed output
    
    Args:
        tax_abun: Aggregated abundances (Taxon × Sample)
        tax_ko: Aggregated KO copy numbers (Taxon × KO)
        pathways: KO to PGPT mapping database
        output_path: Path for output file (tsv.gz)
        tax_level: Taxonomic level name (for output columns)
        pgpt_level: PGPT aggregation level ('Pathway' or 'Trait')
    
    Writes:
        Compressed TSV with columns [tax_level, pgpt_level, Sample, Total_PGPT_Abundance]
    
    Example flow for one taxon:
        tax_abun (Genus='Pseudomonas'):
            Genus         Sample    Total_Tax_Abundance
            Pseudomonas   Sample_A  25
        
        tax_ko (Genus='Pseudomonas'):
            Genus         KO         Avg_Copy_Number
            Pseudomonas   ko:K00001  2.0
        
        After join and KO->PGPT mapping:
            Genus         Pathway          Sample    Total_PGPT_Abundance
            Pseudomonas   nitrogen_fixing  Sample_A  50.0
    """
    print(f"\n  -> Starting Step 3: Join and Calculate (Optimized Group Iteration)...")
    start_time = time.time()
    
    # Pre-join: Map KOs to PGPTs (small operation)
    # This creates (Taxon × KO × PGPT) mapping
    ko_pgpt_map = tax_ko.join(pathways, on='KO', how='inner')
    
    # Group by taxon (creates iterator over groups)
    ko_pgpt_groups = ko_pgpt_map.group_by(tax_level, maintain_order=True)
    
    first_batch = True
    total_rows = 0
    total_columns = 0
    
    # Write results incrementally to compressed file
    with gzip.open(output_path, 'wb', compresslevel=3) as f_gzip:
        # Iterate above the groups (ex: ('Pseudomonas', df_pseudomonas), ('Bacillus', df_bacillus))
        with io.TextIOWrapper(f_gzip, encoding="utf-8") as f_text:
                
            # NOTE: group_by() always returns (key_tuple, df) where key_tuple is ALWAYS a tuple
            for i, (current_taxon_tuple, ko_pgpt_batch_df) in enumerate(ko_pgpt_groups):
                
                # Extract the scalar value from the tuple (e.g., ('GenusA',) -> 'GenusA')
                current_taxon = current_taxon_tuple[0] # Now ['GenusA'] correctly
                
                # print(f"-> Processing Group {i + 1}/{n_groups} ({current_taxon})", flush=True)
                
                # Filter abundance data for this taxon
                abun_batch = tax_abun.filter(pl.col(tax_level).is_in(current_taxon))

                joined = abun_batch.join(ko_pgpt_batch_df, on=tax_level, how='inner')
                
                # Calculate functional abundance
                joined = joined.with_columns(
                    (pl.col('Total_Tax_Abundance') * pl.col('Avg_Copy_Number')).alias('Functional_Abundance')
                )
                
                # Aggregate to final format: (Taxon x PGPT x Sample)
                result = joined.group_by([tax_level, pgpt_level, 'Sample']).agg(
                    pl.col('Functional_Abundance').sum().alias('Total_PGPT_Abundance')
                )

                # Process only if this batch (taxon) yielded results
                if not result.is_empty():
                    # 1. Write this batch (as TSV) directly to the gzipped file stream
                    result.write_csv(
                        f_text,  # Write directly to the open text wrapper
                        separator='\t',
                        include_header=first_batch)
                    f_text.flush() # Force write to disk
                    # 2. Toggle the header flag off after the first pass
                    if first_batch:
                        first_batch = False
                    # 3. Update running totals for the final log message
                    total_rows += len(result)
                    if total_columns == 0:
                        total_columns = len(result.columns)
                
                # Cleanup memory
                del abun_batch, ko_pgpt_batch_df, joined, result
                gc.collect()

    # Adds a verification in the case no data has been written
    if total_rows == 0:
            raise RuntimeError(
                f"No matching data found between taxonomic groups and PGPT mappings.\n"
                f"Check that your KO predictions and pathway database are compatible.")

    elapsed = time.time() - start_time
    print(f"  -> Export complete: {total_rows:,} rows × {total_columns} columns processed in: ({elapsed:.1f}s)")

@profile_memory
def generate_stratified_analysis(
    merged_table_path: Path,
    ko_predicted_path:Path,
    output_dir: Path,
    taxonomic_level: str,
    pgpt_level: str,
    sample_prefix: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None,
    keep_unclassified: bool = False
) -> Path:
    """
    Generates taxonomically-stratified PGPT abundance profiles.
    
    This orchestrates the complete Santo Graal workflow:
    1. Load and validate input data
    2. Aggregate ASV data by taxonomy BEFORE joining (memory efficient)
    3. Calculate functional abundances by taxonomic group
    4. Export stratified PGPT profiles
    
    Args:
        merged_table_path: Path to normalized feature table with taxonomy
        ko_predicted_path: Path to KO predictions per ASV
        output_dir: Directory for output files
        taxonomic_level: Taxonomic level to stratify by (e.g., 'Genus', 'Family')
        pgpt_level: PGPT aggregation level ('Pathway' or 'Trait')
        sample_prefix: Optional prefix to identify sample columns
        exclude_cols: Optional list of columns to exclude from samples
        keep_unclassified: If True, keeps ASVs with null taxonomy as 'Unclassified'
    
    Returns:
        Path to stratified PGPT abundance table (tsv.gz)
    
    Example workflow:
        Input: Feature table (ASV × Sample) + KO predictions (ASV × KO)
        Output: Stratified profile (Genus × PGPT × Sample)
        
        Input feature table:
            OTU/ASV_ID  Genus         Sample_A  Sample_B
            ASV_001     Pseudomonas   10        5
        
        Output stratified profile:
            Genus         Pathway          Sample    Total_PGPT_Abundance
            Pseudomonas   nitrogen_fixing  Sample_A  50.0
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{taxonomic_level.lower()}_stratified_pgpt.tsv.gz"
    
    print(f"Starting stratified analysis for level: '{taxonomic_level}'")
    
    # 1. Load data
    # Load feature table normalized with taxonomy (norm_wt_feature_table.tsv)
    ftable = pl.read_csv(merged_table_path, separator='\t', has_header=True, comment_prefix='#')

    # 2. Identify sample columns 
    sample_cols = identify_sample_columns(ftable, sample_prefix, exclude_cols)

    # 3. Load ko predictions per ASV ('KO_predicted.tsv.gz') 
    ko_df = pl.read_csv(ko_predicted_path, separator='\t', has_header=True,)
    # Drop metadata from KO_predicted.tsv.gz
    cols_to_drop = [c for c in ko_df.columns if c.startswith('metadata_') or c == 'closest_reference_genome']
    ko_df = ko_df.drop(cols_to_drop)
    
    # 4. Load PLaBAse that constains the pathways to link KO -> PGPT 
    pathways = load_pathways_db(pgpt_level=pgpt_level) # Uses the imported function from unstratified.py
    
    # 5. Aggregate (Reduce data BEFORE join)
    tax_abun = aggregate_by_tax_level_sample(ftable, taxonomic_level, sample_cols, keep_unclassified)
    tax_ko = aggregate_by_tax_level_ko(ko_df, tax_abun, taxonomic_level)
    
    # Clean up large dataframes
    del ftable, ko_df
    gc.collect()
    
    # 3. Join, Calculate, and Export in Batches
    join_and_calculate_batched(
        tax_abun,
        tax_ko,
        pathways,
        output_path,
        taxonomic_level,
        pgpt_level,)

    # Sort the output file
    print(f"  -> Sorting output by {taxonomic_level}, Sample...")
    df_sorted = pl.read_csv(output_path, separator='\t').sort([taxonomic_level, pgpt_level, 'Sample'])
    df_sorted.write_csv(output_path, separator='\t')

    # Display output preview (first 3 rows, all columns)
    snippet_df = pl.read_csv(output_path, separator='\t', n_rows=3
        ).sort([taxonomic_level, pgpt_level, 'Sample']).head(3)
        
    print("\n--- Output Preview: First 3 rows, All columns ---")
    with pl.Config(set_fmt_str_lengths=25, tbl_width_chars=160, tbl_cols=4,
        tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(snippet_df)
            
    print(f"Output saved to: {output_dir/output_path.name}")
    print(f"\nStratified analysis complete for '{taxonomic_level}'.")
    return output_path