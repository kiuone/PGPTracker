#!/usr/bin/env python3
"""
Generates taxonomically-stratified functional analysis for PGPTracker.

This module implements the "Santo Graal" logic:
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
from pathlib import Path
import time
from pgptracker.analysis.unstratified import load_pathways_db

def load_seqtab_with_taxonomy(path: Path, tax_level: str) -> pl.DataFrame:
    """
    Loads the normalized feature table with all taxonomy levels.
    (This is the output from 'merge.py')
    
    Args:
        path: Path to 'norm_wt_feature_table.tsv'
        tax_level: The taxonomic level to use (e.g., 'Genus', 'Family')
        
    Returns:
        DataFrame filtered to the necessary columns.
    
    Raises:
        FileNotFoundError: If table not found.
        ValueError: If tax_level column does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Merged feature table not found: {path}")
    
    print(f"  -> Loading feature table: {path.name}")
    
    # Scan to check header first
    try:
        all_columns = pl.read_csv(path, separator='\t', has_header=True, n_rows=0).columns
    except Exception as e:
        raise RuntimeError(f"Could not read header from {path}: {e}")

    # Validate tax_level
    if tax_level not in all_columns:
        valid_levels = [c for c in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] if c in all_columns]
        raise ValueError(
            f"Taxonomic level '{tax_level}' not found in table.\n"
            f"Available levels: {valid_levels}"
        )
        
    # Identify sample columns
    tax_cols = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    meta_cols = [c for c in tax_cols if c in all_columns] + ['OTU/ASV_ID'] # 'OTU/ASV_ID' is from your merge.py
    sample_cols = [c for c in all_columns if c not in meta_cols]
    
    if not sample_cols:
        raise ValueError("No sample columns found in feature table.")
    
    # Keep only necessary columns for this analysis
    keep_cols = ['OTU/ASV_ID', tax_level] + sample_cols
    
    # Load only the columns we need
    df = pl.read_csv(
        path,
        separator='\t',
        has_header=True,
        comment_prefix='#',
        columns=keep_cols
    )

    print(f"  -> Loaded: {len(df)} ASVs × {len(sample_cols)} samples")
    print(f"  -> Aggregation Level: {tax_level} ({df[tax_level].n_unique()} unique entries)")
    
    return df

def load_ko_predicted(path: Path) -> pl.DataFrame:
    """
    Loads KO predictions per ASV ('KO_predicted.tsv.gz').
    """
    if not path.exists():
        raise FileNotFoundError(f"KO predictions file not found: {path}")
    
    print(f"  -> Loading KO predictions: {path.name}")
    
    df = pl.read_csv(
        path,
        separator='\t',
        has_header=True,
        infer_schema_length=1000 # Keep this for speed
    )
    
    # Rename first column (e.g., 'sequence') to 'OTU/ASV_ID' to match seqtab
    first_col = df.columns[0]
    if first_col != 'OTU/ASV_ID':
        df = df.rename({first_col: 'OTU/ASV_ID'})
    
    # Drop metadata
    cols_to_drop = [c for c in df.columns if c.startswith('metadata_') or c == 'closest_reference_genome']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]  # Validate existence
    if cols_to_drop:
        df = df.drop(cols_to_drop)
    
    ko_cols = [c for c in df.columns if c.startswith('ko:')]
    if not ko_cols:
        raise ValueError("No KO columns found (expected format: 'ko:K00001')")

    print(f"  -> Loaded: {len(df)} ASVs × {len(ko_cols)} KOs")
    return df

def aggregate_by_tax_level_sample(seqtab: pl.DataFrame, tax_level: str) -> pl.DataFrame:
    """
    Step 1 (Aggregation): Aggregates ASV abundances by tax_level and sample.
    (e.g., "Total abundance of all 'Pseudomonas' in 'Sample_A'")
    """
    print(f"  -> Aggregating abundances by '{tax_level}' and Sample...")
    
    sample_cols = [c for c in seqtab.columns if c not in ['OTU/ASV_ID', tax_level]]
    
    # Melt (Wide -> Long)
    seqtab_long = seqtab.unpivot(
        index=['OTU/ASV_ID', tax_level],
        on=sample_cols,
        variable_name='Sample',
        value_name='Abundance'
    )
    
    # Filter zeros
    seqtab_long = seqtab_long.filter(pl.col('Abundance') > 0)
    
    # Aggregate by tax_level and sample (SUM)
    tax_abun = seqtab_long.group_by([tax_level, 'Sample']).agg(
        pl.col('Abundance').sum().alias('Total_Tax_Abundance')
    )
    
    # Filter out null tax levels (e.g., ASVs with no 'Genus' assigned)
    tax_abun = tax_abun.filter(pl.col(tax_level).is_not_null())
    
    print(f"  -> Step 1 Result: {len(tax_abun)} '{tax_level}'-Sample pairs")
    return tax_abun

def aggregate_by_tax_level_ko(
    ko_predicted: pl.DataFrame,
    seqtab: pl.DataFrame,
    tax_level: str
) -> pl.DataFrame:
    """
    Step 2 (Aggregation): Aggregates KO copy numbers by tax_level.
    (e.g., "Average copy number of 'K00001' across all 'Pseudomonas' ASVs")
    """
    print(f"  -> Aggregating KO copy numbers by '{tax_level}'...")
    
    # 1. Get taxonomy map (ASV_ID -> tax_level)
    taxonomy_map = seqtab.select(['OTU/ASV_ID', tax_level]).unique()
    
    ko_cols = [c for c in ko_predicted.columns if c.startswith('ko:')]
    
    # 2. Melt KO predictions (Wide -> Long)
    ko_long = ko_predicted.unpivot(
        index='OTU/ASV_ID',
        on=ko_cols,
        variable_name='KO',
        value_name='Copy_Number'
    )
    
    # Filter zeros
    ko_long = ko_long.filter(pl.col('Copy_Number') > 0)
    
    # 3. Join with taxonomy map
    ko_with_tax = ko_long.join(taxonomy_map, on='OTU/ASV_ID', how='inner')
    
    # 4. Aggregate by tax_level and KO (MEAN)
    tax_ko = ko_with_tax.group_by([tax_level, 'KO']).agg(
        pl.col('Copy_Number').mean().alias('Avg_Copy_Number')
    )
    
    # Filter out null tax levels
    tax_ko = tax_ko.filter(pl.col(tax_level).is_not_null())
    
    print(f"  -> Step 2 Result: {len(tax_ko)} '{tax_level}'-KO pairs")
    return tax_ko

# DEPOIS (Implementando sua otimização: iterar sobre os grupos)
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
    Step 3 (Join & Calculate): Joins aggregated tables by iterating over
    the largest dataframe's groups.
    """
    print(f"\n  -> Starting Step 3: Join and Calculate (Optimized Group Iteration)...")
    start_time = time.time()
    
    # Pre-join KOs and Pathways (Taxon x KO) JOIN (KO -> PGPT)
    ko_pgpt_map = tax_ko.join(pathways, on='KO', how='inner')
    n_groups = ko_pgpt_map.get_column(tax_level).n_unique()
    # Groupe the big dataframe (ko_pgpt_map) just once.
    # This creates an object "GroupBy" that we can iterate over.
    ko_pgpt_groups = ko_pgpt_map.group_by(tax_level, maintain_order=True)
    print(f"  -> Processing {n_groups} '{tax_level}' groups...")
    
    first_batch = True
    total_rows = 0
    total_columns = 0
    
    # Iterate above the groups (ex: ('Pseudomonas', df_pseudomonas), ('Bacillus', df_bacillus))
    try:
        with gzip.open(output_path, 'wb', compresslevel=3) as f_gzip:
            with io.TextIOWrapper(f_gzip, encoding="utf-8") as f_text:
                
                first_batch = True
                # NOTE: group_by() always returns (key_tuple, df) where key_tuple is ALWAYS a tuple
                for i, (current_taxon_tuple, ko_pgpt_batch_df) in enumerate(ko_pgpt_groups):
                    
                    # Extract the scalar value from the tuple (e.g., ('GenusA',) -> 'GenusA')
                    current_taxon = current_taxon_tuple[0]
                    batch_taxa = [current_taxon]  # Now ['GenusA'] correctly
                    
                    # print(f"    -> Processing Group {i + 1}/{n_groups} ({current_taxon})", flush=True)

                    abun_batch = tax_abun.filter(pl.col(tax_level).is_in(batch_taxa))

                    # If there is no abudance for this taxon, skip
                    if abun_batch.is_empty():
                        continue

                    # Santo Graal Join (Pequeno x Pequeno)
                    joined = abun_batch.join(ko_pgpt_batch_df, on=tax_level, how='inner')
                    
                    # Calculate functional abundance
                    joined = joined.with_columns(
                        (pl.col('Total_Tax_Abundance') * pl.col('Avg_Copy_Number')).alias('Functional_Abundance')
                    )
                    
                    # Aggregate (Taxon x PGPT x Sample)
                    result = joined.group_by([tax_level, pgpt_level, 'Sample']).agg(
                        pl.col('Functional_Abundance').sum().alias('Total_PGPT_Abundance')
                    )
                
                    if not result.is_empty():
                        # Write to string buffer first
                        string_buffer = io.StringIO()
                        result.write_csv(string_buffer, separator='\t', include_header=first_batch)
                        
                        # Get string and write to gzipped file
                        csv_string = string_buffer.getvalue()
                        f_text.write(csv_string)
                        f_text.flush()  # Force flush to ensure data is written
                        
                        if first_batch:
                            first_batch = False

                        total_rows += len(result)
                        if total_columns == 0:
                            total_columns = len(result.columns)
                    
                    # Cleanup
                    del abun_batch, ko_pgpt_batch_df, joined, result
                    gc.collect()
                    
    except Exception as e:
        raise RuntimeError(f"Failed during batch processing and writing: {e}")

    # Adds a verification in the case no data has been written
    if total_rows == 0:
        print(f"  -> WARNING: No data was written to the output file. (0 rows)")
        if first_batch:  # If first_batch is still True, nothing was written
             print("  -> No matching data found to process.")
             # Writes a blank file with only the data head
             try:
                 # Tries to write a blank datahead if nothing was done
                 if not output_path.exists() or output_path.stat().st_size == 0:
                      header_df = pl.DataFrame({
                          tax_level: [], pgpt_level: [], 'Sample': [], 'Total_PGPT_Abundance': []
                      }).with_columns(pl.all().cast(pl.String)) 
                      
                      with gzip.open(output_path, 'wb', compresslevel=3) as f_gzip:
                           with io.TextIOWrapper(f_gzip, encoding="utf-8") as f_text:
                                header_df.write_csv(f_text, separator='\t', include_header=True)
             except Exception:
                 pass 

    elapsed = time.time() - start_time
    print(f"  -> Export complete: {total_rows:,} rows × {total_columns} columns processed in: ({elapsed:.1f}s)")

def generate_stratified_analysis(
    merged_table_path: Path,
    ko_predicted_path: Path,
    output_dir: Path,
    taxonomic_level: str,
    pgpt_level: str,
    # batch_size: int
) -> Path:
    """
    Main orchestration function for stratified analysis.
    
    Args:
        merged_table_path: Path to 'norm_wt_feature_table.tsv'
        ko_predicted_path: Path to 'KO_predicted.tsv.gz'
        output_dir: Directory to save the output.
        taxonomic_level: String name of the column to stratify by (e.g., 'Genus').
        batch_size: Number of taxa to process in each batch.
        
    Returns:
        Path to the final stratified output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{taxonomic_level.lower()}_stratified_pgpt.tsv.gz"
    print(f"Starting stratified analysis for level: '{taxonomic_level}'")
    
    # 1. Load data
    seqtab = load_seqtab_with_taxonomy(merged_table_path, taxonomic_level)
    if seqtab[taxonomic_level].null_count() == len(seqtab):
        raise ValueError(
            f"All values in '{taxonomic_level}' are null. "
            f"Try a higher taxonomic level (e.g., 'Genus', 'Family')."
        )
    ko_predicted = load_ko_predicted(ko_predicted_path)
    pathways = load_pathways_db(pgpt_level=pgpt_level) # Uses the imported DRY function
    
    # 2. Aggregate (Reduce data BEFORE join)
    tax_abun = aggregate_by_tax_level_sample(seqtab, taxonomic_level)
    tax_ko = aggregate_by_tax_level_ko(ko_predicted, seqtab, taxonomic_level)
    
    # Clean up large dataframes
    del seqtab, ko_predicted
    gc.collect()
    
    # 3. Join, Calculate, and Export in Batches
    join_and_calculate_batched(
        tax_abun,
        tax_ko,
        pathways,
        output_path,
        taxonomic_level,
        pgpt_level,
        # batch_size
    )

    # Display output preview (first 3 rows, all columns)
    try:
        snippet_df = pl.read_csv(
            output_path,
            separator='\t',
            n_rows=3
        )
        
        print("\n--- Output Preview: First 3 rows, All columns ---")
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

    print(f"Output saved to: {output_dir/output_path.name}")
    print(f"\nStratified analysis complete for '{taxonomic_level}'.")
    return output_path