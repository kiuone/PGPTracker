"""
Unit tests for stratified analysis functions in pgptracker.analysis.stratify module.

These tests cover loading input files, aggregation logic, and the main stratified analysis function.
Run with: pytest tests/unit/test_stratify.py -v
"""
import pytest
import polars as pl
from pathlib import Path
import gzip
from unittest.mock import patch, MagicMock

from pgptracker.analysis.stratify import (
    load_seqtab_with_taxonomy,
    load_ko_predicted,
    aggregate_by_tax_level_sample,
    aggregate_by_tax_level_ko,
    generate_stratified_analysis
)
# We mock load_pathways_db as it's tested in test_unstratified.py
from pgptracker.analysis.unstratified import load_pathways_db 

@pytest.fixture
def mock_merged_table_file(tmp_path):
    """Mock 'norm_wt_feature_table.tsv' file."""
    p = tmp_path / "merged_table.tsv"
    content = (
        "OTU/ASV_ID\tSampleA\tSampleB\tKingdom\tPhylum\tGenus\tSpecies\n"
        "ASV_1\t100\t50\tBacteria\tProteobacteria\tPseudomonas\tfluorescens\n"
        "ASV_2\t20\t0\tBacteria\tProteobacteria\tPseudomonas\taeruginosa\n"
        "ASV_3\t0\t80\tBacteria\tFirmicutes\tBacillus\tsubtilis\n"
        "ASV_4\t30\t30\tBacteria\tFirmicutes\tBacillus\tNA\n"
    )
    p.write_text(content)
    return p

@pytest.fixture
def mock_ko_pred_file(tmp_path):
    """Mock 'KO_predicted.tsv.gz' file."""
    p = tmp_path / "ko_pred.tsv.gz"
    content = (
        "sequence\tko:K00001\tko:K00002\tmetadata_NSTI\n"
        "ASV_1\t1.0\t0.5\t0.1\n"
        "ASV_2\t1.2\t0.0\t0.2\n"
        "ASV_3\t0.0\t2.0\t0.3\n"
        "ASV_4\t0.8\t2.2\t0.4\n"
    )
    with gzip.open(p, 'wt') as f:
        f.write(content)
    return p

@pytest.fixture
def mock_pathways_df():
    """Mock pathways DataFrame."""
    return pl.DataFrame({
        'KO': ['ko:K00001', 'ko:K00002'],
        'PGPT_ID': ['PGPT_A', 'PGPT_B']
    })

# Test load_seqtab_with_taxonomy
def test_load_seqtab_success(mock_merged_table_file):
    df = load_seqtab_with_taxonomy(mock_merged_table_file, tax_level="Genus")
    assert 'OTU/ASV_ID' in df.columns
    assert 'Genus' in df.columns
    assert 'SampleA' in df.columns
    assert 'Phylum' not in df.columns # Check that only requested level is kept

def test_load_seqtab_invalid_tax_level(mock_merged_table_file):
    with pytest.raises(ValueError, match="Taxonomic level 'Order' not found"):
        load_seqtab_with_taxonomy(mock_merged_table_file, tax_level="Order")

# Test load_ko_predicted
def test_load_ko_predicted_success(mock_ko_pred_file):
    df = load_ko_predicted(mock_ko_pred_file)
    assert 'OTU/ASV_ID' in df.columns
    assert 'ko:K00001' in df.columns
    assert 'metadata_NSTI' not in df.columns

# Test aggregate_by_tax_level_sample
def test_agg_by_sample_genus_level(mock_merged_table_file):
    df_seqtab = load_seqtab_with_taxonomy(mock_merged_table_file, tax_level="Genus")
    df_agg = aggregate_by_tax_level_sample(df_seqtab, tax_level="Genus")
    
    # Check aggregation
    # Pseudomonas, SampleA = 100 (ASV_1) + 20 (ASV_2) = 120
    # Bacillus, SampleB = 80 (ASV_3) + 30 (ASV_4) = 110
    assert len(df_agg) == 4 # Pseudo/A, Pseudo/B, Baci/A, Baci/B
    
    pseudo_a = df_agg.filter((pl.col("Genus") == "Pseudomonas") & (pl.col("Sample") == "SampleA"))
    assert pseudo_a["Total_Tax_Abundance"][0] == 120
    
    baci_b = df_agg.filter((pl.col("Genus") == "Bacillus") & (pl.col("Sample") == "SampleB"))
    assert baci_b["Total_Tax_Abundance"][0] == 110

def test_agg_by_sample_phylum_level(mock_merged_table_file):
    df_seqtab = load_seqtab_with_taxonomy(mock_merged_table_file, tax_level="Phylum")
    df_agg = aggregate_by_tax_level_sample(df_seqtab, tax_level="Phylum")

    # Check aggregation
    # Proteobacteria, SampleA = 100 (ASV_1) + 20 (ASV_2) = 120
    # Firmicutes, SampleB = 80 (ASV_3) + 30 (ASV_4) = 110
    proteo_a = df_agg.filter((pl.col("Phylum") == "Proteobacteria") & (pl.col("Sample") == "SampleA"))
    assert proteo_a["Total_Tax_Abundance"][0] == 120
    
    firmi_b = df_agg.filter((pl.col("Phylum") == "Firmicutes") & (pl.col("Sample") == "SampleB"))
    assert firmi_b["Total_Tax_Abundance"][0] == 110

# Test aggregate_by_tax_level_ko
def test_agg_by_ko_genus_level(mock_ko_pred_file, mock_merged_table_file):
    df_ko = load_ko_predicted(mock_ko_pred_file)
    df_seqtab = load_seqtab_with_taxonomy(mock_merged_table_file, tax_level="Genus")
    df_agg = aggregate_by_tax_level_ko(df_ko, df_seqtab, tax_level="Genus")
    
    # Check aggregation (MEAN)
    # Pseudomonas, K00001 = mean(1.0, 1.2) = 1.1
    # Bacillus, K00002 = mean(2.0, 2.2) = 2.1
    
    pseudo_k1 = df_agg.filter((pl.col("Genus") == "Pseudomonas") & (pl.col("KO") == "ko:K00001"))
    assert pseudo_k1["Avg_Copy_Number"][0] == 1.1
    
    baci_k2 = df_agg.filter((pl.col("Genus") == "Bacillus") & (pl.col("KO") == "ko:K00002"))
    assert baci_k2["Avg_Copy_Number"][0] == 2.1
    
# Test main function (integration)
@patch('pgptracker.analysis.stratify.load_pathways_db')
def test_generate_stratified_analysis_e2e(
    mock_load_pathways,
    mock_merged_table_file,
    mock_ko_pred_file,
    mock_pathways_df,
    tmp_path
):
    mock_load_pathways.return_value = mock_pathways_df
    output_dir = tmp_path / "output"
    
    result_path = generate_stratified_analysis(
        merged_table_path=mock_merged_table_file,
        ko_predicted_path=mock_ko_pred_file,
        output_dir=output_dir,
        taxonomic_level="Genus",
        pgpt_level="Lv3",
    )
    
    assert result_path.exists()
    assert result_path.name == "genus_stratified_pgpt.tsv.gz"
    
    # Check final content
    df = pl.read_csv(result_path, separator='\t')
    assert len(df) == 8 # 2 Genera x 2 PGPTs x 2 Samples
    
    # Check one value
    # Genus: Pseudomonas, PGPT: PGPT_A (K00001), Sample: SampleA
    # Total_Tax_Abundance = 120 (from test_agg_by_sample_genus_level)
    # Avg_Copy_Number = 1.1 (from test_agg_by_ko_genus_level)
    # Functional_Abundance = 120 * 1.1 = 132.0
    
    check = df.filter(
        (pl.col("Genus") == "Pseudomonas") &
        (pl.col("PGPT_ID") == "PGPT_A") &
        (pl.col("Sample") == "SampleA")
    )
    assert check["Total_PGPT_Abundance"][0] == 132.0