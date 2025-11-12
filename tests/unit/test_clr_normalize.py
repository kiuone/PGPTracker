#!/usr/bin/env python3
"""
Unit tests for CLR normalization in PGPTracker.
Updated for apply_clr returning a dictionary of outputs.

Tests compositional data transformation properties:
1. CLR sum per sample = 0 (geometric mean centering)
2. Zeros are replaced before transformation
3. Wide and long formats handled correctly

Author: Vivian Mello

runs with: pytest tests/unit/test_clr_normalize.py -v
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pgptracker.stage2_analysis.clr_normalize import apply_clr


class TestCLRWideFormat:
    """Tests for 'wide' (unstratified) format."""

    def test_clr_wide_basic(self):
        """Test CLR on simple wide format data."""
        df = pl.DataFrame({
            'PGPT_ID': ['NITROGEN_FIX', 'PHOSPHATE_SOL', 'SIDEROPHORE'],
            'Sample_A': [10.0, 20.0, 30.0],
            'Sample_B': [5.0, 15.0, 25.0]})
        
        outputs = apply_clr(df, format='wide')
        
        # Check if the correct key is returned
        assert 'unstratified_clr' in outputs
        result = outputs['unstratified_clr']
        
        # Check structure preserved
        assert result.columns == df.columns
        assert len(result) == len(df)
        assert result['PGPT_ID'].to_list() == df['PGPT_ID'].to_list()
    
    def test_clr_wide_sum_zero(self):
        """CLR values per sample must sum to ~0 (geometric mean centering)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2', 'PGPT_3'],
            'Sample_A': [10.0, 20.0, 30.0],
            'Sample_B': [100.0, 200.0, 300.0]
        })
        
        outputs = apply_clr(df, format='wide')
        result = outputs['unstratified_clr']
        
        # Sum of CLR values per sample should be ~0
        for col in ['Sample_A', 'Sample_B']:
            col_sum = result[col].sum()
            assert abs(col_sum) < 1e-10, f"{col} sum = {col_sum}, expected ~0"
    
    def test_clr_wide_handles_zeros(self):
        """CLR should handle zeros through multiplicative replacement."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2', 'PGPT_3'],
            'Sample_A': [10.0, 0.0, 30.0],  # Zero in PGPT_2
            'Sample_B': [5.0, 15.0, 0.0]    # Zero in PGPT_3
        })
        
        outputs = apply_clr(df, format='wide')
        result = outputs['unstratified_clr']
        
        # No NaN or inf values should exist
        for col in ['Sample_A', 'Sample_B']:
            assert not result[col].is_nan().any()
            assert not result[col].is_infinite().any()
        
        # CLR sum still ~0
        for col in ['Sample_A', 'Sample_B']:
            assert abs(result[col].sum()) < 1e-10
    
    def test_clr_wide_negative_positive_values(self):
        """CLR produces both negative and positive values (log-ratios)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2', 'PGPT_3'],
            'Sample_A': [10.0, 20.0, 30.0]
        })
        
        outputs = apply_clr(df, format='wide')
        result = outputs['unstratified_clr']
        
        values = result['Sample_A'].to_list()
        assert any(v < 0 for v in values), "Should have negative values"
        assert any(v > 0 for v in values), "Should have positive values"


class TestCLRLongFormat:
    """Tests for 'long' (stratified) format and its dual outputs."""
    
    def test_clr_long_returns_both_outputs(self):
        """Test CLR on stratified long format data."""
        df = pl.DataFrame({
            'Order': ['Bacillales', 'Bacillales', 'Pseudomonadales', 'Pseudomonadales'],
            'Lv3': ['NITROGEN_FIX', 'PHOSPHATE_SOL', 'NITROGEN_FIX', 'PHOSPHATE_SOL'],
            'Sample': ['Sample_A', 'Sample_A', 'Sample_A', 'Sample_A'],
            'Total_PGPT_Abundance': [10.0, 20.0, 30.0, 40.0]
        })
        
        outputs = apply_clr(df, format='long')
        
        # Check if both keys are returned
        assert 'stratified_wide_clr' in outputs
        assert 'stratified_long_clr' in outputs
        
        result_wide = outputs['stratified_wide_clr']
        result_long = outputs['stratified_long_clr']
        
        # Check wide structure
        assert 'Sample_A' in result_wide.columns
        assert 'Order' in result_wide.columns
        assert 'Lv3' in result_wide.columns
        assert len(result_wide) == 4

        # Check long structure
        assert 'CLR_Total_PGPT_Abundance' in result_long.columns
        assert 'Total_PGPT_Abundance' not in result_long.columns
        assert 'Sample' in result_long.columns
        assert len(result_long) == 4
    
    def test_clr_long_sum_zero_per_sample(self):
        """CLR values per sample must sum to ~0 in both outputs."""
        df = pl.DataFrame({
            'Genus': ['Bacillus', 'Pseudomonas', 'Bacillus', 'Pseudomonas'],
            'Lv3': ['NITROGEN_FIX', 'NITROGEN_FIX', 'PHOSPHATE_SOL', 'PHOSPHATE_SOL'],
            'Sample': ['Sample_A', 'Sample_A', 'Sample_A', 'Sample_A'],
            'Total_PGPT_Abundance': [10.0, 20.0, 15.0, 25.0]
        })
        
        outputs = apply_clr(df, format='long')
        result_wide = outputs['stratified_wide_clr']
        result_long = outputs['stratified_long_clr']
        
        # Test long output sum (group by sample)
        for sample in result_long['Sample'].unique():
            sample_data = result_long.filter(pl.col('Sample') == sample)
            clr_sum = sample_data['CLR_Total_PGPT_Abundance'].sum()
            assert abs(clr_sum) < 1e-10, f"Long format {sample} sum = {clr_sum}"

        # Test wide output sum (sum sample column)
        for col in ['Sample_A']:
            clr_sum = result_wide[col].sum()
            assert abs(clr_sum) < 1e-10, f"Wide format {col} sum = {clr_sum}"
    
    def test_clr_long_handles_zeros(self):
        """CLR should handle zeros in long format (in both outputs)."""
        df = pl.DataFrame({
            'Genus': ['Bacillus', 'Pseudomonas'],
            'Lv3': ['NITROGEN_FIX', 'NITROGEN_FIX'],
            'Sample': ['Sample_A', 'Sample_A'],
            'Total_PGPT_Abundance': [10.0, 0.0]  # Zero abundance
        })
        
        outputs = apply_clr(df, format='long')
        result_wide = outputs['stratified_wide_clr']
        result_long = outputs['stratified_long_clr']
        
        # Test long output for NaN/inf
        assert not result_long['CLR_Total_PGPT_Abundance'].is_nan().any()
        assert not result_long['CLR_Total_PGPT_Abundance'].is_infinite().any()

        # Test wide output for NaN/inf
        assert not result_wide['Sample_A'].is_nan().any()
        assert not result_wide['Sample_A'].is_infinite().any()
    
    def test_clr_long_multiple_samples(self):
        """Test CLR across multiple samples (the previously failing test)."""
        df = pl.DataFrame({
            'Genus': ['Bacillus', 'Bacillus', 'Pseudomonas', 'Pseudomonas'],
            'Lv3': ['NITROGEN_FIX', 'NITROGEN_FIX', 'NITROGEN_FIX', 'NITROGEN_FIX'],
            'Sample': ['Sample_A', 'Sample_B', 'Sample_A', 'Sample_B'],
            'Total_PGPT_Abundance': [10.0, 15.0, 20.0, 25.0]
        })
        
        outputs = apply_clr(df, format='long')
        result_wide = outputs['stratified_wide_clr']
        result_long = outputs['stratified_long_clr']
        
        # Test long output (group by)
        for sample in ['Sample_A', 'Sample_B']:
            sample_data = result_long.filter(pl.col('Sample') == sample)
            clr_sum = sample_data['CLR_Total_PGPT_Abundance'].sum()
            assert abs(clr_sum) < 1e-10, f"Long format {sample} sum failed"

        # Test wide output (column sum)
        for sample in ['Sample_A', 'Sample_B']:
            clr_sum = result_wide[sample].sum()
            assert abs(clr_sum) < 1e-10, f"Wide format {sample} sum failed"


class TestCLREdgeCases:
    """Edge cases and error conditions (mostly for wide format)."""
    
    def test_all_zeros_in_sample(self):
        """Handle sample with all zeros (CLR should be 0)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2'],
            'Sample_A': [0.0, 0.0],  # All zeros
            'Sample_B': [10.0, 20.0]
        })
        
        outputs = apply_clr(df, format='wide')
        result = outputs['unstratified_clr']
        
        # Should not crash and produce valid CLR values
        assert not result['Sample_A'].is_nan().any()
        assert not result['Sample_B'].is_nan().any()
        
        # All-zero sample CLR sum must be 0
        assert abs(result['Sample_A'].sum()) < 1e-10
        assert abs(result['Sample_B'].sum()) < 1e-10

    def test_single_feature(self):
        """Handle single feature (CLR should be 0)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1'],
            'Sample_A': [100.0]
        })
        
        outputs = apply_clr(df, format='wide')
        result = outputs['unstratified_clr']
        
        # Single feature: CLR = log(x / geom_mean) = log(x / x) = 0
        assert abs(result['Sample_A'][0]) < 1e-10
    
    def test_preserves_feature_order(self):
        """Feature order should be preserved in wide format."""
        df = pl.DataFrame({
            'PGPT_ID': ['Z_PGPT', 'A_PGPT', 'M_PGPT'],
            'Sample_A': [10.0, 20.0, 30.0]
        })
        
        outputs = apply_clr(df, format='wide')
        result = outputs['unstratified_clr']
        
        assert result['PGPT_ID'].to_list() == ['Z_PGPT', 'A_PGPT', 'M_PGPT']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])