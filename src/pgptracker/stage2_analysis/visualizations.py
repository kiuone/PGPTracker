# src/pgptracker/stage2_analysis/visualizations.py

import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import textwrap

from pgptracker.stage2_analysis.plot_funcs import export_figure, setup_matplotlib_style

# Initialize style globally
setup_matplotlib_style()

def _wrap_labels(labels: List[str], width: int = 15) -> List[str]:
    """Helper to wrap long text labels into multiple lines."""
    return ['\n'.join(textwrap.wrap(str(l), width)) for l in labels]

def plot_ordination(
    df_scores: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_col: str,
    group_col: str,
    df_loadings: Optional[pl.DataFrame] = None,
    x_col: str = "PC1",
    y_col: str = "PC2",
    title: str = "Ordination Plot",
    output_dir: Path = Path("."),
    base_name: str = "ordination"
) -> None:
    """
    Plots 2D ordination results (PCA, PCoA, t-SNE).
    Supports Biplot if df_loadings is provided.
    """
    # Join scores with metadata if needed
    if group_col not in df_scores.columns:
        df_plot = df_scores.join(
            metadata.select([sample_col, group_col]), 
            on=sample_col, 
            how="inner"
        )
    else:
        df_plot = df_scores

    pdf = df_plot.to_pandas()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Main Scatter
    sns.scatterplot(
        data=pdf, 
        x=x_col, 
        y=y_col, 
        hue=group_col, 
        style=group_col,
        s=100, 
        alpha=0.8, 
        ax=ax
    )
    
    # Biplot Logic (Arrows)
    if df_loadings is not None and x_col == "PC1":
        # Calculate magnitude to pick top 5 features
        loadings = df_loadings.with_columns(
            (pl.col("PC1")**2 + pl.col("PC2")**2).alias("magnitude")
        ).sort("magnitude", descending=True).head(5)
        
        # Scale factor to make arrows visible against data points
        x_range = pdf[x_col].max() - pdf[x_col].min()
        y_range = pdf[y_col].max() - pdf[y_col].min()
        scale_x = x_range * 0.8
        scale_y = y_range * 0.8
        
        # Offset to avoid text overlap
        offsets = [(1.1, 1.1), (1.1, -0.1), (-0.1, 1.1), (1.2, 0.5), (0.5, 1.2)]
        
        for i, row in enumerate(loadings.iter_rows(named=True)):
            lx = row["PC1"] * scale_x
            ly = row["PC2"] * scale_y
            
            ax.arrow(
                0, 0, lx, ly, 
                color='r', alpha=0.6, 
                width=x_range*0.002, 
                head_width=x_range*0.03
            )
            
            # Clean feature name (remove pipe prefix if exists)
            feat_name = row["Feature"].split('|')[-1] if '|' in row["Feature"] else row["Feature"]
            
            # Use rotating offset to spread text
            off_x, off_y = offsets[i % len(offsets)]
            
            ax.text(
                lx * off_x, ly * off_y, 
                feat_name, 
                color='darkred', 
                fontsize=10, 
                fontweight='bold'
            )
    
    ax.set_title(title)
    # Place legend outside plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title=group_col)
    
    export_figure(fig, base_name, output_dir)

def plot_alpha_diversity(
    df_alpha: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_col: str,
    group_col: str,
    output_dir: Path
) -> None:
    """Generates boxplots for each Alpha Diversity metric."""
    df_plot = df_alpha.join(
        metadata.select([sample_col, group_col]),
        on=sample_col,
        how="inner"
    )
    
    pdf = df_plot.to_pandas()
    metrics = pdf['Metric'].unique()
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 7))
        subset = pdf[pdf['Metric'] == metric]
        
        sns.boxplot(
            data=subset, x=group_col, y='Value', 
            hue=group_col, palette="Set2", ax=ax, legend=False
        )
        sns.stripplot(
            data=subset, x=group_col, y='Value', 
            color='black', alpha=0.5, jitter=True, ax=ax
        )
        
        ax.set_title(f"Alpha Diversity: {metric}")
        ax.set_ylabel(metric)
        
        # Wrap X labels to prevent overlap
        labels = [item.get_text() for item in ax.get_xticklabels()]
        wrapped_labels = _wrap_labels(labels, width=12)
        ax.set_xticklabels(wrapped_labels, rotation=0)
        
        export_figure(fig, f"alpha_{metric}", output_dir)

def plot_feature_importance(
    df_importance: pl.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    output_dir: Path = Path("."),
    base_name: str = "feature_importance"
) -> None:
    """Plots top N important features from ML models."""
    val_col = 'Importance' if 'Importance' in df_importance.columns else 'Coefficient'
    
    df_top = (
        df_importance
        .with_columns(pl.col(val_col).abs().alias("abs_val"))
        .sort("abs_val", descending=True)
        .head(top_n)
    )
    
    pdf = df_top.to_pandas()
    
    # Dynamically adjust height based on number of features
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    
    sns.barplot(
        data=pdf, 
        x=val_col, 
        y='Feature', 
        hue='Feature',
        palette="viridis",
        legend=False,
        ax=ax
    )
    
    ax.set_title(f"{title} (Top {top_n})")
    ax.set_xlabel("Importance Score")
    
    export_figure(fig, base_name, output_dir)

def plot_volcano(
    df_stats: pl.DataFrame,
    p_val_col: str = "p_value",
    effect_col: str = "test_statistic",
    p_threshold: float = 0.05,
    output_dir: Path = Path("."),
    base_name: str = "volcano_plot"
) -> None:
    """Creates a Volcano-like plot (Effect Size vs -log10 p-value)."""
    # Filter out extreme p-values (0 or None) for log calculation
    df_clean = df_stats.filter(
        pl.col(p_val_col).is_not_null() & (pl.col(p_val_col) > 0)
    )
    
    df_plot = df_clean.with_columns([
        (-pl.col(p_val_col).log10()).alias("log_p"),
        (pl.col(p_val_col) < p_threshold).alias("Significant")
    ])
    
    pdf = df_plot.to_pandas()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        data=pdf,
        x=effect_col,
        y="log_p",
        hue="Significant",
        palette={True: "#e74c3c", False: "#95a5a6"},
        alpha=0.7,
        ax=ax
    )
    
    ax.axhline(
        -np.log10(p_threshold), 
        color='blue', 
        linestyle='--', 
        alpha=0.5, 
        label=f'p={p_threshold}'
    )
    
    ax.set_title("Feature Significance (Volcano Plot)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_xlabel("Test Statistic / Effect Size")
    ax.legend()
    
    export_figure(fig, base_name, output_dir)

def plot_heatmap(
    df_wide_N_D: pl.DataFrame,
    metadata: Optional[pl.DataFrame] = None,
    sample_col: str = "Sample",
    group_col: Optional[str] = None,
    top_n_features: int = 50,
    method: str = "ward",
    metric: str = "euclidean",
    output_dir: Path = Path("."),
    base_name: str = "heatmap_clustering"
) -> None:
    """
    Generates a clustered heatmap (samples x features).
    
    Args:
        df_wide_N_D: Abundance matrix (samples x features).
        metadata: Optional metadata to annotate samples.
        group_col: Metadata column to use for sample color bar.
        top_n_features: Use only top N features by variance.
    """
    # 1. Select numeric features
    feat_cols = [c for c in df_wide_N_D.columns if c != sample_col]
    
    # 2. Filter top N features by variance if too many
    if len(feat_cols) > top_n_features:
        vars = df_wide_N_D.select(feat_cols).var().transpose(include_header=True)
        top_feats = (
            vars.sort("column_0", descending=True)
            .head(top_n_features)["column"]
            .to_list()
        )
        df_subset = df_wide_N_D.select([sample_col] + top_feats)
    else:
        df_subset = df_wide_N_D

    # 3. Prepare Pandas DataFrame for Seaborn (Samples as Index)
    pdf = df_subset.to_pandas().set_index(sample_col)
    
    # 4. Prepare Row Colors (Sample Annotation)
    row_colors = None
    if metadata is not None and group_col is not None:
        # Align metadata with samples
        meta_aligned = (
            metadata
            .filter(pl.col(sample_col).is_in(pdf.index))
            .to_pandas()
            .set_index(sample_col)
            .reindex(pdf.index)
        )
        
        if group_col in meta_aligned.columns:
            # Create color map for groups
            groups = meta_aligned[group_col].unique()
            palette = sns.color_palette("Set2", len(groups))
            lut = dict(zip(groups, palette))
            row_colors = meta_aligned[group_col].map(lut)

    # 5. Create Clustermap
    # Note: Transpose pdf so Features are Rows, Samples are Columns
    g = sns.clustermap(
        pdf.T,
        method=method,
        metric=metric,
        col_colors=row_colors,
        cmap="viridis",
        yticklabels=True,
        xticklabels=False, # Hide sample names if too many
        figsize=(14, 12),
        dendrogram_ratio=(.1, .2),
        cbar_pos=(.02, .32, .03, .2)
    )
    
    # Adjust font size of features
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
    
    # Add title
    g.fig.suptitle(f"Clustermap (Top {top_n_features} Features)", y=1.02)
    
    # Export using the figure object from clustermap
    export_figure(g.fig, base_name, output_dir)