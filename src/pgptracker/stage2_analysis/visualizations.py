# src/pgptracker/stage2_analysis/visualizations.py

import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import textwrap

from pgptracker.stage2_analysis.plot_funcs import export_figure, setup_matplotlib_style

# Global Constants
SPLIT_CHAR = "|"

# Initialize style globally
setup_matplotlib_style()

def _wrap_labels(labels: List[str], width: int = 15) -> List[str]:
    """Helper to wrap long text labels into multiple lines."""
    return ['\n'.join(textwrap.wrap(str(l), width)) for l in labels]

def _filter_list_for_diversity(
    df: pl.DataFrame,
    candidate_features: List[str],
    top_n: int,
    max_per_taxon: int = 3
) -> List[str]:
    """
    Filters a list of significant features to ensure taxonomic diversity.
    Selects top 'max_per_taxon' features from each Taxon based on variance.
    """
    # 1. Quick exit if list is already small enough
    if len(candidate_features) <= top_n:
        print(f"  [Visualizations] Warning: Input list size ({len(candidate_features)}) <= top_n ({top_n}). Diversity filter skipped.")
        return candidate_features

    # 2. Validate columns exist
    valid_cols = [c for c in candidate_features if c in df.columns]
    if not valid_cols:
        return []

    # 3. Calculate Variance (Proxy for visual impact)
    var_df = df.select(valid_cols).var().transpose(
        include_header=True, header_name="Feature", column_names=["Variance"]
    )

    if var_df.height == 0:
        return []

    # 4. Check for Stratification
    sample_feat = str(var_df["Feature"][0])
    if SPLIT_CHAR not in sample_feat:
        return var_df.sort("Variance", descending=True).head(top_n)["Feature"].to_list()

    # 5. Diversity Logic
    var_df = var_df.with_columns(
        pl.col("Feature").str.split(SPLIT_CHAR).list.get(0).alias("Taxon")
    )

    diverse_df = (
        var_df
        .with_columns(
            pl.col("Variance").rank("ordinal", descending=True).over("Taxon").alias("Taxon_Rank")
        )
        .filter(pl.col("Taxon_Rank") <= max_per_taxon)
        .sort("Variance", descending=True)
    )

    selected_features = diverse_df["Feature"].to_list()

    # 6. Backfill
    if len(selected_features) < top_n:
        remaining_slots = top_n - len(selected_features)
        
        leftover_features = (
            var_df
            .filter(~pl.col("Feature").is_in(selected_features))
            .sort("Variance", descending=True)
            .head(remaining_slots)
            ["Feature"].to_list()
        )
        selected_features.extend(leftover_features)

    return selected_features[:top_n]

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
    base_name: str = "ordination",
    formats: List[str] = ['png', 'pdf']
) -> None:
    """Plots 2D ordination results (PCA, PCoA, t-SNE)."""
    if group_col not in df_scores.columns:
        df_plot = df_scores.join(
            metadata.select([sample_col, group_col]), 
            on=sample_col, 
            how="inner"
        )
    else:
        df_plot = df_scores

    pdf = df_plot.to_pandas()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.scatterplot(
        data=pdf, x=x_col, y=y_col, hue=group_col, style=group_col,
        s=120, alpha=0.8, ax=ax
    )
    
    if df_loadings is not None and x_col == "PC1":
        loadings = df_loadings.with_columns(
            (pl.col("PC1")**2 + pl.col("PC2")**2).alias("magnitude")
        ).sort("magnitude", descending=True).head(5)
        
        x_range = pdf[x_col].max() - pdf[x_col].min()
        y_range = pdf[y_col].max() - pdf[y_col].min()
        scale_x = x_range * 0.8
        scale_y = y_range * 0.8
        
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
            
            # Keep full name or simplify just for biplot if needed, currently keeping full
            feat_name = row["Feature"]
            
            off_x, off_y = offsets[i % len(offsets)]
            
            ax.text(
                lx * off_x, ly * off_y, 
                feat_name, 
                color='darkred', 
                fontsize=10, 
                fontweight='bold',
                ha='center', 
                va='center'
            )
    
    ax.set_title(title)
    if pdf[group_col].nunique() > 0:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title=group_col)
    
    export_figure(fig, base_name, output_dir, formats=formats)
    plt.close(fig)

def plot_alpha_diversity(
    df_alpha: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_col: str,
    group_col: str,
    output_dir: Path,
    formats: List[str] = ['png', 'pdf']
) -> None:
    """Generates boxplots for each Alpha Diversity metric."""
    df_plot = df_alpha.join(metadata.select([sample_col, group_col]), on=sample_col, how="inner")
    pdf = df_plot.to_pandas()
    metrics = pdf['Metric'].unique()
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 7))
        subset = pdf[pdf['Metric'] == metric]
        
        sns.boxplot(data=subset, x=group_col, y='Value', hue=group_col, palette="Set2", ax=ax, legend=False)
        sns.stripplot(data=subset, x=group_col, y='Value', color='black', alpha=0.5, ax=ax)
        
        ax.set_title(f"Alpha Diversity: {metric}")
        ax.set_ylabel(metric)
        
        labels = [item.get_text() for item in ax.get_xticklabels()]
        wrapped_labels = _wrap_labels(labels, width=12)
        
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(wrapped_labels, rotation=0)
        
        export_figure(fig, f"alpha_{metric}", output_dir, formats=formats)
        plt.close(fig)

def plot_feature_importance(
    df_importance: pl.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    output_dir: Path = Path("."),
    base_name: str = "feature_importance",
    formats: List[str] = ['png', 'pdf']
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
    
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    
    # Use full names as requested
    sns.barplot(
        data=pdf, x=val_col, y='Feature', hue='Feature',
        palette="viridis", legend=False, ax=ax
    )
    
    ax.set_title(f"{title} (Top {top_n})")
    ax.set_xlabel("Importance Score")
    
    export_figure(fig, base_name, output_dir, formats=formats)
    plt.close(fig)

def plot_volcano(
    df_stats: pl.DataFrame,
    p_val_col: str = "p_value",
    effect_col: str = "test_statistic",
    p_threshold: float = 0.05,
    output_dir: Path = Path("."),
    base_name: str = "volcano_plot",
    formats: List[str] = ['png', 'pdf']
) -> None:
    """Creates a Volcano-like plot."""
    df_clean = df_stats.filter(pl.col(p_val_col).is_not_null() & (pl.col(p_val_col) > 0))
    
    df_plot = df_clean.with_columns([
        (-pl.col(p_val_col).log10()).alias("log_p"),
        (pl.col(p_val_col) < p_threshold).alias("Significant")
    ])
    
    pdf = df_plot.to_pandas()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        data=pdf, x=effect_col, y="log_p", hue="Significant",
        palette={True: "#e74c3c", False: "#95a5a6"},
        alpha=0.7, ax=ax
    )
    
    ax.axhline(-np.log10(p_threshold), color='blue', linestyle='--', alpha=0.5, label=f'p={p_threshold}')
    ax.set_title("Feature Significance (Volcano Plot)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_xlabel("Test Statistic / Effect Size")
    ax.legend()
    
    export_figure(fig, base_name, output_dir, formats=formats)
    plt.close(fig)

def plot_heatmap(
    df_wide_N_D: pl.DataFrame,
    metadata: Optional[pl.DataFrame] = None,
    sample_col: str = "Sample",
    group_col: Optional[str] = None,
    feature_list: Optional[List[str]] = None,
    top_n_features: int = 50,
    method: str = "ward",
    metric: str = "euclidean",
    output_dir: Path = Path("."),
    base_name: str = "heatmap_clustering",
    formats: List[str] = ['png', 'pdf']
) -> None:
    """
    Generates a clustered heatmap with improved layout and readability.
    """
    all_feat_cols = [c for c in df_wide_N_D.columns if c != sample_col]
    
    # 1. Feature Selection (Full List passed, filtering happens here)
    candidates = feature_list if feature_list else all_feat_cols
    
    final_feats = _filter_list_for_diversity(
        df_wide_N_D, 
        candidates, 
        top_n=top_n_features, 
        max_per_taxon=3)

    if not final_feats:
        print("Warning: No features selected for heatmap.")
        return

    # 2. Prepare Data
    df_subset = df_wide_N_D.select([sample_col] + final_feats)
    pdf = df_subset.to_pandas().set_index(sample_col)
    
    # 3. Prepare Metadata Colors
    col_colors = None
    lut = {}
    
    if metadata is not None and group_col is not None:
        meta_aligned = (
            metadata.filter(pl.col(sample_col).is_in(pdf.index))
            .to_pandas()
            .set_index(sample_col)
            .reindex(pdf.index))
        
        if group_col in meta_aligned.columns:
            groups = meta_aligned[group_col].dropna().unique()
            # Use high contrast palette
            if len(groups) <= 10:
                palette = sns.color_palette("tab10", len(groups))
            else:
                palette = sns.color_palette("turbo", len(groups))
                
            lut = dict(zip(groups, palette))
            col_colors = meta_aligned[group_col].map(lut)

    # 4. Plotting Configuration
    show_sample_names = len(pdf.index) < 60
    
    # Increase width significantly to accommodate long labels and legend
    figsize = (18, max(12, len(final_feats) * 0.3))

    g = sns.clustermap(
        pdf.T,
        method=method,
        metric=metric,
        col_colors=col_colors,
        cmap="viridis",
        robust=True,
        yticklabels=True,
        xticklabels=show_sample_names,
        figsize=figsize,
        dendrogram_ratio=(.1, .15), # Aumenta espaço do dendrograma de amostras
        cbar_pos=(0.02, 0.8, 0.03, 0.15), # Posição da barra de cores (Esq, Cima, Largura, Altura)
        cbar_kws={'label': 'CLR Abundance', 'orientation': 'vertical'},
        tree_kws={'linewidths': 1.5} # Linhas do dendrograma mais grossas
)
    
    g.ax_heatmap.set_xlabel("")
    
    if show_sample_names:
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    
    # Adjust Y-axis font size dynamically based on feature count
    y_fontsize = 10 if len(final_feats) < 30 else 8
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=y_fontsize)
    
    # 5. Legend Improvements
    if col_colors is not None and lut:
        # Create invisible bars for the legend
        for label, color in lut.items():
            g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
        
        # Move legend FAR to the right to avoid overlap
        g.ax_col_dendrogram.legend(
            loc="center left", 
            bbox_to_anchor=(1.1, 0.5), # X=1.1 tira do gráfico, Y=0.5 centraliza
            ncol=1,
            frameon=True, 
            title=group_col,
            fontsize=10)

    # Title adjustments
    g.figure.suptitle(
        f"Top {len(final_feats)} Significant Features (Diversity Filtered)\nCLR-Normalized Abundance", 
        y=1.02, fontsize=16, fontweight='bold')
    
    export_figure(g.figure, base_name, output_dir, formats=formats)
    plt.close(g.figure)