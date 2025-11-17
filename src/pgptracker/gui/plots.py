# src/pgptracker/gui/plots.py

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from typing import Optional


def create_boxplot(
    df: pl.DataFrame,
    feature_col: str,
    group_col: str
) -> go.Figure:
    """
    Create a boxplot showing feature distribution across groups.

    Args:
        df: DataFrame containing feature and group data
        feature_col: Column name of the feature to plot on Y-axis
        group_col: Column name of the grouping variable for X-axis

    Returns:
        Plotly Figure object

    Example:
        Input DataFrame:
            Sample  Treatment  IAA_Synthesis_CLR
            S1      Control    -0.5
            S2      Control    0.2
            S3      Treated    1.5

        Output: Boxplot with Treatment on X-axis, IAA_Synthesis_CLR on Y-axis
    """
    pdf = df.to_pandas()

    fig = px.box(
        pdf,
        x=group_col,
        y=feature_col,
        color=group_col,
        points="all",
        title=f"Distribution of {feature_col} by {group_col}"
    )

    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=feature_col,
        showlegend=True,
        template="plotly_white",
        height=500
    )

    return fig


def create_scatter(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Create a scatter plot with optional color grouping.

    Args:
        df: DataFrame containing X, Y, and optional color data
        x_col: Column name for X-axis
        y_col: Column name for Y-axis
        color_col: Optional column name for color grouping

    Returns:
        Plotly Figure object

    Example:
        Input DataFrame:
            Sample  PC1    PC2    Treatment
            S1      -2.1   0.5    Control
            S2      1.3    -1.2   Treated

        Output: Scatter plot with PC1 on X, PC2 on Y, colored by Treatment
    """
    pdf = df.to_pandas()

    fig = px.scatter(
        pdf,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_name="Sample" if "Sample" in pdf.columns else None,
        title=f"{y_col} vs {x_col}"
    )

    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        height=500,
        showlegend=True if color_col else False
    )

    fig.update_traces(marker=dict(size=10, opacity=0.7))

    return fig


def create_empty_figure(message: str = "No data available") -> go.Figure:
    """
    Create an empty figure with a centered message.

    Args:
        message: Text to display in the empty figure

    Returns:
        Plotly Figure object with annotation
    """
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        height=500
    )

    return fig
