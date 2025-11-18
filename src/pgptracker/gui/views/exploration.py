"""
Data exploration view with interactive visualizations.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl


def render():
    """Render the data exploration view."""

    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please load data first in the 'Upload Data' tab.")
        return

    st.markdown("## 🔍 Interactive Data Exploration")

    # Get data from session state
    df_merged = st.session_state.df_merged
    metadata_cols = st.session_state.metadata_cols
    feature_cols = st.session_state.feature_cols

    # Convert to pandas for plotting (Plotly works better with pandas)
    df_pandas = df_merged.to_pandas()

    # Control panel
    st.markdown("### 📊 Plot Controls")

    col1, col2 = st.columns(2)

    with col1:
        group_col = st.selectbox(
            "Group By (Metadata Column):",
            options=metadata_cols,
            help="Select a metadata column to group samples"
        )

    with col2:
        feature_col = st.selectbox(
            "Feature (Abundance):",
            options=feature_cols,
            help="Select a feature to visualize"
        )

    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📦 Boxplot", "📈 Scatter Plot", "📋 Summary Table"])

    with viz_tab1:
        st.markdown(f"### Distribution of **{feature_col}** by **{group_col}**")

        # Create boxplot with points
        fig = px.box(
            df_pandas,
            x=group_col,
            y=feature_col,
            color=group_col,
            points="all",  # Show all points
            title=f"Distribution of {feature_col} by {group_col}"
        )

        fig.update_layout(
            height=500,
            showlegend=True,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with viz_tab2:
        st.markdown("### Scatter Plot")

        col_x, col_y, col_color = st.columns(3)

        with col_x:
            scatter_x = st.selectbox(
                "X-axis:",
                options=feature_cols,
                key="scatter_x"
            )

        with col_y:
            scatter_y = st.selectbox(
                "Y-axis:",
                options=feature_cols,
                index=min(1, len(feature_cols) - 1),
                key="scatter_y"
            )

        with col_color:
            scatter_color = st.selectbox(
                "Color by:",
                options=["None"] + metadata_cols,
                key="scatter_color"
            )

        # Create scatter plot
        fig_scatter = px.scatter(
            df_pandas,
            x=scatter_x,
            y=scatter_y,
            color=scatter_color if scatter_color != "None" else None,
            hover_name="Sample" if "Sample" in df_pandas.columns else None,
            title=f"{scatter_y} vs {scatter_x}"
        )

        fig_scatter.update_traces(marker=dict(size=10, opacity=0.7))
        fig_scatter.update_layout(
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    with viz_tab3:
        st.markdown(f"### Summary Statistics for **{feature_col}** by **{group_col}**")

        # Calculate summary statistics
        summary = (
            df_merged
            .group_by(group_col)
            .agg([
                pl.col(feature_col).count().alias("N"),
                pl.col(feature_col).mean().alias("Mean"),
                pl.col(feature_col).std().alias("Std"),
                pl.col(feature_col).min().alias("Min"),
                pl.col(feature_col).median().alias("Median"),
                pl.col(feature_col).max().alias("Max"),
            ])
            .with_columns(
                (pl.col("Std") / pl.col("Mean") * 100).alias("CV%")
            )
            .sort(group_col)
        )

        st.dataframe(
            summary.to_pandas(),
            use_container_width=True,
            height=400
        )

    # Full data table
    st.markdown("---")
    st.markdown("### 📄 Full Data Table")

    # Initialize with full data
    df_filtered = df_pandas

    # Add filters
    with st.expander("🔎 Filter Data"):
        filter_col = st.selectbox("Filter by column:", options=metadata_cols + feature_cols)

        if filter_col in metadata_cols:
            # Categorical filter
            unique_values = df_pandas[filter_col].unique()
            selected_values = st.multiselect(
                f"Select {filter_col} values:",
                options=unique_values,
                default=unique_values
            )
            df_filtered = df_pandas[df_pandas[filter_col].isin(selected_values)]
        else:
            # Numeric filter
            min_val = float(df_pandas[filter_col].min())
            max_val = float(df_pandas[filter_col].max())
            range_val = st.slider(
                f"Range for {filter_col}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            df_filtered = df_pandas[
                (df_pandas[filter_col] >= range_val[0]) &
                (df_pandas[filter_col] <= range_val[1])
            ]

        st.info(f"Showing {len(df_filtered)} / {len(df_pandas)} samples")

    # Display table
    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=400
    )

    # Download button
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Filtered Data as CSV",
        data=csv,
        file_name="pgptracker_filtered_data.csv",
        mime="text/csv",
    )
