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
    format_type = st.session_state.get('format_type', 'wide')

    # Show format info
    if format_type == "long":
        st.info("📊 **LONG/STRATIFIED Format** detected - Using Abundance column for visualization")
    else:
        st.info("📊 **WIDE Format** detected - Each column is a separate feature")

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
        if format_type == "long":
            # For long format, let user select taxonomic/functional column
            feature_col = st.selectbox(
                "Stratification Column:",
                options=feature_cols,
                help="Select the taxonomic or functional stratification column (e.g., Taxonomy, PGPT)"
            )
        else:
            # For wide format, select numeric feature column
            feature_col = st.selectbox(
                "Feature (Abundance):",
                options=feature_cols,
                help="Select a feature to visualize"
            )

    # For long format, add a second selector to choose specific stratification value
    feature_value = None
    if format_type == "long" and feature_col:
        unique_values = df_merged[feature_col].unique().to_list()
        feature_value = st.selectbox(
            f"Select Specific {feature_col}:",
            options=sorted(unique_values),
            help=f"Choose which {feature_col} to visualize"
        )

    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📦 Boxplot", "📈 Scatter Plot", "📋 Summary Table"])

    with viz_tab1:
        if format_type == "long":
            st.markdown(f"### Distribution of **{feature_value}** by **{group_col}**")
            st.caption(f"Stratification type: {feature_col}")

            # For long format, use Abundance column
            # Detect abundance column name
            abundance_col = None
            for col in df_merged.columns:
                if col.lower() in ['abundance', 'count', 'value']:
                    abundance_col = col
                    break

            if abundance_col and feature_value:
                # CRITICAL FIX: Filter data for the selected feature value
                df_filtered = df_merged.filter(pl.col(feature_col) == feature_value)

                # Show count of filtered samples
                st.caption(f"📊 Showing {len(df_filtered)} observations for this {feature_col}")

                # Create boxplot with Abundance on y-axis
                fig = px.box(
                    df_filtered,
                    x=group_col,
                    y=abundance_col,
                    color=group_col,
                    points="all",
                    title=f"Abundance of {feature_value} by {group_col}"
                )
            else:
                st.error("⚠️ Could not find Abundance/Count/Value column in data")
                fig = None
        else:
            st.markdown(f"### Distribution of **{feature_col}** by **{group_col}**")

            # Create boxplot with points (Plotly 5.18+ supports Polars directly)
            fig = px.box(
                df_merged,
                x=group_col,
                y=feature_col,
                color=group_col,
                points="all",  # Show all points
                title=f"Distribution of {feature_col} by {group_col}"
            )

        if fig:
            fig.update_layout(
                height=500,
                showlegend=True,
                template="plotly_white"
            )
            st.plotly_chart(fig, width='stretch')

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

        # Create scatter plot (Plotly supports Polars)
        fig_scatter = px.scatter(
            df_merged,
            x=scatter_x,
            y=scatter_y,
            color=scatter_color if scatter_color != "None" else None,
            hover_name="Sample" if "Sample" in df_merged.columns else None,
            title=f"{scatter_y} vs {scatter_x}"
        )

        fig_scatter.update_traces(marker=dict(size=10, opacity=0.7))
        fig_scatter.update_layout(
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig_scatter, width='stretch')

    with viz_tab3:
        if format_type == "long":
            st.markdown(f"### Summary Statistics for **{feature_value}** by **{group_col}**")
        else:
            st.markdown(f"### Summary Statistics for **{feature_col}** by **{group_col}**")

        # Calculate summary statistics based on format
        if format_type == "long":
            # For long format, compute stats on Abundance column (FILTERED by feature_value)
            abundance_col = None
            for col in df_merged.columns:
                if col.lower() in ['abundance', 'count', 'value']:
                    abundance_col = col
                    break

            if abundance_col and feature_value:
                # CRITICAL FIX: Filter data for the selected feature value
                df_filtered = df_merged.filter(pl.col(feature_col) == feature_value)

                summary = (
                    df_filtered
                    .group_by(group_col)
                    .agg([
                        pl.col(abundance_col).count().alias("N"),
                        pl.col(abundance_col).mean().alias("Mean"),
                        pl.col(abundance_col).std().alias("Std"),
                        pl.col(abundance_col).min().alias("Min"),
                        pl.col(abundance_col).median().alias("Median"),
                        pl.col(abundance_col).max().alias("Max"),
                    ])
                    .with_columns(
                        (pl.col("Std") / pl.col("Mean") * 100).alias("CV%")
                    )
                    .sort(group_col)
                )
            else:
                st.error("⚠️ Could not find Abundance/Count/Value column in data")
                summary = None
        else:
            # For wide format, compute stats on selected feature column
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

        # Streamlit supports Polars DataFrames directly
        if summary is not None:
            st.dataframe(
                summary,
                width='stretch',
                height=400
            )

    # Full data table
    st.markdown("---")
    st.markdown("### 📄 Full Data Table")

    # Initialize with full data
    df_filtered = df_merged

    # Add filters
    with st.expander("🔎 Filter Data"):
        filter_col = st.selectbox("Filter by column:", options=metadata_cols + feature_cols)

        if filter_col in metadata_cols:
            # Categorical filter (Polars native)
            unique_values = df_merged[filter_col].unique().to_list()
            selected_values = st.multiselect(
                f"Select {filter_col} values:",
                options=unique_values,
                default=unique_values
            )
            df_filtered = df_merged.filter(pl.col(filter_col).is_in(selected_values))
        else:
            # Numeric filter (Polars native)
            min_val = float(df_merged[filter_col].min())
            max_val = float(df_merged[filter_col].max())
            range_val = st.slider(
                f"Range for {filter_col}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            df_filtered = df_merged.filter(
                (pl.col(filter_col) >= range_val[0]) &
                (pl.col(filter_col) <= range_val[1])
            )

        st.info(f"Showing {len(df_filtered)} / {len(df_merged)} samples")

    # Display table
    st.dataframe(
        df_filtered,
        width='stretch',
        height=400
    )

    # Download button (use Polars native CSV export)
    csv = df_filtered.write_csv()
    st.download_button(
        label="💾 Download Filtered Data as CSV",
        data=csv,
        file_name="pgptracker_filtered_data.csv",
        mime="text/csv",
    )
