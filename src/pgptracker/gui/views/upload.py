"""
Upload/Data Loading view with smart auto-discovery.
"""

import streamlit as st
import polars as pl
from pathlib import Path
from pgptracker.gui import utils


def auto_load_from_directory(results_dir: Path):
    """
    Automatically load CLR data and metadata from results directory.
    Stores data in session state without displaying messages.

    Args:
        results_dir: Path to pipeline results directory
    """
    # Only load once
    if st.session_state.get('auto_load_attempted', False):
        return

    st.session_state.auto_load_attempted = True

    # Look for expected files
    clr_file = results_dir / "clr_wide_N_D.tsv"
    metadata_file = results_dir / "metadata.tsv"

    if clr_file.exists() and metadata_file.exists():
        try:
            # Load files with caching
            df_clr = utils.load_tsv_file(clr_file)
            df_metadata = utils.load_tsv_file(metadata_file)

            # Auto-detect sample column
            metadata_sample_col = utils.auto_detect_sample_column(df_metadata.columns)
            if not metadata_sample_col:
                # Store for manual selection
                st.session_state.pending_metadata = df_metadata
                st.session_state.pending_clr = df_clr
                st.session_state.auto_load_status = "needs_manual_selection"
                return

            # Merge data
            df_merged, metadata_cols, feature_cols, format_type, is_clr = utils.merge_data(
                df_clr, df_metadata, metadata_sample_col, clr_filename="clr_wide_N_D.tsv"
            )

            # Store in session state
            st.session_state.df_merged = df_merged
            st.session_state.metadata_cols = metadata_cols
            st.session_state.feature_cols = feature_cols
            st.session_state.format_type = format_type
            st.session_state.is_clr = is_clr
            st.session_state.n_samples = df_merged.shape[0]
            st.session_state.n_features = len(feature_cols)
            st.session_state.n_metadata_cols = len(metadata_cols)
            st.session_state.data_loaded = True
            st.session_state.auto_load_status = "success"

        except Exception as e:
            st.session_state.auto_load_status = f"error: {e}"
    else:
        st.session_state.auto_load_status = "files_not_found"


def render():
    """Render the upload/data loading view."""

    st.markdown("## 📁 Data Loading")

    # Show auto-load status
    if st.session_state.get('results_dir'):
        st.info(f"📂 Results directory: `{st.session_state.results_dir}`")

        # Display auto-load status messages
        auto_status = st.session_state.get('auto_load_status')
        if auto_status == "success":
            st.success(f"✅ Auto-loaded data from results directory")
        elif auto_status == "files_not_found":
            st.warning("⚠️ Expected files (clr_wide_N_D.tsv, metadata.tsv) not found in results directory")
        elif auto_status and auto_status.startswith("error:"):
            st.error(f"❌ {auto_status}")
        elif auto_status == "needs_manual_selection":
            st.info("ℹ️ Sample ID column could not be auto-detected. Please select manually below.")

    # Check if data already loaded
    if st.session_state.get('data_loaded', False):
        st.success("✅ Data successfully loaded!")

        # Show format type
        format_type = st.session_state.get('format_type', 'wide')
        st.info(f"📊 Data format detected: **{format_type.upper()}**")

        # Show summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", st.session_state.n_samples)
        with col2:
            st.metric("Features", st.session_state.n_features)
        with col3:
            st.metric("Metadata Columns", st.session_state.n_metadata_cols)
        with col4:
            st.metric("Format", format_type.upper())

        # Show preview (Streamlit supports Polars DataFrames directly)
        st.markdown("### 👀 Data Preview")
        st.dataframe(
            st.session_state.df_merged.head(10),
            width='stretch',
            height=300
        )

        # Option to reset
        if st.button("🗑️ Clear Data & Load Different Files", type="secondary"):
            # Clear all data from session state
            for key in ['data_loaded', 'df_merged', 'metadata_cols', 'feature_cols',
                       'format_type', 'n_samples', 'n_features', 'n_metadata_cols',
                       'auto_load_attempted', 'auto_load_status']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    else:
        # Manual upload interface
        st.markdown("### 📤 Manual Upload")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1. Upload Metadata")
            metadata_file = st.file_uploader(
                "metadata file",
                type=["tsv", "csv", "txt", "gz"],
                key="metadata_upload",
                help="Upload your metadata file (TSV, CSV, or compressed .gz)"
            )

        with col2:
            st.markdown("#### 2. Upload CLR/Feature Data")
            clr_file = st.file_uploader(
                "CLR or feature data",
                type=["tsv", "csv", "txt", "gz"],
                key="clr_upload",
                help="Upload your CLR-transformed data (TSV, CSV, or compressed .gz)\n"
                     "Supports both WIDE (N×D) and LONG/STRATIFIED formats"
            )

        # Process uploads
        if metadata_file and clr_file:
            with st.spinner("Loading data..."):
                try:
                    # Load files
                    df_metadata = utils.load_uploaded_file(metadata_file)
                    df_clr = utils.load_uploaded_file(clr_file)

                    # Detect format (using filename for better accuracy)
                    format_type = utils.detect_table_format(df_clr, filename=clr_file.name)
                    st.info(f"📊 Detected format: **{format_type.upper()}** (from: {clr_file.name})")

                    # Let user select sample column
                    st.markdown("#### 3. Select Sample ID Column")
                    detected_col = utils.auto_detect_sample_column(df_metadata.columns)
                    default_index = df_metadata.columns.index(detected_col) if detected_col else 0

                    metadata_sample_col = st.selectbox(
                        "Which column contains the sample IDs in metadata?",
                        options=df_metadata.columns,
                        index=default_index
                    )

                    if st.button("✅ Load Data", type="primary"):
                        # Merge data
                        df_merged, metadata_cols, feature_cols, format_type, is_clr = utils.merge_data(
                            df_clr, df_metadata, metadata_sample_col, clr_filename=clr_file.name
                        )

                        # Store in session state
                        st.session_state.df_merged = df_merged
                        st.session_state.metadata_cols = metadata_cols
                        st.session_state.feature_cols = feature_cols
                        st.session_state.format_type = format_type
                        st.session_state.is_clr = is_clr
                        st.session_state.n_samples = df_merged.shape[0]
                        st.session_state.n_features = len(feature_cols)
                        st.session_state.n_metadata_cols = len(metadata_cols)
                        st.session_state.data_loaded = True

                        # Show data type detection result
                        data_type = "CLR-transformed" if is_clr else "Raw counts"
                        st.success(f"✅ Data loaded successfully! Detected: **{data_type}**")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
