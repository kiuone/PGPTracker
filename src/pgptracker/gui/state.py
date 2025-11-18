"""
Session state management for Streamlit GUI.
"""

import streamlit as st


def init_session_state():
    """Initialize session state variables."""

    # Data loading state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if 'df_merged' not in st.session_state:
        st.session_state.df_merged = None

    if 'metadata_cols' not in st.session_state:
        st.session_state.metadata_cols = []

    if 'feature_cols' not in st.session_state:
        st.session_state.feature_cols = []

    if 'sample_col' not in st.session_state:
        st.session_state.sample_col = "Sample"

    # Metrics
    if 'n_samples' not in st.session_state:
        st.session_state.n_samples = 0

    if 'n_features' not in st.session_state:
        st.session_state.n_features = 0

    if 'n_metadata_cols' not in st.session_state:
        st.session_state.n_metadata_cols = 0

    # Results directory (if auto-loaded)
    if 'results_dir' not in st.session_state:
        st.session_state.results_dir = None

    # Auto-load tracking
    if 'auto_load_attempted' not in st.session_state:
        st.session_state.auto_load_attempted = False

    if 'auto_load_status' not in st.session_state:
        st.session_state.auto_load_status = None

    # Data format type (wide or long)
    if 'format_type' not in st.session_state:
        st.session_state.format_type = 'wide'
