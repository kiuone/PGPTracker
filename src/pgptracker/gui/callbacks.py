# src/pgptracker/gui/callbacks.py

import base64
import io
import logging
import polars as pl
from typing import List, Optional, Tuple
from dash import callback, Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
from pgptracker.gui import ids, plots

logger = logging.getLogger(__name__)


@callback(
    [
        Output(ids.STORE_RAW_METADATA, "data"),
        Output(ids.DROPDOWN_SAMPLE_ID, "options"),
        Output(ids.UPLOAD_METADATA, "children")
    ],
    Input(ids.UPLOAD_METADATA, "contents"),
    State(ids.UPLOAD_METADATA, "filename"),
    prevent_initial_call=True
)
def store_raw_metadata(metadata_content, filename):
    """
    Store raw metadata and populate sample ID dropdown with all column names.

    Args:
        metadata_content: Base64 encoded metadata file content
        filename: Name of uploaded file

    Returns:
        Tuple of (metadata_json, dropdown_options)
    """
    print(f"DEBUG: Metadata upload callback triggered for file: {filename}")
    logger.info(f"Metadata upload callback triggered for file: {filename}")

    if not metadata_content:
        logger.warning("Metadata upload triggered but no content received")
        raise PreventUpdate

    try:
        logger.info(f"Processing metadata upload: {filename}")
        meta_decoded = base64.b64decode(metadata_content.split(",")[1])
        df_metadata = pl.read_csv(
            io.BytesIO(meta_decoded),
            separator="\t",
            infer_schema_length=None,  # Scan entire file to avoid type inference errors
            null_values=["NA", "nan", "null", ""]  # Handle common null representations
        )

        logger.info(f"Metadata loaded: {df_metadata.shape[0]} rows, {df_metadata.shape[1]} columns")

        metadata_json = df_metadata.to_pandas().to_json(orient="split")

        # Populate dropdown with all columns for manual selection if needed
        column_options = [{"label": col, "value": col} for col in df_metadata.columns]

        # Create visual feedback showing file uploaded
        upload_visual = html.Div(
            [
                html.I(className="bi bi-file-earmark-check-fill text-success me-2", style={"fontSize": "24px"}),
                html.Div(
                    [
                        html.Strong(filename, className="text-success"),
                        html.Br(),
                        html.Small(f"{df_metadata.shape[0]} rows × {df_metadata.shape[1]} columns", className="text-muted")
                    ],
                    style={"display": "inline-block", "verticalAlign": "middle"}
                )
            ],
            className="d-flex align-items-center justify-content-center",
            style={"padding": "20px"}
        )

        logger.info(f"Metadata successfully stored with {len(column_options)} column options")
        return metadata_json, column_options, upload_visual

    except Exception as e:
        logger.error(f"Failed to process metadata upload: {e}", exc_info=True)
        raise PreventUpdate


@callback(
    [
        Output(ids.STORE_RAW_CLR_DATA, "data"),
        Output(ids.UPLOAD_CLR_DATA, "children")
    ],
    Input(ids.UPLOAD_CLR_DATA, "contents"),
    State(ids.UPLOAD_CLR_DATA, "filename"),
    prevent_initial_call=True
)
def store_raw_clr_data(clr_content, filename):
    """
    Store raw CLR data for merging.

    Args:
        clr_content: Base64 encoded CLR data file content
        filename: Name of uploaded file

    Returns:
        JSON string of CLR DataFrame
    """
    print(f"DEBUG: CLR data upload callback triggered for file: {filename}")
    logger.info(f"CLR data upload callback triggered for file: {filename}")

    if not clr_content:
        logger.warning("CLR data upload triggered but no content received")
        raise PreventUpdate

    try:
        logger.info(f"Processing CLR data upload: {filename}")
        clr_decoded = base64.b64decode(clr_content.split(",")[1])
        df_clr = pl.read_csv(
            io.BytesIO(clr_decoded),
            separator="\t",
            infer_schema_length=None,  # Scan entire file to avoid type inference errors
            null_values=["NA", "nan", "null", ""]  # Handle common null representations
        )

        logger.info(f"CLR data loaded: {df_clr.shape[0]} rows, {df_clr.shape[1]} columns")

        clr_json = df_clr.to_pandas().to_json(orient="split")

        # Create visual feedback showing file uploaded
        upload_visual = html.Div(
            [
                html.I(className="bi bi-file-earmark-check-fill text-success me-2", style={"fontSize": "24px"}),
                html.Div(
                    [
                        html.Strong(filename, className="text-success"),
                        html.Br(),
                        html.Small(f"{df_clr.shape[0]} rows × {df_clr.shape[1]} columns", className="text-muted")
                    ],
                    style={"display": "inline-block", "verticalAlign": "middle"}
                )
            ],
            className="d-flex align-items-center justify-content-center",
            style={"padding": "20px"}
        )

        logger.info("CLR data successfully stored")
        return clr_json, upload_visual

    except Exception as e:
        logger.error(f"Failed to process CLR data upload: {e}", exc_info=True)
        raise PreventUpdate


def _auto_detect_sample_column(columns: List[str]) -> Optional[str]:
    """
    Attempt to auto-detect sample ID column from common patterns.

    Args:
        columns: List of column names to search

    Returns:
        Detected column name or None if not found
    """
    common_patterns = ["Sample", "SampleID", "sample", "sampleID", "sample_id", "SAMPLE"]

    for candidate in common_patterns:
        if candidate in columns:
            return candidate

    return None


@callback(
    [
        Output(ids.STORE_DATA_N_D, "data"),
        Output(ids.STORE_METADATA_COLS, "data"),
        Output(ids.STORE_FEATURE_COLS, "data"),
        Output(ids.DIV_UPLOAD_STATUS, "children"),
        Output(ids.DIV_DATA_SUMMARY, "children")
    ],
    [
        Input(ids.STORE_RAW_METADATA, "data"),
        Input(ids.STORE_RAW_CLR_DATA, "data"),
        Input(ids.DROPDOWN_SAMPLE_ID, "value")
    ],
    [
        State(ids.UPLOAD_METADATA, "filename"),
        State(ids.UPLOAD_CLR_DATA, "filename")
    ]
)
def load_and_merge_data(
    metadata_json,
    clr_json,
    selected_sample_col,
    metadata_filename,
    clr_filename
):
    """
    Merge metadata and CLR data with flexible sample ID column detection.

    Uses auto-detection first. If that fails, requires manual selection via dropdown.
    Handles large data serialization errors gracefully.

    Args:
        metadata_json: JSON string of metadata DataFrame
        clr_json: JSON string of CLR DataFrame
        selected_sample_col: User-selected sample column (or None for auto-detect)
        metadata_filename: Name of uploaded metadata file
        clr_filename: Name of uploaded CLR file

    Returns:
        Tuple of (merged_data_json, metadata_cols, feature_cols, status_div, summary_div)
    """
    if not metadata_json or not clr_json:
        status = html.Div(
            [
                html.I(className="bi bi-info-circle me-2"),
                "Waiting for both files..."
            ],
            className="text-muted"
        )
        return None, None, None, status, ""

    try:
        import pandas as pd
        from io import StringIO

        # Load dataframes from stored JSON (using StringIO to avoid FutureWarning)
        df_metadata = pl.from_pandas(pd.read_json(StringIO(metadata_json), orient="split"))
        df_clr = pl.from_pandas(pd.read_json(StringIO(clr_json), orient="split"))

        # Step 1: Determine sample column in METADATA
        # Use manual selection if provided, otherwise auto-detect
        if selected_sample_col:
            metadata_sample_col = selected_sample_col
        else:
            metadata_sample_col = _auto_detect_sample_column(df_metadata.columns)

        # If auto-detection failed and no manual selection, prompt user
        if not metadata_sample_col:
            error_status = html.Div(
                [
                    html.I(className="bi bi-exclamation-triangle-fill text-warning me-2"),
                    html.Div([
                        html.Strong("Sample ID column not detected."),
                        html.Br(),
                        "Please select the correct column from the '3. Sample ID Column' dropdown above."
                    ])
                ],
                className="text-warning"
            )
            return None, None, None, error_status, ""

        # Validate sample column exists in metadata
        if metadata_sample_col not in df_metadata.columns:
            raise ValueError(
                f"Selected column '{metadata_sample_col}' not found in metadata. "
                f"Available columns: {', '.join(df_metadata.columns)}"
            )

        # Step 2: Determine sample column in CLR DATA
        # Auto-detect in CLR data (usually "Sample" or first column)
        clr_sample_col = _auto_detect_sample_column(df_clr.columns)
        if not clr_sample_col:
            # Fallback: assume first column is sample ID
            clr_sample_col = df_clr.columns[0]
            logger.info(f"CLR sample column auto-detection failed, using first column: {clr_sample_col}")

        # Step 3: Harmonize column names to standard "Sample"
        # Rename metadata sample column to "Sample" if needed
        if metadata_sample_col != "Sample":
            df_metadata = df_metadata.rename({metadata_sample_col: "Sample"})
            logger.info(f"Renamed metadata column '{metadata_sample_col}' -> 'Sample'")

        # Rename CLR sample column to "Sample" if needed
        if clr_sample_col != "Sample":
            df_clr = df_clr.rename({clr_sample_col: "Sample"})
            logger.info(f"Renamed CLR column '{clr_sample_col}' -> 'Sample'")

        # Step 4: Merge datasets on harmonized "Sample" column
        df_merged = df_clr.join(df_metadata, on="Sample", how="inner")
        logger.info(f"Merged {df_clr.shape[0]} CLR rows with {df_metadata.shape[0]} metadata rows -> {df_merged.shape[0]} final rows")

        # Extract metadata and feature column names (excluding harmonized "Sample" column)
        metadata_cols = [col for col in df_metadata.columns if col != "Sample"]
        feature_cols = [col for col in df_clr.columns if col != "Sample"]

        # Create status message
        status = html.Div(
            [
                html.Div(
                    [
                        html.I(className="bi bi-check-circle-fill text-success me-2"),
                        f"Metadata: {metadata_filename}"
                    ]
                ),
                html.Div(
                    [
                        html.I(className="bi bi-check-circle-fill text-success me-2"),
                        f"CLR Data: {clr_filename}"
                    ],
                    className="mt-2"
                ),
                html.Div(
                    [
                        html.I(className="bi bi-check-circle-fill text-success me-2"),
                        f"Sample ID Column: {metadata_sample_col}"
                    ],
                    className="mt-2"
                )
            ]
        )

        # Create summary
        summary = html.Div(
            [
                html.P([html.Strong("Samples: "), str(df_merged.shape[0])]),
                html.P([html.Strong("Features: "), str(len(feature_cols))]),
                html.P([html.Strong("Metadata Columns: "), str(len(metadata_cols))]),
                html.Hr(),
                html.P([html.Strong("Available Metadata:")], className="mb-1"),
                html.Ul([html.Li(col) for col in metadata_cols], className="small")
            ]
        )

        # Convert to JSON for storage with error handling for large datasets
        try:
            merged_json = df_merged.to_pandas().to_json(orient="split")
        except (MemoryError, ValueError) as e:
            error_status = html.Div(
                [
                    html.I(className="bi bi-exclamation-triangle-fill text-danger me-2"),
                    html.Div([
                        html.Strong("Dataset too large for browser storage."),
                        html.Br(),
                        f"Error: {str(e)}",
                        html.Br(),
                        "Please filter your data to reduce size or use command-line tools."
                    ])
                ],
                className="text-danger"
            )
            return None, None, None, error_status, ""

        return merged_json, metadata_cols, feature_cols, status, summary

    except Exception as e:
        error_status = html.Div(
            [
                html.I(className="bi bi-exclamation-triangle-fill text-danger me-2"),
                f"Error: {str(e)}"
            ],
            className="text-danger"
        )
        return None, None, None, error_status, ""


@callback(
    [
        Output(ids.TABLE_DATA_EXPLORER, "columnDefs"),
        Output(ids.TABLE_DATA_EXPLORER, "rowData")
    ],
    Input(ids.STORE_DATA_N_D, "data")
)
def update_data_table(merged_data_json):
    """
    Update the ag-grid table with merged data.

    Args:
        merged_data_json: JSON string of merged DataFrame

    Returns:
        Tuple of (columnDefs, rowData) for ag-grid
    """
    if not merged_data_json:
        raise PreventUpdate

    import pandas as pd
    from io import StringIO
    df = pd.read_json(StringIO(merged_data_json), orient="split")

    # Generate column definitions
    column_defs = [{"field": col, "headerName": col} for col in df.columns]

    # Convert to row data
    row_data = df.to_dict("records")

    return column_defs, row_data


@callback(
    [
        Output(ids.DROPDOWN_GROUPBY, "options"),
        Output(ids.DROPDOWN_FEATURE, "options"),
        Output(ids.DROPDOWN_SCATTER_X, "options"),
        Output(ids.DROPDOWN_SCATTER_Y, "options")
    ],
    [
        Input(ids.STORE_METADATA_COLS, "data"),
        Input(ids.STORE_FEATURE_COLS, "data"),
        Input(ids.STORE_DATA_N_D, "data")
    ]
)
def populate_dropdowns(metadata_cols, feature_cols, merged_data_json):
    """
    Populate all dropdown options based on loaded data.

    Args:
        metadata_cols: List of metadata column names
        feature_cols: List of feature column names
        merged_data_json: JSON string of merged DataFrame

    Returns:
        Tuple of (groupby_options, feature_options, scatter_x_options, scatter_y_options)
    """
    if not metadata_cols or not feature_cols or not merged_data_json:
        raise PreventUpdate

    import pandas as pd
    from io import StringIO
    df = pd.read_json(StringIO(merged_data_json), orient="split")

    # Group by options: metadata columns only
    groupby_options = [{"label": col, "value": col} for col in metadata_cols]

    # Feature options: feature columns only
    feature_options = [{"label": col, "value": col} for col in feature_cols]

    # Scatter options: all numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    scatter_options = [{"label": col, "value": col} for col in numeric_cols]

    return groupby_options, feature_options, scatter_options, scatter_options


@callback(
    Output(ids.GRAPH_BOXPLOT, "figure"),
    [
        Input(ids.DROPDOWN_GROUPBY, "value"),
        Input(ids.DROPDOWN_FEATURE, "value"),
        Input(ids.STORE_DATA_N_D, "data"),
        Input(ids.STORE_THEME, "data")
    ]
)
def update_boxplot(group_col, feature_col, merged_data_json, theme):
    """
    Update boxplot based on selected group and feature.

    Args:
        group_col: Selected grouping column
        feature_col: Selected feature column
        merged_data_json: JSON string of merged DataFrame
        theme: Current theme (light or dark)

    Returns:
        Plotly Figure object
    """
    plotly_theme = "plotly_dark" if theme == "dark" else "plotly_white"

    if not merged_data_json or not group_col or not feature_col:
        return plots.create_empty_figure("Select both Group and Feature to display plot", theme=plotly_theme)

    import pandas as pd
    from io import StringIO
    df_pandas = pd.read_json(StringIO(merged_data_json), orient="split")
    df = pl.from_pandas(df_pandas)

    return plots.create_boxplot(df, feature_col, group_col, theme=plotly_theme)


@callback(
    Output(ids.GRAPH_SCATTER, "figure"),
    [
        Input(ids.DROPDOWN_SCATTER_X, "value"),
        Input(ids.DROPDOWN_SCATTER_Y, "value"),
        Input(ids.DROPDOWN_GROUPBY, "value"),
        Input(ids.STORE_DATA_N_D, "data"),
        Input(ids.STORE_THEME, "data")
    ]
)
def update_scatter(x_col, y_col, color_col, merged_data_json, theme):
    """
    Update scatter plot based on selected X, Y, and color columns.

    Args:
        x_col: Selected X-axis column
        y_col: Selected Y-axis column
        color_col: Selected color grouping column
        merged_data_json: JSON string of merged DataFrame
        theme: Current theme (light or dark)

    Returns:
        Plotly Figure object
    """
    plotly_theme = "plotly_dark" if theme == "dark" else "plotly_white"

    if not merged_data_json or not x_col or not y_col:
        return plots.create_empty_figure("Select both X and Y axes to display plot", theme=plotly_theme)

    import pandas as pd
    from io import StringIO
    df_pandas = pd.read_json(StringIO(merged_data_json), orient="split")
    df = pl.from_pandas(df_pandas)

    return plots.create_scatter(df, x_col, y_col, color_col, theme=plotly_theme)


@callback(
    [
        Output(ids.STORE_THEME, "data"),
        Output(ids.BTN_THEME_TOGGLE, "children")
    ],
    Input(ids.BTN_THEME_TOGGLE, "n_clicks"),
    State(ids.STORE_THEME, "data"),
    prevent_initial_call=True
)
def toggle_theme(n_clicks, current_theme):
    """
    Toggle between light and dark themes.

    Args:
        n_clicks: Number of button clicks
        current_theme: Current theme state

    Returns:
        Tuple of (new_theme, button_icon)
    """
    if current_theme == "light":
        return "dark", html.I(className="bi bi-sun-fill")
    else:
        return "light", html.I(className="bi bi-moon-fill")


@callback(
    Output(ids.MAIN_CONTAINER, "style"),
    Input(ids.STORE_THEME, "data")
)
def update_main_container_theme(theme):
    """
    Update main container styling based on theme.

    Args:
        theme: Current theme (light or dark)

    Returns:
        Style dict for main container
    """
    base_style = {
        "marginLeft": "340px",
        "marginTop": "100px",
        "padding": "20px",
        "minHeight": "calc(100vh - 100px)"
    }

    if theme == "dark":
        base_style.update({
            "backgroundColor": "#1E1E1E",
            "color": "#CCCCCC"
        })
    else:
        base_style.update({
            "backgroundColor": "#ffffff",
            "color": "#212529"
        })

    return base_style


@callback(
    Output("app-container", "style"),
    Input(ids.STORE_THEME, "data")
)
def update_app_container_theme(theme):
    """
    Update app container background for full viewport coverage.

    Args:
        theme: Current theme (light or dark)

    Returns:
        Style dict for app container
    """
    base_style = {
        "padding": "0",
        "minHeight": "100vh"
    }

    if theme == "dark":
        base_style.update({
            "backgroundColor": "#1E1E1E"
        })
    else:
        base_style.update({
            "backgroundColor": "#ffffff"
        })

    return base_style


@callback(
    Output("sidebar-container", "style"),
    Input(ids.STORE_THEME, "data")
)
def update_sidebar_theme(theme):
    """
    Update sidebar styling based on theme.
    Overrides Bootstrap card white backgrounds with professional dark mode colors.

    Args:
        theme: Current theme (light or dark)

    Returns:
        Style dict for sidebar container
    """
    base_style = {
        "position": "fixed",
        "top": "120px",
        "left": "0",
        "bottom": "0",
        "width": "320px",
        "padding": "20px",
        "overflowY": "auto",
        "zIndex": "100"
    }

    if theme == "dark":
        base_style.update({
            "backgroundColor": "#252526",
            "color": "#E0E0E0"
        })
    else:
        base_style.update({
            "backgroundColor": "#f8f9fa",
            "color": "#212529"
        })

    return base_style


@callback(
    Output(ids.TABLE_DATA_EXPLORER, "className"),
    Input(ids.STORE_THEME, "data")
)
def update_table_theme(theme):
    """
    Update ag-grid theme based on app theme.

    Args:
        theme: Current theme (light or dark)

    Returns:
        className for ag-grid
    """
    if theme == "dark":
        return "ag-theme-alpine-dark"
    else:
        return "ag-theme-alpine"
