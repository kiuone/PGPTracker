# src/pgptracker/gui/callbacks.py

import base64
import io
import polars as pl
from typing import List, Optional, Tuple
from dash import callback, Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
from pgptracker.gui import ids, plots


@callback(
    [
        Output(ids.STORE_RAW_METADATA, "data"),
        Output(ids.DROPDOWN_SAMPLE_ID, "options")
    ],
    Input(ids.UPLOAD_METADATA, "contents")
)
def store_raw_metadata(metadata_content):
    """
    Store raw metadata and populate sample ID dropdown with all column names.

    Args:
        metadata_content: Base64 encoded metadata file content

    Returns:
        Tuple of (metadata_json, dropdown_options)
    """
    if not metadata_content:
        raise PreventUpdate

    try:
        meta_decoded = base64.b64decode(metadata_content.split(",")[1])
        df_metadata = pl.read_csv(io.BytesIO(meta_decoded), separator="\t")

        metadata_json = df_metadata.to_pandas().to_json(orient="split")

        # Populate dropdown with all columns for manual selection if needed
        column_options = [{"label": col, "value": col} for col in df_metadata.columns]

        return metadata_json, column_options

    except Exception:
        raise PreventUpdate


@callback(
    Output(ids.STORE_RAW_CLR_DATA, "data"),
    Input(ids.UPLOAD_CLR_DATA, "contents")
)
def store_raw_clr_data(clr_content):
    """
    Store raw CLR data for merging.

    Args:
        clr_content: Base64 encoded CLR data file content

    Returns:
        JSON string of CLR DataFrame
    """
    if not clr_content:
        raise PreventUpdate

    try:
        clr_decoded = base64.b64decode(clr_content.split(",")[1])
        df_clr = pl.read_csv(io.BytesIO(clr_decoded), separator="\t")

        clr_json = df_clr.to_pandas().to_json(orient="split")

        return clr_json

    except Exception:
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

        # Load dataframes from stored JSON
        df_metadata = pl.from_pandas(pd.read_json(metadata_json, orient="split"))
        df_clr = pl.from_pandas(pd.read_json(clr_json, orient="split"))

        # Determine sample column: use manual selection if provided, otherwise auto-detect
        if selected_sample_col:
            sample_col = selected_sample_col
        else:
            sample_col = _auto_detect_sample_column(df_metadata.columns)

        # If auto-detection failed and no manual selection, prompt user
        if not sample_col:
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

        # Validate sample column exists in both datasets
        if sample_col not in df_metadata.columns:
            raise ValueError(
                f"Selected column '{sample_col}' not found in metadata. "
                f"Available columns: {', '.join(df_metadata.columns)}"
            )

        if sample_col not in df_clr.columns:
            raise ValueError(
                f"Selected column '{sample_col}' not found in CLR data. "
                f"Available columns: {', '.join(df_clr.columns)}"
            )

        # Merge datasets
        df_merged = df_clr.join(df_metadata, on=sample_col, how="inner")

        # Extract metadata and feature column names
        metadata_cols = [col for col in df_metadata.columns if col != sample_col]
        feature_cols = [col for col in df_clr.columns if col != sample_col]

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
                        f"Sample ID Column: {sample_col}"
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
    df = pd.read_json(merged_data_json, orient="split")

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
    df = pd.read_json(merged_data_json, orient="split")

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
        Input(ids.STORE_DATA_N_D, "data")
    ]
)
def update_boxplot(group_col, feature_col, merged_data_json):
    """
    Update boxplot based on selected group and feature.

    Args:
        group_col: Selected grouping column
        feature_col: Selected feature column
        merged_data_json: JSON string of merged DataFrame

    Returns:
        Plotly Figure object
    """
    if not merged_data_json or not group_col or not feature_col:
        return plots.create_empty_figure("Select both Group and Feature to display plot")

    import pandas as pd
    df_pandas = pd.read_json(merged_data_json, orient="split")
    df = pl.from_pandas(df_pandas)

    return plots.create_boxplot(df, feature_col, group_col)


@callback(
    Output(ids.GRAPH_SCATTER, "figure"),
    [
        Input(ids.DROPDOWN_SCATTER_X, "value"),
        Input(ids.DROPDOWN_SCATTER_Y, "value"),
        Input(ids.DROPDOWN_GROUPBY, "value"),
        Input(ids.STORE_DATA_N_D, "data")
    ]
)
def update_scatter(x_col, y_col, color_col, merged_data_json):
    """
    Update scatter plot based on selected X, Y, and color columns.

    Args:
        x_col: Selected X-axis column
        y_col: Selected Y-axis column
        color_col: Selected color grouping column
        merged_data_json: JSON string of merged DataFrame

    Returns:
        Plotly Figure object
    """
    if not merged_data_json or not x_col or not y_col:
        return plots.create_empty_figure("Select both X and Y axes to display plot")

    import pandas as pd
    df_pandas = pd.read_json(merged_data_json, orient="split")
    df = pl.from_pandas(df_pandas)

    return plots.create_scatter(df, x_col, y_col, color_col)
