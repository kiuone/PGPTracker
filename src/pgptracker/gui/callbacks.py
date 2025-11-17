# src/pgptracker/gui/callbacks.py

import base64
import io
import polars as pl
from dash import callback, Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
from pgptracker.gui import ids, plots


@callback(
    [
        Output(ids.STORE_DATA_N_D, "data"),
        Output(ids.STORE_METADATA_COLS, "data"),
        Output(ids.STORE_FEATURE_COLS, "data"),
        Output(ids.DIV_UPLOAD_STATUS, "children"),
        Output(ids.DIV_DATA_SUMMARY, "children")
    ],
    [
        Input(ids.UPLOAD_METADATA, "contents"),
        Input(ids.UPLOAD_CLR_DATA, "contents")
    ],
    [
        State(ids.UPLOAD_METADATA, "filename"),
        State(ids.UPLOAD_CLR_DATA, "filename")
    ]
)
def load_and_merge_data(metadata_content, clr_content, metadata_filename, clr_filename):
    """
    Load and merge metadata and CLR data files.

    Triggered when either file is uploaded. Only processes when both files are available.

    Returns:
        Tuple of (merged_data_json, metadata_cols, feature_cols, status_div, summary_div)
    """
    if not metadata_content or not clr_content:
        status = html.Div(
            [
                html.I(className="bi bi-info-circle me-2"),
                "Waiting for both files..."
            ],
            className="text-muted"
        )
        return None, None, None, status, ""

    try:
        # Decode and load metadata
        meta_decoded = base64.b64decode(metadata_content.split(",")[1])
        df_metadata = pl.read_csv(io.BytesIO(meta_decoded), separator="\t")

        # Decode and load CLR data
        clr_decoded = base64.b64decode(clr_content.split(",")[1])
        df_clr = pl.read_csv(io.BytesIO(clr_decoded), separator="\t")

        # Identify sample column (flexible: "Sample" or "SampleID")
        sample_col = None
        for candidate in ["Sample", "SampleID", "sample", "sampleID"]:
            if candidate in df_metadata.columns:
                sample_col = candidate
                break

        if not sample_col:
            raise ValueError("Metadata must contain 'Sample' or 'SampleID' column")

        if sample_col not in df_clr.columns:
            raise ValueError(f"CLR data must contain '{sample_col}' column")

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

        # Convert to JSON for storage
        merged_json = df_merged.to_pandas().to_json(orient="split")

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
