# src/pgptracker/gui/components.py

import dash_bootstrap_components as dbc
from dash import dcc, html
from pgptracker.gui import ids


def create_upload_card(
    card_title: str,
    upload_id: str,
    upload_text: str
) -> dbc.Card:
    """
    Create a reusable upload card component.

    Args:
        card_title: Title displayed at top of card
        upload_id: Unique ID for the dcc.Upload component
        upload_text: Text displayed in upload area

    Returns:
        dbc.Card containing upload component
    """
    return dbc.Card(
        [
            dbc.CardHeader(html.H5(card_title)),
            dbc.CardBody(
                [
                    dcc.Upload(
                        id=upload_id,
                        children=html.Div(
                            [
                                html.I(className="bi bi-cloud-upload me-2"),
                                upload_text
                            ],
                            className="d-flex align-items-center justify-content-center"
                        ),
                        style={
                            "width": "100%",
                            "height": "80px",
                            "lineHeight": "80px",
                            "borderWidth": "2px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "cursor": "pointer"
                        },
                        multiple=False,
                        max_size=-1
                    )
                ]
            )
        ],
        className="mb-3 sidebar-card"
    )


def create_sidebar() -> html.Div:
    """
    Create the left sidebar containing file upload components.

    Returns:
        html.Div containing sidebar layout
    """
    return html.Div(
        [
            html.H4("Data Input", className="mb-4", id="sidebar-title"),

            dcc.Loading(
                id="loading-metadata",
                type="circle",
                children=[
                    create_upload_card(
                        card_title="1. Upload Metadata",
                        upload_id=ids.UPLOAD_METADATA,
                        upload_text="Drag and drop or click to upload metadata.tsv"
                    )
                ]
            ),

            dcc.Loading(
                id="loading-clr",
                type="circle",
                children=[
                    create_upload_card(
                        card_title="2. Upload CLR Data",
                        upload_id=ids.UPLOAD_CLR_DATA,
                        upload_text="Drag and drop or click to upload clr_wide_N_D.tsv"
                    )
                ]
            ),

            dbc.Card(
                [
                    dbc.CardHeader(html.H5("3. Sample ID Column")),
                    dbc.CardBody(
                        [
                            html.P(
                                "Auto-detected or select manually if needed:",
                                className="small text-muted mb-2"
                            ),
                            dcc.Dropdown(
                                id=ids.DROPDOWN_SAMPLE_ID,
                                placeholder="Auto-detect or select...",
                                clearable=True
                            )
                        ]
                    )
                ],
                className="mb-3 sidebar-card"
            ),

            dcc.Loading(
                id="loading-merge-status",
                type="circle",
                children=[
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5("Upload Status")),
                            dbc.CardBody(id=ids.DIV_UPLOAD_STATUS)
                        ],
                        className="mb-3 sidebar-card"
                    )
                ]
            ),

            dcc.Loading(
                id="loading-data-summary",
                type="circle",
                children=[
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5("Data Summary")),
                            dbc.CardBody(id=ids.DIV_DATA_SUMMARY)
                        ],
                        className="sidebar-card"
                    )
                ]
            )
        ],
        id="sidebar-container"
        # Style is now controlled by update_sidebar_theme callback in callbacks.py
    )


def create_plot_controls() -> dbc.Card:
    """
    Create the plot control panel for Tab 2.

    Returns:
        dbc.Card containing plot control dropdowns
    """
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Plot Controls")),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Group By:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id=ids.DROPDOWN_GROUPBY,
                                        placeholder="Select metadata column",
                                        clearable=False
                                    )
                                ],
                                md=6
                            ),
                            dbc.Col(
                                [
                                    html.Label("Feature:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id=ids.DROPDOWN_FEATURE,
                                        placeholder="Select feature",
                                        clearable=False
                                    )
                                ],
                                md=6
                            )
                        ],
                        className="mb-3"
                    ),
                    html.Hr(),
                    html.H6("Scatter Plot Settings", className="mt-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("X-Axis:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id=ids.DROPDOWN_SCATTER_X,
                                        placeholder="Select X column",
                                        clearable=False
                                    )
                                ],
                                md=6
                            ),
                            dbc.Col(
                                [
                                    html.Label("Y-Axis:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id=ids.DROPDOWN_SCATTER_Y,
                                        placeholder="Select Y column",
                                        clearable=False
                                    )
                                ],
                                md=6
                            )
                        ]
                    )
                ]
            )
        ],
        className="mb-4"
    )
