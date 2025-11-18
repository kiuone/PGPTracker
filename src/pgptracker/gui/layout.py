# src/pgptracker/gui/layout.py

import dash_bootstrap_components as dbc
from dash import dcc, html
import dash_ag_grid as dag
from pgptracker.gui import ids, components


def create_layout() -> dbc.Container:
    """
    Create the main application layout.

    Returns:
        dbc.Container with complete app structure
    """
    return dbc.Container(
        [
            # Hidden stores for data
            dcc.Store(id=ids.STORE_DATA_N_D),
            dcc.Store(id=ids.STORE_METADATA_COLS),
            dcc.Store(id=ids.STORE_FEATURE_COLS),
            dcc.Store(id=ids.STORE_RAW_METADATA),
            dcc.Store(id=ids.STORE_RAW_CLR_DATA),

            # Header
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H2("PGPTracker Stage 2 Data Explorer", className="mb-0"),
                            html.P(
                                "Interactive exploration of CLR-transformed feature tables",
                                className="text-muted mb-0"
                            )
                        ],
                        className="p-3 bg-primary text-white"
                    ),
                    width=12
                ),
                className="mb-4"
            ),

            # Sidebar
            components.create_sidebar(),

            # Main content area (offset for sidebar)
            html.Div(
                [
                    dbc.Tabs(
                        id=ids.TABS_MAIN,
                        children=[
                            dbc.Tab(
                                label="Data Explorer",
                                tab_id="tab-explorer",
                                children=[
                                    html.Div(
                                        [
                                            html.H4("Joined Data Table", className="mt-3 mb-3"),
                                            html.P(
                                                "Interactive table showing merged CLR data and metadata. "
                                                "Use filters and sorting to explore the data.",
                                                className="text-muted"
                                            ),
                                            dag.AgGrid(
                                                id=ids.TABLE_DATA_EXPLORER,
                                                columnDefs=[],
                                                rowData=[],
                                                defaultColDef={
                                                    "filter": True,
                                                    "sortable": True,
                                                    "resizable": True,
                                                    "minWidth": 100
                                                },
                                                dashGridOptions={
                                                    "pagination": True,
                                                    "paginationPageSize": 50
                                                },
                                                style={"height": "600px"}
                                            )
                                        ],
                                        className="p-3"
                                    )
                                ]
                            ),

                            dbc.Tab(
                                label="Interactive Plots",
                                tab_id="tab-plots",
                                children=[
                                    html.Div(
                                        [
                                            html.H4("Interactive Visualization", className="mt-3 mb-3"),

                                            components.create_plot_controls(),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.H5("Boxplot: Feature by Group"),
                                                            dcc.Graph(
                                                                id=ids.GRAPH_BOXPLOT,
                                                                config={"displayModeBar": True}
                                                            )
                                                        ],
                                                        md=6
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.H5("Scatter Plot"),
                                                            dcc.Graph(
                                                                id=ids.GRAPH_SCATTER,
                                                                config={"displayModeBar": True}
                                                            )
                                                        ],
                                                        md=6
                                                    )
                                                ]
                                            )
                                        ],
                                        className="p-3"
                                    )
                                ]
                            )
                        ],
                        active_tab="tab-explorer"
                    )
                ],
                style={
                    "margin-left": "340px",
                    "padding": "20px"
                }
            )
        ],
        fluid=True,
        style={"padding": "0"}
    )
