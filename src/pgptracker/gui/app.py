# src/pgptracker/gui/app.py

import dash
import dash_bootstrap_components as dbc
from pgptracker.gui.layout import create_layout
from pgptracker.gui import callbacks


# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP
    ],
    suppress_callback_exceptions=True,
    title="PGPTracker Stage 2 Explorer"
)

# Expose server for deployment
server = app.server

# Set layout
app.layout = create_layout()

# Callbacks are automatically registered via imports


def run_app(debug=True, port=8050):
    """
    Run the Dash application.

    Args:
        debug: Enable debug mode (default: True)
        port: Port number to run the server (default: 8050)
    """
    app.run_server(debug=debug, port=port, host="0.0.0.0")


if __name__ == "__main__":
    run_app()
