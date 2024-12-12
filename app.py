# Import libraries
import dash_bootstrap_components as dbc
import dash
from dash import Dash

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
           suppress_callback_exceptions=True, prevent_initial_callbacks=True)

server = app.server

from assets.nav import _nav # Import shared components

# App Layout
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([_nav], width=2),
        dbc.Col([
            dbc.Row([dash.page_container])
        ], width=10),
    ]),
], fluid=True)


# Run App
if __name__ == '__main__':
    app.run_server()