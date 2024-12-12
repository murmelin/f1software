# Import libraries
from dash import html
import dash_bootstrap_components as dbc
import dash

_nav = dbc.Container([
    
	dbc.Row([dbc.Col([html.H3([' '])], width = 8)]),

        # Header
        html.Div(children="Italian GP 2020: Race Session", className="big-title"),

        # NavBar
	dbc.Row([
        dbc.Nav(
	        [dbc.NavLink(page["name"], active='exact', href=page["path"]) for page in dash.page_registry.values()],
	        vertical=True, pills=True, class_name='my-nav')
    ])
])
