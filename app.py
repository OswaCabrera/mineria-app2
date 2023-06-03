import base64
import datetime
import io
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash_bootstrap_components._components.Container import Container

from components import home_component, tree_classif_component, tree_regre_component, forest_classif_component, forest_regre_component

from components.eda_component import Eda
from components.pca_component import Pca_Propio

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL, "/assets/styles.css"])
app.title = 'MineriApp'


navbar = dbc.NavbarSimple(
    className="sticky-top",
    children=[
        dbc.NavItem(dbc.NavLink("Inicio", href="/")),
        dbc.NavItem(dbc.NavLink("EDA", href="/eda")),
        dbc.NavItem(dbc.NavLink("PCA", href="/pca")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("De clasificación", href="/arboles_clasificacion"),
                dbc.DropdownMenuItem("De regresión", href="/arboles_regresion"),
            ],
            nav=True,
            in_navbar=True,
            label="Árboles",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("De clasificación", href="/bosques_clasificacion"),
                dbc.DropdownMenuItem("De regresión", href="/bosques_regresion"),
            ],
            nav=True,
            in_navbar=True,
            label="Bosques Aleatorios",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Inicia Sesión", href="/login"),
                dbc.DropdownMenuItem("Regístrate", href="/register"),
            ],
            nav=True,
            in_navbar=True,
            label="Autenticación",
        ),
    ],
    brand="MineriApp",
    brand_href="/",
    color="primary",
    dark=True,
)

# Diccionario de rutas y componentes correspondientes
routes = {
    "/": home_component.layout,
    # "/eda": Eda.layout,
    # "/pca": Pca_Propio.layout,
    "/arboles_clasificacion": tree_classif_component.layout,
    "/arboles_regresion": tree_regre_component.layout,
    "/bosques_clasificacion": forest_classif_component.layout,
    "/bosques_regresion": forest_regre_component.layout,
}

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    html.Div(id="page-content", style={'margin-top': '60px'})
])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    # Obtener el componente correspondiente a la ruta desde el diccionario
    component = routes.get(pathname)

    if component is None:
        return html.Div(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ],
            className="p-3 bg-light rounded-3",
        )
    return component


if __name__ == '__main__':
    app.run_server(debug=True)
