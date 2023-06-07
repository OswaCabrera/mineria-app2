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

external_stylesheets = [dbc.themes.UNITED, "/assets/styles.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.title = 'MineriApp'
app.config.suppress_callback_exceptions = True

navbar = dbc.NavbarSimple(
    className="sticky-top bg-dark",
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
                dbc.DropdownMenuItem("EDA", href="/"),
                dbc.DropdownMenuItem("PCA", href="/"),
                dbc.DropdownMenuItem("Arboles", href="/"),
                dbc.DropdownMenuItem("Bosques", href="/"),
            ],
            nav=True,
            in_navbar=True,
            label="Acerca de",
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
    dbc.Container(id="page-content", fluid=True)
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