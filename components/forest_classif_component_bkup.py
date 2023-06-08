import base64
import datetime
import io
from io import BytesIO
import dash # Biblioteca principal de Dash.
# from msilib.schema import Component
from dash import dcc, html, Input, Output, callback# Módulo de Dash para acceder a componentes interactivos y etiquetas de HTML.
from dash.dependencies import Input, Output, State # Dependencias de Dash para la implementación de Callbacks.
import dash_bootstrap_components as dbc # Biblioteca de componentes de Bootstrap en Dash para el Front-End responsive.
from components import home_component, tree_classif_component, tree_regre_component, forest_classif_component, forest_regre_component

import pathlib
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



#---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def classforest_card():
    """
    :retorna: Un div que contiene la explicación del módulo de Bosque Aleatorio: Clasificación.

    """

    return html.Div(

        # ID del div.
        id="classforest-card",

        # Elementos hijos del div "eda-card".
        children=[
            html.H5("Mining Analytics"), # Título de página.
            html.H3("Bosque Aleatorio: Clasificación"), # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div(
                id="intro",
                children="Explicación bosque clasificacion"
                ,
            ),
            # Texto secundario de explicacion.
            html.Div(
                id="intro2",
                children = "En esta sección podrás llevar a cabo este procedimiento de forma automatizada cargando uno de los datasets de prueba, o bien, cargando tu propio dataset."
            ),

            # Muestra una figura de exploración (GIF de lupa)
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[
                    html.Img(
                        id="eda",
                        src="/assets/eda.gif",
                        style = {'width': '25em', 'height': '15em'}
                    )
                ]
            ),

        ],

    )

dropdown_options = [
    {'label': 'Dataset 1', 'value': 'assets/dt1.csv'},
    {'label': 'Dataset 2', 'value': 'assets/dt2.csv'},
    {'label': 'Dataset 3', 'value': 'assets/dt3.csv'}
]


# Contenedor principal de la página en un Div.
forest_classif_component.layout = html.Div(
    id = "page-content",
    children=[

        # Contenido principal de la aplicación: se divide en 2 columnas: una con contenido explicativo y otra para elementos interactivos.
        html.Div(

            className="row",
            children=[

                # Columna a la izquierda: invoca a description_card para mostrar el texto explicativo de la izquierda.
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[classforest_card()],
                ),
                # Columna de la derecha: parte de la página pensada para mostrar elementos interactivos en la página principal.
                html.Div(
                    id = "right-column",
                    className="four columns",
                    children = html.Div([

                        html.H4("Carga o elige el dataset para iniciar la regresión con Árboles de Decisión", className= "text-upload"),

                        # Muestra el módulo de carga del dataset.
                        dcc.Upload(
                        id = 'upload-data-classforest',
                        children = html.Div([
                            'Arrastra aquí el archivo en formato CSV o selecciónalo'
                        ]),

                    # Por limitación de Dash estos elementos de carga no se pueden modificar desde la hoja de estilos y se debe hacer CSS inline.
                    style = {
                        'font-family': 'Acumin',
                        'width': '50%',
                        'height': '100%',
                        'lineHeight': '60px',
                        'borderWidth': '2px',
                        'borderStyle': 'solid',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '2em auto',
                        'display': 'grid',
                        'justify-content': 'center',
                        'align-items': 'center',
                        'flex-direction': 'column',
                        #'borderColor': '#2196F3',
                        'background-color': '#fEfEfE'
                    },
                    multiple = True,
                    accept = '.csv'
                ),

                html.Div(
                        children = "O selecciona un dataset predeterminado aquí",
                        style = {
                        'font-family': 'Acumin',
                        'width' : '100%',
                        'text-align': 'center'
                    }
                    ),

                    # Muestra el módulo de carga del dataset.
                    dcc.Dropdown(
                    id='upload-data-static-classforest',
                    options = dropdown_options,
                    value = dropdown_options[0]['value'],
                    className='my-dropdown'
                    ),

                html.Hr(),
                html.Div(id = 'output-data-upload-classforest'),
                ]),
                ),
                #html.Div(id = 'output-dataset-upload'),
            ],
        ),
    ],
)


def parse_contents(contents, filename, date):

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
        # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            return classforest(df, filename)
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return classforest(df, filename)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


def classforest(df, filename):
    """
    retorna: modelo de regresión usando un árbol de decisión regresor para la generación de pronósticos y valores siguientes en series de tiempo.

    """
    # Preparación de variables para su despliegue.

    # Div de visualización en el layout.
    return html.Div([

        dbc.Alert('El archivo cargado es: {}'.format(filename), color="success"),
        # Solo mostramos las primeras 5 filas del dataframe, y le damos estilo para que las columnas se vean bien
        dash_table.DataTable(
            data=df.to_dict('records'),
            page_size=8,
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=True,
            cell_selectable=True,
            editable=True,
            row_selectable='multi',
            columns=[{'name': i, 'id': i, "deletable":True} for i in df.columns],
            style_table={'height': '300px', 'overflowX': 'auto'},
        ),
        
        html.Hr(),  # Línea horizontal

    ])

@callback(Output('output-data-upload-classforest', 'children'),
              [Input('upload-data-classforest', 'contents'),
               Input('upload-data-static-classforest', 'value')],
              [State('upload-data-classforest', 'filename'),
               State('upload-data-classforest', 'last_modified')])
def update_output(list_of_contents, selected_file, list_of_names, list_of_dates):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    if ctx.triggered[0]['prop_id'] == 'upload-data-classforest.contents':
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children
    elif ctx.triggered[0]['prop_id'] == 'upload-data-static-classforest.value':
        df = pd.read_csv(selected_file)
        return classforest(df, selected_file)