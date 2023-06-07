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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt
# Bibliotecas adicionales para Bosques Aleatorios
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf # Para descargar un dataframe a partir de un ticker
from sklearn.tree import export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import uuid
import graphviz

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



#---------------------------------------------------Definición de funciones para el front--------------------------------------------------------#
def regforest_card():
    """
    :retorna: Un div que contiene la explicación del módulo de Bosque Aleatorio: Regresión.

    """

    return html.Div(

        # ID del div.
        id="regforest-card",

        # Elementos hijos del div "regforest-card".
        children=[
            html.H5("Mining Analytics"), # Título de página.
            html.H3("Bosque Aleatorio: Regresión"), # Subtítulo.
            # Texto que explica la temática de la página web.
            html.Div(
                id="intro",
                children="Los árboles de decisión representan uno de los algoritmos de aprendizaje supervisado más utilizados, los cuales soportan tanto valores numéricos como nominales. Para esto, se construye una estructura jerárquica que divide los datos en función de condicionales."
                ,
            ),

            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[
                    html.Img(
                        id="tree1",
                        src="https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/50/f9/ICLH_Diagram_Batch_03_27-RandomForest.component.xl.ts=1679336476850.png/content/adobe-cms/us/en/topics/random-forest/jcr:content/root/table_of_contents/body/simple_narrative/image",
                        style = {'width': '25em', 'height': '15em'}
                    )
                ]
            ),

            # Texto secundario de explicacion.
            html.Div(
                id="intro2",
                children = "En esta sección podrás llevar a cabo este procedimiento de forma automatizada cargando tu propio dataset o cargando los históricos de algún activo (Stock, Criptos, etc.) recuperados directamente desde Yahoo Finance."
            ),

            # Muestra una GIF
            html.Div(
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'height': '20em'},
                children=[
                    html.Img(
                        id="eda",
                        src="https://miro.medium.com/v2/resize:fit:960/1*w-b0xHDoUsCcwx4nY3x5Og.gif",
                        style = {'width': '80%', 'height': '80%'}
                    )
                ]
            ),

        ],

    )


#Contenedor principal de la página en un Div
forest_regre_component.layout = html.Div(
    id="page-content",
    children=[
        # El contenido se divide en 2 columnas: descripción | resultados
        html.Div(
            className="row",
            children=[
                #Columna izquierda: para la descripción
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[regforest_card()],
                ),
                #Columa derecha: para los resultados
                html.Div(
                    id="right-column",
                    className="four columns",
                    children=html.Div(
                        [
                            html.H4("Carga el dataset para iniciar la regresión con Bosques Aleatorios", className="text-upload"),
                            # Muestra el módulo de carga
                            dcc.Upload(
                                id='upload-data-regforest',
                                children=html.Div(
                                    [
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ],
                                ),
                            style={
                                'font-family':'Acumin',
                                'width': '50%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'margin': '2em auto',
                                'cursor': 'pointer',
                            },
                            multiple=True,
                            accept='.csv',
                            className="drag"
                            ),
                            # Cargar dataframe de yfinance por medio de un ticker
                            html.P(
                                "O utiliza como datos de entrada los históricos de algún activo (Stocks, Criptomonedas o Index)",
                                style = {
                                    'text-align': 'center',
                                    'font-size':'18px',
                                }
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        id='ticker-input',
                                        placeholder='Ingrese el ticker aquí',
                                        style={
                                            'font-size':'16px',
                                        }
                                    ),
                                    dbc.Button(
                                        'Enviar',
                                        id='submit-ticker',
                                        n_clicks=0,
                                        color='primary',
                                        style={
                                            'text-transform':'none',
                                            'font-size':'16px',
                                        }
                                    ),
                                ],
                                style={
                                    'width':'25%',
                                    'margin': '20px auto',
                                }
                            ),
                            html.Div(id = 'output-data-upload-regforest'),
                        ],
                    ),
                ),
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
                io.StringIO(decoded.decode('utf-8')), index_col=None)
            return regforest(df, filename, df.columns)
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return regforest(df, filename, df.columns)
    except Exception as e:
        print(e)
        return html.Div([
            dbc.Alert('There was an error processing this file.', color="danger")
        ])

def get_yahoo_finance_data(ticker):
    """
    retorna: dataset con el histórico del ticker especificado.
    """
    df = yf.download(ticker, period="max", interval = "1d")
    return df

def create_yahoo_finance_chart(df, filename):
    # Crea el gráfico de Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines+markers', name='Open', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df.index, y=df['High'], mode='lines+markers', name='High', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Low'], mode='lines+markers', name='Low', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines+markers', name='Close', line=dict(color='green')))

    fig.update_layout(
        title=f"Histórico de {filename}",
        xaxis_title="Fecha",
        yaxis_title="Precio de las acciones",
        legend_title="Precios",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        ),
        yaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        )
    )
    return fig

def regforest(df, filename, columns):
    """
    retorna: modelo de regresión usando un bosque aleatorio regresor regresor para la generación de pronósticos y valores siguientes en series de tiempo.

    """
    # Se hace global el dataframe
    global global_df
    global_df = df

    # Preparación de variables para su despliegue.
    fig = create_yahoo_finance_chart(df, filename)

    # Div de visualización en el layout.
    return html.Div(
        [
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

            dcc.Graph(
                id='yahoo-finance-chart',
                figure = fig
            ),

            html.H3(
                "Elección de Variables Predictoras y Dependiente",
                style={'margin-top': '30px'}
            ),

            html.Div(
                html.P("Selecciona de la siguiente lista las variables que deseas elegir como predictoras y tu variable target para realizar la regresión.")
            ),
            html.Div(
                children=[
                    dcc.Store(id="original-options", data=[{'label': col, 'value': col} for col in df.columns]),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Row(
                                        html.Div(
                                            [
                                                dbc.Badge("ⓘ Variables predictoras", color="primary",
                                                        id="tooltip-predictoras", style={"cursor": "pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%", 'font-size':'16px'},
                                                        ),
                                                dbc.Tooltip(
                                                    "Características o atributos que se utilizan como entrada para predecir o estimar el valor de la variable objetivo o variable regresora.",
                                                    target="tooltip-predictoras", style={"font-size":"10px"},
                                                ),
                                            ],
                                            style={"height": "50px", "padding": "0"},
                                        ),
                                    ),
                                    dbc.Row(
                                        dbc.Checklist(
                                            id='select-predictors',
                                            options = [{'label': col, 'value': col} for col in df.columns],
                                            style={"font-size": "14px", "display": "grid", "justify-items": "start", 'border':'1px solid #e1e1e1', 'border-radius':'5px', 'background-color':'white'}
                                        ),
                                        style={"height": "auto"}
                                    ),
                                ],
                                class_name="me-3"
                            ),
                            dbc.Col(
                                [
                                    dbc.Row(
                                        html.Div(
                                            [
                                                dbc.Badge("ⓘ Variable regresora", color="primary",
                                                        id="tooltip-regresora", style={"cursor": "pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%", 'font-size':'16px'},
                                                        ),
                                                dbc.Tooltip(
                                                    "Es la variable objetivo que se intenta predecir o estimar utilizando las variables predictoras como entrada.",
                                                    target="tooltip-regresora", style={"font-size":"10px"}
                                                ),
                                            ],
                                            style={"height": "50px", "padding": "0"},
                                        ),
                                    ),
                                    dbc.Row(
                                        dbc.Checklist(
                                            id='select-regressor',
                                            options = [{'label': col, 'value': col} for col in df.columns],
                                            style={"font-size": "14px", "display": "grid", "justify-items": "start", 'border':'1px solid #e1e1e1', 'border-radius':'5px', 'background-color':'white'}
                                        ),
                                    ),
                                ],
                                class_name="me-3"
                            ),
                        ],
                        style={"justify-content": "between", "height": "100%"}
                    ),

                     html.H3(
                        "Generación del Modelo"
                    ),
                    html.P(
                        "Una vez que hayas identificado las variables predictoras y la variable objetivo, el siguiente paso consiste en configurar los parámetros necesarios para que el modelo funcione correctamente."
                    ),
                    html.P(
                        "Al terminar, presiona sobre el botón rojo para observar los resultados."
                    ),
                    dbc.Alert(
                        "ⓘ Es posible dejar vacíos los campos que controlan los parámetros de los árboles de decisión que se utilizarán en el bosque. Sin embargo, es importante tener en cuenta que esto puede aumentar el consumo de recursos y potencialmente llevar a un modelo sobreajustado.", color="warning", style={"font-size": "10px", 'margin-bottom': '0px'}
                    ),

                    # Div para los parámetros del Bosque
                    html.Div(
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                html.Div(
                                                    [
                                                        dbc.Badge("ⓘ Profundad máxima de los árboles", color="primary",
                                                            id="tooltip-depht", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                                        ),
                                                        dbc.Tooltip(
                                                            [
                                                                dcc.Markdown('''
                                                                    **📏 Max Depth:**  
                                                                    Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.
                                                                ''', style={'text-align': 'left'}),
                                                            ],
                                                            target="tooltip-depht", placement="left", style={"font-size":"10px"},
                                                        ),
                                                    ],
                                                    style={"height":"50px", "padding": "0"},
                                                ),
                                            ),
                                            dbc.Row(
                                                dbc.Input(
                                                    id='input-max-depth',
                                                    type='number',
                                                    placeholder='None',
                                                    min=1,
                                                    step=1,
                                                    style={"font-size": "medium"}
                                                ),
                                                style={"height":"50px"}
                                            ),
                                        ],
                                        class_name="me-3", style={'flex':'1 0 25%'}
                                    ),

                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                html.Div(
                                                    [
                                                        dbc.Badge("ⓘ Muestras mínimas de división", color="primary",
                                                            id="tooltip-div", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                                        ),
                                                        dbc.Tooltip(
                                                            [
                                                                dcc.Markdown('''
                                                                    **✂️ Min Samples Split:**  
                                                                    Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.
                                                                ''', style={'text-align': 'left'}),
                                                            ],
                                                            target="tooltip-div", placement="left", style={"font-size":"10px"},
                                                        ),
                                                    ],
                                                    style={"height":"50px", "padding": "0"},
                                                ),
                                            ),
                                            dbc.Row(
                                                dbc.Input(
                                                    id='input-min-samples-split',
                                                    type='number',
                                                    placeholder='None',
                                                    min=1,
                                                    step=1,
                                                    style={"font-size": "medium"}
                                                ),
                                                style={"height":"50px"}
                                            ),
                                        ],
                                        class_name="me-3", style={'flex':'1 0 25%'}
                                    ),

                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                html.Div(
                                                    [
                                                        dbc.Badge("ⓘ Muestras mínimas por hoja", color="primary",
                                                            id="tooltip-leaf", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                                        ),
                                                        dbc.Tooltip(
                                                            [
                                                                dcc.Markdown('''
                                                                    **🍃 Min Samples Leaf:**  
                                                                    Indica la cantidad mínima de datos que debe tener un nodo hoja.
                                                                ''', style={'text-align': 'left'}),
                                                            ],
                                                            target="tooltip-leaf", placement="left", style={"font-size":"10px"},
                                                        ),
                                                    ],
                                                    style={"height":"50px", "padding": "0"},
                                                ),
                                            ),
                                            dbc.Row(
                                                dbc.Input(
                                                    id='input-min-samples-leaf',
                                                    type='number',
                                                    placeholder='None',
                                                    min=1,
                                                    step=1,
                                                    style={"font-size": "medium"}
                                                ),
                                                style={"height":"50px"}
                                            ),
                                        ],
                                        class_name="me-3", style={'flex':'1 0 25%'}
                                    ),

                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                html.Div(
                                                    [
                                                        dbc.Badge("ⓘ Tamaño de la muestra", color="primary",
                                                            id="tooltip-sample", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                                        ),
                                                        dbc.Tooltip(
                                                            [
                                                                dcc.Markdown('''
                                                                    **Tamaño de la muestra**  
                                                                    Indica el tamaño del conjunto de datos original que se utilizará para verificar el rendimiento del modelo. Por defecto se utiliza una división '80/20' en la que el 80% de los datos originales se utilizan para entrenar el modelo y el 20% restante para validarlo.
                                                                ''', style={'text-align': 'left'}),
                                                            ],
                                                            target="tooltip-sample", placement="left", style={"font-size":"10px"},
                                                        ),
                                                    ],
                                                    style={"height":"50px", "padding": "0"},
                                                ),
                                            ),
                                            dbc.Row(
                                                dbc.Input(
                                                    id='input-test-size',
                                                    type='number',
                                                    placeholder='None',
                                                    value=0.2,
                                                    min=0.2,
                                                    max = 0.5,
                                                    step=0.1,
                                                    style={"font-size": "medium"}
                                                ),
                                                style={"height":"50px"}
                                            ),
                                        ],
                                        class_name="me-3", style={'flex':'1 0 25%'}
                                    ),

                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                html.Div(
                                                    [
                                                        dbc.Badge("ⓘ Número de Estimadores", color="primary",
                                                            id="tooltip-estimators", style={"cursor":"pointer", "display": "flex", "align-items": "center", "justify-content": "center", "height": "100%"}
                                                        ),
                                                        dbc.Tooltip(
                                                            [
                                                                dcc.Markdown('''
                                                                    **🌳🌳 Número de Estimadores:**  
                                                                    Indica el número de árboles que va a tener el bosque aleatorio. Normalmente,
                                                                    cuantos más árboles es mejor, pero a partir de cierto punto deja de mejorar y se vuelve más lento.
                                                                    El valor por defecto es 100 árboles.
                                                                ''', style={'text-align': 'left'}),
                                                            ],
                                                            target="tooltip-estimators", placement="left", style={"font-size":"10px"},
                                                        ),
                                                    ],
                                                    style={"height":"50px", "padding": "0"},
                                                ),
                                            ),
                                            dbc.Row(
                                                dbc.Input(
                                                    id='input-estimators',
                                                    type='number',
                                                    value=100,
                                                    min=100,
                                                    max=200,
                                                    step=10,
                                                    style={"font-size": "medium"}
                                                ),
                                                style={"height":"50px"}
                                            ),
                                        ],
                                        class_name="me-3", style={'flex':'1 0 25%'}
                                    ),
                                ],
                                style={"justify-content": "between", "height": "100%"}
                            ),
                        ],
                        style={"font-size":"20px", "margin":"20px 0"}
                    ),
                    html.Div(
                        children=
                        [
                            dbc.Button(
                                "Generar Bosque", id="submit-button", color="danger", style={"width":"40%"},
                            ),
                        ],
                        style={"display": "flex", "justify-content": "center"},
                    ),
                    html.Div(id="output-data-regforest", style = {"margin-top": "1em"}),
                ],
            ),
        ]
    )

@callback(Output('output-data-upload-regforest', 'children'),
          [Input('upload-data-regforest', 'contents'),
           Input('submit-ticker', 'n_clicks')],
          [State('upload-data-regforest', 'filename'),
           State('upload-data-regforest', 'last_modified'),
           State('ticker-input', 'value')])
def update_output(list_of_contents, submit_ticker_clicks, list_of_names, list_of_dates, ticker):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    if ctx.triggered[0]['prop_id'] == 'upload-data-regforest.contents':
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children
    elif ctx.triggered[0]['prop_id'] == 'submit-ticker.n_clicks':
        if ticker:
            df = get_yahoo_finance_data(ticker)
            return regforest(df, ticker, df.columns)
        else:
            return html.Div([
                dbc.Alert('ⓘ Primero escribe un Ticker, por ejemplo: "AAPL" (Apple), "MSFT" (Microsoft), "GOOGL" (Google), etc. ', color="danger")
            ])

# Y_TEST vs Y_PREDICTED : Chart
def create_comparison_chart(Y_test, Y_Predicted):
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=Y_test.flatten(), mode='lines', name='Real', marker=dict(color='red', symbol='cross')))
    fig.add_trace(go.Scatter(y=Y_Predicted, mode='lines', name='Estimado', marker=dict(color='green', symbol='cross')))

    fig.update_layout(
        title="Pronóstico de las acciones",
        xaxis_title="Fecha",
        yaxis_title="Precio de las acciones",
        legend_title="Valores",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        ),
        yaxis=dict(
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        )
    )
    return fig

# GENERACIÓN DEL BOSQUE ALEATORIO: Con parámetros
def generate_forest(X_train, X_test, Y_train, Y_test, max_depth, min_samples_split, min_samples_leaf, estimadores):
    reg_forest = RandomForestRegressor(
        random_state=0,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_estimators=estimadores
    )
    reg_forest.fit(X_train, Y_train)
    Y_Predicted = reg_forest.predict(X_test)
    comparison_df = pd.DataFrame({"Y_Real": Y_test.flatten(), "Y_Pronosticado": Y_Predicted})

    # Devuelve también el árbol de regresión y sus parámetros
    forest_parameters = {
        "criterion": reg_forest.criterion,
        "feature_importances": reg_forest.feature_importances_,
        "MAE": mean_absolute_error(Y_test, Y_Predicted),
        "MSE": mean_squared_error(Y_test, Y_Predicted),
        "RMSE": mean_squared_error(Y_test, Y_Predicted, squared=False),
        "score": r2_score(Y_test, Y_Predicted),
    }
    return comparison_df, reg_forest, forest_parameters, Y_Predicted

# GENERACIÓN DEL BOSQUE: sin parámetros
def generate_forestS(X_train, X_test, Y_train, Y_test, estimadores):
    reg_forest = RandomForestRegressor(
        random_state=0,
        n_estimators=estimadores
    )
    reg_forest.fit(X_train, Y_train)
    Y_Predicted = reg_forest.predict(X_test)
    comparison_df = pd.DataFrame({"Y_Real": Y_test.flatten(), "Y_Pronosticado": Y_Predicted})

    # Devuelve también el árbol de regresión y sus parámetros
    forest_parameters = {
        "criterion": reg_forest.criterion,
        "feature_importances": reg_forest.feature_importances_,
        "MAE": mean_absolute_error(Y_test, Y_Predicted),
        "MSE": mean_squared_error(Y_test, Y_Predicted),
        "RMSE": mean_squared_error(Y_test, Y_Predicted, squared=False),
        "score": r2_score(Y_test, Y_Predicted),
    }
    return comparison_df, reg_forest, forest_parameters, Y_Predicted


def create_input_form(predictors):
    input_form = []
    for predictor in predictors:
        input_form.append(
            html.Div(
                [
                    html.Label(predictor),
                    dcc.Input(
                        type="number",
                        id=f"input-{predictor}",  # Agrega el atributo id a la entrada
                    ),
                ],
                className="form-group",
            )
        )
    return input_form


@callback(Output("input-form-forest", "children"), Input("submit-button", "n_clicks"))
def update_input_form(n_clicks):
    if n_clicks is None:
        return ""
    return create_input_form(global_predictors)

def predict_new_values(reg_tree, predictors, input_values):
    input_data = pd.DataFrame(input_values, columns=predictors)
    prediction = reg_tree.predict(input_data)
    return prediction

@callback(
    Output("prediction-result-regforest", "children"),
    Input("predict-button", "n_clicks"),
    State("input-form-forest", "children"),
)
def show_prediction(n_clicks, input_form):
    if n_clicks is None or input_form is None:
        return ""

    input_values = {}
    all_states = dash.callback_context.states
    for child in input_form:
        label = child['props']['children'][0]['props']['children']
        if label in global_predictors:
            input_id = child['props']['children'][1]['props']['id']
            try:
                # Agrega el id del campo de entrada a all_states
                all_states[f"{input_id}.value"] = child['props']['children'][1]['props']['value']
                input_values[label] = float(all_states[f"{input_id}.value"])
            except KeyError:
                print(f"Error: No se encontró la clave '{input_id}.value' en dash.callback_context.states")
                print("Valores de entrada:", input_values)
                print("Claves presentes en dash.callback_context.states:", dash.callback_context.states.keys())

    prediction = predict_new_values(global_reg_forest, global_predictors, [input_values])
    return f"La predicción con base en los valores introducidos es: {prediction[0]:.2f}"


# CALLBACK PARA CALCULAR EL ÁRBOL Y GENERAR LAS TABS PARA MOSTRAR LOS RESULTADOS
@callback(
    Output("output-data-regforest", "children"),
    Input("submit-button", "n_clicks"),
    State("select-predictors", "value"),
    State("select-regressor", "value"),
    State("input-max-depth", "value"),
    State("input-min-samples-split", "value"),
    State("input-min-samples-leaf", "value"),
    State("input-test-size", "value"),
    State("input-estimators", "value")
)
def create_model(n_clicks, predictors, regressor, max_depth, min_samples_split, min_samples_leaf, test_size, estimators):
    global global_df
    global global_predictors
    global global_regressor

    if n_clicks is None:
        return ""

    if predictors is None or regressor is None:
        return "Por favor, seleccione las variables predictoras y la variable regresora."

    if global_df is None:
        return "No se ha cargado ningún dataset."

    global_predictors = predictors
    global_regressor = regressor
    print(global_predictors)
    print(global_regressor)
    print(global_df)
    # Resto del código

    X = np.array(global_df[global_predictors])
    global global_X 
    global global_Y
    global_X = X
    print(X)
    print(global_df[global_regressor])
    Y = np.array(global_df[global_regressor])
    global_Y = Y 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = 0, shuffle = True)
    
    if max_depth is None and min_samples_split is None and min_samples_leaf is None:
        comparison_df, reg_forest, forest_parameters, Y_Predicted = generate_forestS(
            X_train, X_test, Y_train, Y_test, estimators
        )
    else:
        comparison_df, reg_forest, forest_parameters, Y_Predicted = generate_forest(
            X_train, X_test, Y_train, Y_test, max_depth, min_samples_split, min_samples_leaf, estimators
        )

    global global_reg_forest 
    global_reg_forest = reg_forest
    comparison_chart = create_comparison_chart(Y_test, Y_Predicted)

    
    comparison_table = dash_table.DataTable(
        data=comparison_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in comparison_df.columns],
        style_table={'height': '300px', 'overflowX': 'auto'},
    )

    # Crea una tabla con los parámetros del bosque
    parameters_list = [
        {"parameter": key, "value": value}
        for key, values in forest_parameters.items()
        for value in (values if isinstance(values, (list, np.ndarray)) else [values])
    ]
    parameters_df = pd.DataFrame(parameters_list)
    parameters_table = dash_table.DataTable(
        data=parameters_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in parameters_df.columns],
        style_table={'overflowX': 'auto', "border": "none"},
    )

    importance_df = pd.DataFrame({'Variable': predictors, 'Importancia': forest_parameters['feature_importances']}).sort_values('Importancia', ascending=False)
    importance_table = dash_table.DataTable(
        data=importance_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in importance_df.columns],
        style_table={'overflowX': 'auto'},
    )

    new_forecasts_section = html.Div(
        [
            html.H3("Generar nuevos pronósticos"),
            html.P("Introduce los valores de las variables predictoras:"),
            html.Div(id="input-form-forest"),
            html.Button("Predecir", id="predict-button", className="mt-3"),
            html.Div(id="prediction-result-forest", className="mt-4"),
        ],
        className="mt-4",
    )


    return html.Div(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(
                            children=[
                                    html.H5("Los parámetros del bosque generado son los siguientes:"),
                                    parameters_table,
                                    html.Br(),
                                    html.H5("Se han obtenido los siguiente valores de pronóstico en el set de entrenamiento, los cuales se comparan con los valores reales:"),
                                    comparison_table,
                                    html.Br(),
                                    html.H5("A continuación se especifica la importancia numérica [0-1] de las variables predictoras en el modelo construido:"),
                                    importance_table,

                            ],
                            label="Parámetros del Bosque Aleatorio", tab_id="tab-1", tab_style={"width": "auto"}),


                        dbc.Tab(
                            children=[

                                    html.H5("El siguiente gráfico permite comparar los valores estimados por el Bosque Aleatorio contra los valores reales de prueba:"),
                                    dcc.Graph(figure=comparison_chart),

                            ],
                            label="Comparación entre Valores reales y Predecidos", tab_id="tab-3", tab_style={"width": "auto"}
                        ),

                        dbc.Tab(
                            children=[

                                    new_forecasts_section

                            ],
                            label="Nuevos Pronósticos", tab_id="tab-4", tab_style={"width": "auto"}

                        ),

                    ],
                    id="tabs",
                    active_tab="tab-1",
                    style={"margin-top": "45px"}
                ),
            ],
        )