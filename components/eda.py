import base64
import datetime
import io
import dash_bootstrap_components as dbc
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib         
import dash
import time
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.express as px
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
theme_change = ThemeChangerAIO(
    aio_id="theme",button_props={
        "color": "danger",
        "children": "SELECT THEME",
        "outline": True,
    },
    radio_props={
        "persistence": True,
    },
)

CONTENT_STYLE = {
    "margin-left": "32rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 30,
    "left": 0,
    "bottom": 0,
    "width": "30rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.H2("Empecemos por el análisis de datos", className="display-4"),
        html.Hr(),
        html.P(
            "¿Qué es el EDA?", className="lead"
        ),
        html.P(
            """Los científicos de datos utilizan el análisis exploratorio de datos (EDA) para analizar e investigar conjuntos de datos y
             resumir sus características principales, a menudo empleando métodos de visualización de datos. Ayuda a determinar la mejor manera 
             de gestionar las fuentes de datos para obtener las respuestas que necesita, lo que facilita que los científicos de datos descubran 
             patrones, detecten anomalías, prueben una hipótesis o verifiquen suposiciones."""
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'Black',
    'color': 'white',
    'padding': '6px'
}

layout = html.Div([
    html.H1('Análisis exploratorio de Datos', style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Arrastra y suelta tu archivo aquí o selecciona uno',
        ]),
        style={
            'width': '100%',
            'height': '100%',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'flex-direction': 'column'
        },
        multiple=True, # Allow multiple files to be uploaded
        accept='.csv, .txt, .xls, .xlsx' # Restrict to csv, txt, xls, xlsx files
    ),
    html.Div(id='output-data-upload'),

])

def parse_contents(contents, filename,date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global df
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'Uy lo siento, no me fue posible cargar tu documento. Aegurate que la extensión sea la correcta.'
        ])

    return html.Div([
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="loading-output-1")
        ),
        html.P('Estás trabajando con el archivo: {}. Si quieres cambiar de archivo vuelve a cargar otro'.format(filename)),
        # dbc.Alert('{}'.format(df.shape[0]).' Filas X {}'.format(df.shape[1]).' Columnas', color="info"),
        # Print the number of rows and columns in the same line
        html.H3("Tus datos son:" , style={'text-align': 'center'}),
        html.P(
            " {} Filas X {} Columnas.".format(df.shape[0], df.shape[1]),
            style={'text-align': 'center'}
        ),
        html.H3("Información acerca de tus variables:", style={'text-align': 'center'}),

        dash_table.DataTable(
            data=df.to_dict('records'),
            page_size=8,
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=True,
            editable=True,
            columns=[{'name': i, 'id': i, "deletable":True} for i in df.columns],
            style_table={'height': '300px', 'overflowX': 'auto'},
        ),
        
        html.H3("Estadísticas descriptivas de tus variables:", style={'text-align': 'center'}),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    # Primer columna: nombre de la columna y las demás columnas: nombre de las estadísticas (count, mean, std, min, 25%, 50%, 75%, max)
                                    html.Th('Variable'),
                                    html.Th('Tipo de dato'),
                                    html.Th('Total de datos'),
                                    html.Th('Valores nulos'),
                                    html.Th('Valores únicos'),
                                    html.Th('Dato más frecuente'),
                                    html.Th('Dato menos frecuente'),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(column), # Primera columna: nombre de la columna
                                        html.Td(
                                            str(df.dtypes[column]),
                                        ),
                                        # Count del tipo de dato (y porcentaje)
                                        html.Td(
                                            [
                                                html.P("{}".format(df[column].count())),
                                            ]
                                        ),
                                        html.Td(
                                            df[column].isnull().sum(),
                                        ),

                                        #Valores únicos
                                        html.Td(
                                            df[column].nunique(),
                                        ),

                                        # Top valores más frecuentes
                                        html.Td(
                                            [
                                                html.P("{}".format(df[column].value_counts().index[0])),
                                            ]
                                        ),

                                        # Top valores menos frecuentes
                                        html.Td(
                                            [
                                                html.P("{}".format(df[column].value_counts().index[-1])),
                                            ]
                                        ),
                                    ]
                                ) for column in df.dtypes.index
                            ]
                        )
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True,
                    # Texto centrado y tabla alineada al centro de la página
                    style={'textAlign': 'center', 'width': '100%'}
                )
            ,html.Br(),
                dbc.Table(
                    # Mostamos el resumen estadístico de las variables de tipo object, con su descripción a la izquierda
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    # Primer columna: nombre de la estadística (count, mean, std, min, 25%, 50%, 75%, max) y las demás columnas: nombre de las columnas (recorremos las columnas del dataframe)
                                    html.Th('Estadística'),
                                    *[html.Th(column) for column in df.describe().columns]

                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        # Recorremos el for para mostrar el nombre de la estadística a la izquierda de cada fila
                                        html.Td('count'),
                                        *[html.Td(df.describe().loc['count'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('mean'),
                                        *[html.Td(df.describe().loc['mean'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('std'),
                                        *[html.Td(df.describe().loc['std'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('min'),
                                        *[html.Td(df.describe().loc['min'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('25%'),
                                        *[html.Td(df.describe().loc['25%'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('50%'),
                                        *[html.Td(df.describe().loc['50%'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('75%'),
                                        *[html.Td(df.describe().loc['75%'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('max'),
                                        *[html.Td(df.describe().loc['max'][column]) for column in df.describe().columns]
                                    ]
                                ),
                            ]
                        )
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True,
                    style={'textAlign': 'center', 'width': '100%'},
                ),
            
            html.Br(),
            html.H3("Histograma y gráfica de cajas y bigotes:", style={'text-align': 'center'}),
                html.Div([
                    "Selecciona la o las variables para mostrar su histograma:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][3:3],
                        id='value-histograma-eda',
                        multi=True
                    ),

                    dcc.Graph(id='histograma-eda'),
                ])
            # ]),
            ,html.Br(),
                html.Div([
                    "Selecciona la o las variables para mostrar su gráfica de cajas y bigotes:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos por defecto todas las columnas numéricas, a partir de la segunda
                        value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][1:1],
                        id='value-bigotes-eda',
                        multi=True
                    ),
                
                dcc.Graph(id='bigotes-eda'),
                ]),
                html.Br(),
                html.H3("Matriz de correlación:", style={'text-align': 'center'}),
                dcc.Graph(
                    id='matriz',
                    figure={
                        'data': [
                            {'x': df.corr().columns, 'y': df.corr().columns, 'z': df.corr().values, 'type': 'heatmap', 'colorscale': 'RdBu'}
                        ],
                        'layout': {
                            'title': 'Matriz de correlación',
                            'xaxis': {'side': 'down'},
                            'yaxis': {'side': 'left'},
                            # Agregamos el valor de correlación por en cada celda (text_auto = True)
                            'annotations': [
                                dict(
                                    x=df.corr().columns[i],
                                    y=df.corr().columns[j],
                                    text=str(round(df.corr().values[i][j], 4)),
                                    showarrow=False,
                                    font=dict(
                                        color='white' if abs(df.corr().values[i][j]) >= 0.67  else 'black'
                                    )
                                ) for i in range(len(df.corr().columns)) for j in range(len(df.corr().columns))
                            ]
                        }
                    }
                )
            ,
            html.Br(),
            html.Button("Download CSV", id="btn_csv",
                        style={'textAlign': 'center', 'width': '100%'}
            ),
            dcc.Download(id="download-dataframe-csv"),
            html.Div(
                [
                    html.P("Encontraste relaciones fuertes entre las variables? Deberías aplicar Análisis de Componentes Principales (PCA) para reducir la dimensionalidad de tus datos."),
                    html.Button("Ir a PCA", id="btn_pca", style={'textAlign': 'center', 'width': '100%'}),
                ]
            ),
            # html.Button("Download Excel", id="btn_xlsx"),
            # dcc.Download(id="download_xslx"),
    ])

# Redirect to PCA
@callback(Output("url", "pathname"), Input("btn_pca", "n_clicks"), prevent_initial_call=True)
def func(n_clicks):
    if n_clicks> 0:
        return "/pca"

@callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(df.to_csv, "mydf.csv")

@callback(Output("download_xslx", "data"), Input("btn_xslx", "n_clicks"), prevent_initial_call=True)
def generate_xlsx(n_nlicks):

    def to_xlsx(bytes_io):
        xslx_writer = pd.ExcelWriter(bytes_io, engine="xlsxwriter")  # requires the xlsxwriter package
        df.to_excel(xslx_writer, index=False, sheet_name="sheet1")
        xslx_writer.close()

    return dcc.send_bytes(to_xlsx, "some_name.xlsx")

@callback(Output("loading-1", "children"), Input("upload-data", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@callback(Output('output-data-upload', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c,n,d) for c,n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children


@callback(
    Output('histograma-eda', 'figure'),
    Input('value-histograma-eda', 'value'))
def update_graph1(value):
    # Conforme se van seleccionando las variables, se van agregando a la gráfica de histogramas
    import plotly.graph_objects as go
    fig = go.Figure()
    for i in value:
        fig.add_trace(go.Histogram(x=df[i], name=i))
    fig.update_layout(
        xaxis_title=str(", ".join(value)),
        yaxis_title='Variable(s)',
    )

    return fig


@callback(
    Output('bigotes-eda', 'figure'),
    Input('value-bigotes-eda', 'value'))
def update_graph2(value):
    # Conforme se van seleccionando las variables, se van agregando a la gráfica de bigotes
    import plotly.graph_objects as go
    fig = go.Figure()
    for i in value:
        fig.add_trace(go.Box(y=df[i], name=i, boxpoints='all'))
    fig.update_layout(
        yaxis_title='COUNT',
    )

    return fig


@callback(
    Output("modal-body-scroll-eda", "is_open"),
    [
        Input("open-body-scroll-eda", "n_clicks"),
        Input("close-body-scroll-eda", "n_clicks"),
    ],
    [State("modal-body-scroll-eda", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


