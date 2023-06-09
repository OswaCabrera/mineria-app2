import base64
import datetime
import io
# from msilib.schema import Component
import dash_bootstrap_components as dbc
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib         
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.express as px
import plotly.graph_objs as go         # Para la visualización de datos basado en plotly

from sklearn.preprocessing import StandardScaler, MinMaxScaler

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])


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
    html.H1('Árboles - Regresión', style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Arrastra y suelta tu archivo aquí o selecciona uno'
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
        # Allow multiple files to be uploaded
        multiple=True,
        accept='.csv, .txt, .xls, .xlsx'
    ),
    html.Div(id='output-data-upload-arboles-regresion'), # output-datatable
    html.Div(id='output-div'),
])


def parse_contents(contents, filename,date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global df
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Hacemos una copia del dataframe original para poder hacer las modificaciones que queramos
            df_original = df.copy()
            
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'Uy lo siento, no me fue posible cargar tu documento. Aegurate que la extensión sea la correcta.'
        ])

    return html.Div([
        html.P('Estás trabajando con el archivo: {}. Si quieres cambiar de archivo vuelve a cargar otro'.format(filename)),

        html.H3("Tus datos son:" , style={'text-align': 'center'}),
        html.P(
            " {} Filas X {} Columnas.".format(df.shape[0], df.shape[1]),
            style={'text-align': 'center'}
        ),
        html.H3("Información acerca de tus variables:", style={'text-align': 'center'}),
        dash_table.DataTable(
            #Centramos la tabla de datos:
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
            row_selectable='multi',
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'height': '300px', 'overflowY': 'auto'},
        ),

        html.Hr(),

        html.H3("Matriz de correlación:", style={'text-align': 'center'}),
            dcc.Graph(
                id='matriz',
                figure={
                    # Solo se despliega la mitad de la matriz de correlación, ya que la otra mitad es simétrica
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
            ),

        html.Br(),
        html.H3("Estadísticas descriptivas de tus variables (EDA)", style={'text-align': 'center'}),
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
            style={'textAlign': 'center', 'width': '100%'}
        ),


        html.Br(),
        dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            # Primer columna: nombre de la columna y las demás columnas: nombre de las estadísticas (count, mean, std, min, 25%, 50%, 75%, max)
                            html.Th('Variable'),
                            html.Th('Tipo de dato'),
                            html.Th('Count'),
                            html.Th('Valores nulos'),
                            html.Th('Valores únicos'),
                            html.Th('Datos más frecuentes'),
                            html.Th('Datos menos frecuentes'),
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
                                        html.P("{}".format(df[column].value_counts().index[0])+" ("+str(round(df[column].value_counts().values[0]*1,2))+")"),
                                    ]
                                ),

                                # Top valores menos frecuentes
                                html.Td(
                                    [
                                        html.P("{}".format(df[column].value_counts().index[-1])+" ("+str(round(df[column].value_counts().values[-1]*1,2))+")"),
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
        ),

            html.Br(),
            html.H3("Creación del modelo", style={'text-align': 'center'}),
                html.H3("Selecciona la variable clase"),
                dcc.Dropdown(
                    [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                    # value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[0],
                    id='Y_Clase_Arbol_Regresion',
                ),

                html.H3("Selecciona las variables predictoras"),
                dcc.Dropdown(
                    # En las opciones que aparezcan en el Dropdown, queremos que aparezcan todas las columnas numéricas, excepto la columna Clase
                    [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                    # value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][1:],
                    id='X_Clase_Arbol_Regresion',
                    multi=True,
                ),

                # Salto de línea
                html.Br(),

                html.H2("Ajuste del algoritmo", style={'text-align': 'center'}),
                html.Br(),

                # html.P("Criterio: Puede ser gini (índice de Gini) y entropy (entropía) en clasificación."),
                # html.P("Divisor: Indica el criterio que se utilizará para dividir los nodos. Puede ser Squared Error, Friedman MSE, Absolute Error, Poisson"),
                # html.P("Profundidad: Indica la máxima profundidad que puede alcanzar el árbol. "),
                # html.P("Muestras-Dividir: Indica la cantidad mínima de muestras requeridas para que un nodo de decisión pueda dividirse."),
                # html.P("Muestras-Hoja: Indica la cantidad mínima de muestras que debe haber en una hoja del árbol. "),

                html.Hr(),
                dcc.Markdown('''**Datos para entrenar**'''),
                dcc.Input(
                    id='criterio_division_ADR',
                    type='number',
                    placeholder='None',
                    value=0.2,
                    min=0.2,
                    max=0.3,
                    step=0.05,
                ),

                dcc.Markdown('''**Criterio**'''),
                dbc.Select(
                    id='criterion_ADR',
                    options=[
                        {'label': 'Squared Error', 'value': 'squared_error'},
                        {'label': 'Friedman MSE', 'value': 'friedman_mse'},
                        {'label': 'Absolute Error', 'value': 'absolute_error'},
                        {'label': 'Poisson', 'value': 'poisson'},
                    ],
                    value='squared_error',
                    placeholder='None',
                ),

                dcc.Markdown('''**Divisor**'''),
                dbc.Select(
                    id='splitter_ADR',
                    options=[
                        {'label': 'Best', 'value': 'best'},
                        {'label': 'Random', 'value': 'random'},
                    ],
                    value='best',
                    placeholder='None',
                ),

                dcc.Markdown('''**Profundidad**'''),
                dbc.Input(
                    id='max_depth_ADR',
                    type='number',
                    placeholder='None',
                    value=7,
                    min=1,
                    max=100,
                    step=1,
                ),

                dcc.Markdown('''**Muestras - Dividir**'''),
                dbc.Input(
                    id='min_samples_split_ADR',
                    type='number',
                    placeholder='None',
                    value=2,
                    min=1,
                    max=100,
                    step=1,
                ),

                dcc.Markdown('''**Muestras - Hojas**'''),
                dbc.Input(
                    id='min_samples_leaf_ADR',
                    type='number',
                    placeholder='None',
                    value=1,
                    min=1,
                    max=100,
                    step=1,
                ),

                html.Br(),

                # Estilizamos el botón con Bootstrap
                dbc.Button("Entrenar", className="mr-1", color="dark", id='submit-button-arbol-regresion', style={'text-align': 'center' ,'width': '20%'}),

                html.Hr(),

                html.H2("Comparación Valores Reales y de Predicción", style={'text-align': 'center'}),
                # Mostramos la matriz de confusión
                dcc.Graph(id='matriz-arbol-regresion'),

                html.Hr(),

                html.H2("Reporte de la efectividad", style={'text-align': 'center'}),
                # Mostramos el reporte de clasificación
                html.Div(id='clasificacion-arbol-regresion'),

                # Mostramos la importancia de las variables
                dcc.Graph(id='importancia-arbol-regresion'),

                html.Hr(),

                dbc.Button(
                    "Árbol obtenido", id="open-body-scroll-ADR", n_clicks=0, color="dark", className="mr-1", style={'text-align': 'center' ,'width': '20%'}
                ),

                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Árbol de Decisión obtenido")),
                        dbc.ModalBody(
                            [
                                html.Div(id='arbol-arbol-regresion'),
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="close-body-scroll-ADR",
                                className="ms-auto",
                                n_clicks=0,
                                color="dark",
                            )
                        ),
                    ],
                    id="modal-body-scroll-ADR",
                    scrollable=True,
                    is_open=False,
                    size='xl',
                ),

                html.Hr(),

                html.Div(id='button-arbol-svg-ar'),

            dcc.Tab(label='Nuevos Pronósticos', style=tab_style, selected_style=tab_selected_style, children=[
                html.Div(id="output-regresion-arbol-regresion-Final"),

                html.Div(id='valor-regresion2'),
                html.Div(id='valor-regresion'),

                html.Hr(),

                dcc.Store(id='memory-output-arbol-regresion', data=df.to_dict('records')),
            ]),
    ]) #Fin del layout


@callback(Output('output-data-upload-arboles-regresion', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n,d) for c, n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children

@callback(
    Output('indicator_graphic_regression', 'figure'),
    Input('xaxis_column-arbol-regresion', 'value'),
    Input('yaxis_column-arbol-regresion', 'value'))
def update_graph2(xaxis_column2, yaxis_column2):
    # Conforme se van seleccionando las variables, se van agregando a la gráfica de líneas
    fig = go.Figure()
    for i in yaxis_column2:
        fig.add_trace(go.Scatter(x=df[xaxis_column2], y=df[i], mode='lines', name=i))
    fig.update_layout(xaxis_rangeslider_visible=True,showlegend=True, xaxis_title=xaxis_column2, yaxis_title='Valores',
                    font=dict(family="Courier New, monospace", size=18, color="black"))
    fig.update_traces(mode='markers+lines')

    return fig

@callback(
    Output('matriz-arbol-regresion', 'figure'),
    Output('clasificacion-arbol-regresion', 'children'),
    Output('importancia-arbol-regresion', 'figure'),
    Output('arbol-arbol-regresion', 'children'),
    Output('output-regresion-arbol-regresion-Final', 'children'),
    Output('valor-regresion2', 'children'),
    Output('button-arbol-svg-ar', 'children'),
    Input('submit-button-arbol-regresion', 'n_clicks'),
    State('X_Clase_Arbol_Regresion', 'value'),
    State('Y_Clase_Arbol_Regresion', 'value'),
    State('criterio_division_ADR', 'value'),
    State('criterion_ADR', 'value'),
    State('splitter_ADR', 'value'),
    State('max_depth_ADR', 'value'),
    State('min_samples_split_ADR', 'value'),
    State('min_samples_leaf_ADR', 'value'))
def regresion(n_clicks, X_Clase, Y_Clase, criterio_division,criterion, splitter, max_depth, min_samples_split, min_samples_leaf):
    if n_clicks is not None:
        global X
        global X_Clase2
        X_Clase2 = X_Clase
        X = np.array(df[X_Clase])
        Y = np.array(df[Y_Clase])

        global PronosticoAD

        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn import model_selection

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                                test_size = criterio_division, 
                                                                                random_state = 0,
                                                                                shuffle = True)

        #Se entrena el modelo a partir de los datos de entrada
        PronosticoAD = DecisionTreeRegressor(criterion = criterion, splitter = splitter, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, random_state = 0)
        PronosticoAD.fit(X_train, Y_train)

        #Se genera el pronóstico
        Y_PronosticoArbol = PronosticoAD.predict(X_test)
        
        ValoresArbol = pd.DataFrame(Y_test, Y_PronosticoArbol)

        # Comparación de los valores reales y los pronosticados en Plotly
        fig = px.line(Y_test, color_discrete_sequence=['green'])
        fig.add_scatter(y=Y_PronosticoArbol, name='Y_Pronostico', mode='lines', line=dict(color='red'))
        fig.update_layout(title='Comparación de valores reales vs Pronosticados',xaxis_rangeslider_visible=True)
        #Cambiamos el nombre de la leyenda
        fig.update_layout(legend_title_text='Valores')
        fig.data[0].name = 'Valores Reales'
        fig.data[1].name = 'Valores Pronosticados'
        # Renombramos el nombre de las leyendas:
        fig.update_traces(mode='markers+lines') #Agregamos puntos a la gráfica
        
        
        criterio = PronosticoAD.criterion
        profundidad = PronosticoAD.get_depth()
        hojas = PronosticoAD.get_n_leaves()
        splitter_report = PronosticoAD.splitter
        nodos = PronosticoAD.get_n_leaves() + PronosticoAD.get_depth()
        #MAE:
        MAEArbol = mean_absolute_error(Y_test, Y_PronosticoArbol)
        #MSE:
        MSEArbol = mean_squared_error(Y_test, Y_PronosticoArbol)
        #RMSE:
        RMSEArbol = mean_squared_error(Y_test, Y_PronosticoArbol, squared=False)
        
        global ScoreArbol
        ScoreArbol = r2_score(Y_test, Y_PronosticoArbol)
        

        # Importancia de las variables
        importancia = pd.DataFrame({'Variable': list(df[X_Clase].columns),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)

        # Graficamos la importancia de las variables
        fig2 = px.bar(importancia, x='Variable', y='Importancia', color='Importancia', color_continuous_scale='Bluered', text='Importancia')
        fig2.update_layout(title_text='Importancia de las variables', xaxis_title="Variables", yaxis_title="Importancia")
        fig2.update_traces(texttemplate='%{text:.2}', textposition='outside')
        fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig2.update_layout(legend_title_text='Importancia de las variables')

        # Generamos en texto el árbol de decisión
        from sklearn.tree import export_text
        r = export_text(PronosticoAD, feature_names=list(df[X_Clase].columns))
        
        return fig, html.Div([
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Score"),
                                html.Th("MAE"),
                                html.Th("MSE"),
                                html.Th("RMSE"),
                                html.Th("Criterio"),
                                html.Th("Divisor"),
                                html.Th("Profundidad"),
                                html.Th("Maxima Profundidad"),
                                html.Th("Muestras - Dividir"),
                                html.Th("Muestras - Hoja"),
                                html.Th("Nodos"),
                                html.Th("Hojas"),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(str(round(ScoreArbol, 6)*100) + '%'),
                                    html.Td(str(round(MAEArbol, 6))),
                                    html.Td(str(round(MSEArbol, 6))),
                                    html.Td(str(round(RMSEArbol, 6))),
                                    html.Td(criterio),
                                    html.Td(splitter_report),
                                    html.Td(profundidad),
                                    html.Td(str(max_depth)),
                                    html.Td(min_samples_split),
                                    html.Td(min_samples_leaf),
                                    html.Td(nodos),
                                    html.Td(PronosticoAD.get_n_leaves()),
                                ]
                            ),
                        ]
                    ),
                ],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                style={'width': '100%', 'text-align': 'center'},
                class_name='table table-hover table-bordered table-striped',
            ),


        ]), fig2, html.Div([
            dbc.Alert(r, style={'whiteSpace': 'pre-line'}, className="mb-3")
        ]), html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='values_X1', type="number", placeholder=df[X_Clase].columns[0],style={'width': '100%'}),
                    dbc.FormText("Nuevo valor de: " + str(df[X_Clase].columns[0])),
                    dbc.Input(id='values_X2', type="number", placeholder=df[X_Clase].columns[1],style={'width': '100%'}),
                    dbc.FormText("Nuevo valor de: " + str(df[X_Clase].columns[1])),
                    dbc.Input(id='values_X3', type="number", placeholder=df[X_Clase].columns[2],style={'width': '100%'}),
                    dbc.FormText("Nuevo valor de: " + str(df[X_Clase].columns[2])),
                    dbc.Input(id='values_X4', type="number", placeholder=df[X_Clase].columns[3],style={'width': '100%'}),
                    dbc.FormText("Nuevo valor de: " + str(df[X_Clase].columns[3])),
                    dbc.Input(id='values_X5', type="number", placeholder=df[X_Clase].columns[4],style={'width': '100%'}),
                    dbc.FormText("Nuevo valor de: " + str(df[X_Clase].columns[4])),
                ], width=6),
            ])

        ]), html.Div([
                dbc.Button("Nuevo pronóstico", id="collapse-button", className="mb-3", color="dark"),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody([
                        html.Div(id='output-container-button'),
                    ])),
                    id="collapse",
                ),
        ]), html.Div([
            dbc.Button(id='btn-ar', children='Árbol en formato PDF', color="dark", className="mr-1", style={'width': '100%', 'text-align': 'center'}),
            dcc.Download(id="download-ar"),
        ]),

    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate

# make sure that x and y values can't be the same variable
def filter_options(v):
    """Disable option v"""
    return [
        {"label": col, "value": col, "disabled": col == v}
        for col in [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]
    ]

# functionality is the same for both dropdowns, so we reuse filter_options
callback(Output("X_Clase_Arbol_Regresion", "options"), [Input("Y_Clase_Arbol_Regresion", "value")])(
    filter_options
)
callback(Output("Y_Clase_Arbol_Regresion", "options"), [Input("X_Clase_Arbol_Regresion", "value")])(
    filter_options
)

@callback(
    Output('valor-regresion', 'children'),
    Input('collapse-button', 'n_clicks'),
    # Mostar los valores de los inputs
    State('memory-output-arbol-regresion', 'data'),
    State('values_X1', 'value'),
    State('values_X2', 'value'),
    State('values_X3', 'value'),
    State('values_X4', 'value'),
    State('values_X5', 'value'),
)
def regresionFinal(n_clicks, data, values_X1, values_X2, values_X3, values_X4, values_X5):
    if n_clicks is not None:
        if values_X1 is None or values_X2 is None or values_X3 is None or values_X4 is None or values_X5 is None:
            return html.Div([
                dbc.Alert('Debe ingresar todos los valores de las variables', color="danger")
            ])
        else:
            XPredict = pd.DataFrame([[values_X1, values_X2, values_X3, values_X4, values_X5]])

            clasiFinal = PronosticoAD.predict(XPredict)
            return html.Div([
                dbc.Alert('Exactitud de: ' + str(round(ScoreArbol, 4)*100) + '% con un valor de: ' + str(clasiFinal[0]), style={'textAlign': 'center'})
            ])


@callback(
    Output("modal-body-scroll-ADR", "is_open"),
    [
        Input("open-body-scroll-ADR", "n_clicks"),
        Input("close-body-scroll-ADR", "n_clicks"),
    ],
    [State("modal-body-scroll-ADR", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(
    Output("download-ar", "data"),
    Input("btn-ar", "n_clicks"),
    prevent_initial_call=True,
)
def generar_arbol_svg(n_clicks):
    import graphviz
    from sklearn.tree import export_graphviz

    Elementos = export_graphviz(PronosticoAD,
                            feature_names = df[X_Clase2].columns,
                            filled = True,
                            rounded = True,
                            special_characters = True)
    Arbol = graphviz.Source(Elementos)
    Arbol.format = 'pdf'

    return dcc.send_file(Arbol.render(filename='ArbolAR', view=False))