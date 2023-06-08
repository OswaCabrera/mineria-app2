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
    html.H1('Árboles - Clasificación', style={'text-align': 'center'}),
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

        multiple=True,  # Allow multiple files to be uploaded
        accept='.csv, .txt, .xls, .xlsx'
    ),
    html.Div(id='output-data-upload-arboles-clasificacion'), # output-datatable
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
        dbc.Badge("Variables Predictoras", color="light", className="mr-1", text_color="dark"),
            dcc.Dropdown(
                [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                # Seleccionamos la segunda columna numérica del dataframe
                value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns,
                id='X_Clase',
                multi=True,
            ),

        # Seleccionamos la variable Clase con un Dropdown
        dbc.Badge("Variable Clase", color="light", className="mr-1", text_color="dark"),
        dcc.Dropdown(
            [i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)],
            # Seleccionamos por defecto la primera columna
            value=df[[i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)]].columns[0],
            id='Y_Clase',
            multi=True,
        ),

        # Salto de línea
        html.Br(),

        html.H2("Ajuste del algoritmo", style={'text-align': 'center'}),
        html.Br(),

        html.P("Criterio: Puede ser gini (índice de Gini) y entropy (entropía) en clasificación."),
        html.P("Divisor: Indica el criterio que se utilizará para dividir los nodos. Puede ser best (mejor) o random (aleatorio)."),
        html.P("Profundidad: Indica la máxima profundidad que puede alcanzar el árbol. "),
        html.P("Muestras-Dividir: Indica la cantidad mínima de muestras requeridas para que un nodo de decisión pueda dividirse."),
        html.P("Muestras-Hoja: Indica la cantidad mínima de muestras que debe haber en una hoja del árbol. "),

        dcc.Markdown('''**Datos para entrenar**'''),
        dcc.Input(
            id='criterio_division',
            type='number',
            placeholder='None',
            value=0.2,
            min=0.2,
            max=0.3,
            step=0.05,
        ),

        dcc.Markdown('''**Criterio**'''),
        dbc.Select(
            id='criterion',
            options=[
                {'label': 'Gini', 'value': 'gini'},
                {'label': 'Entropy', 'value': 'entropy'},
            ],
            value='gini',
            placeholder='None',
        ),

        dcc.Markdown('''**Divisor**'''),
        dbc.Select(
            id='splitter',
            options=[
                {'label': 'Best', 'value': 'best'},
                {'label': 'Random', 'value': 'random'},
            ],
            value='best',
            placeholder='None',
        ),

        dcc.Markdown('''**Profundidad**'''),
        dbc.Input(
            id='max_depth',
            type='number',
            placeholder='None',
            value=7,
            min=1,
            max=100,
            step=1,
        ),

        dcc.Markdown('''**Muestras - Dividir**'''),
        dbc.Input(
            id='min_samples_split',
            type='number',
            placeholder='None',
            value=2,
            min=1,
            max=100,
            step=1,
        ),

        dcc.Markdown('''**Muestras - Hojas**'''),
        dbc.Input(
            id='min_samples_leaf',
            type='number',
            placeholder='None',
            value=1,
            min=1,
            max=100,
            step=1,
        ),

        html.Br(),

        dbc.Button("Entrenar", className="mr-1", id='submit-button-clasificacion',style={'width': '100%'}),

        html.Hr(),

        # Mostramos la matriz de confusión
        html.Div(id='matriz'),

        html.Hr(),

        # Mostramos el reporte de clasificación
        html.Div(id='clasificacion'),

        # Mostramos la importancia de las variables
        dcc.Graph(id='importancia'),
        # Ocultamos el gráfico de la importancia de las variables hasta que se pulse el botón

        html.Hr(),

        html.H2("Curva ROC", style={'text-align': 'center'}),
        dcc.Graph(id='roc-arbol-clasificacion'),


        dbc.Button(
            "Árbol obtenido", id="open-body-scroll-ADC", n_clicks=0, color="primary", className="mr-1", style={'width': '100%'}
        ),

        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Árbol de Decisión obtenido")),
                dbc.ModalBody(
                    [
                        html.Div(id='arbol'),
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-body-scroll-ADC",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="modal-body-scroll-ADC",
            scrollable=True,
            is_open=False,
            size='xl',
        ),

        html.Hr(),

        html.Div(id='button-arbol-svg-modelos'),


        dcc.Tab(label='Nuevas Clasificaciones', style=tab_style, selected_style=tab_selected_style, children=[
            html.H2("Introduce los datos de las nuevas clasificaciones", style={'text-align': 'center'}),
            html.Hr(),
            html.Div(id='output-clasificacion-arbol-decision'),
            html.Hr(),
            html.Div(id='valor-clasificacion-arbol-decision'),
            html.Div(id='valor-clasificacion-arbol-decision2'),
            
        ]),
    ]) #Fin del layout

@callback(Output('output-data-upload-arboles-clasificacion', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n,d) for c, n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children

# CALLBACK PARA LA SELECCIÓN DEL USUARIO
@callback(
    Output('indicator_graphic', 'figure'),
    Input('xaxis_column', 'value'),
    Input('yaxis_column', 'value'),
    Input('caxis_column', 'value'))
def update_graph(xaxis_column, yaxis_column, caxis_column):
    dff = df
    dff[caxis_column] = dff[caxis_column].astype('category')
    fig = px.scatter(dff, x=xaxis_column, y=yaxis_column, color=caxis_column, title='Gráfico de dispersión',symbol=caxis_column,marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(showlegend=True, xaxis_title=xaxis_column, yaxis_title=yaxis_column,
                    font=dict(family="Courier New, monospace", size=18, color="black"),legend_title_text=caxis_column)
    #Modificamos el color de los puntos:
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    # str(df.groupby(caxis_column).size()[0])
    return fig

@callback(
    Output('matriz', 'children'),
    Output('clasificacion', 'children'),
    Output('importancia', 'figure'),
    Output('roc-arbol-clasificacion', 'figure'),
    Output('arbol', 'children'),
    Output('output-clasificacion-arbol-decision', 'children'),
    Output('valor-clasificacion-arbol-decision', 'children'),
    Output('button-arbol-svg-modelos', 'children'),
    Input('submit-button-clasificacion','n_clicks'),
    State('X_Clase', 'value'),
    State('Y_Clase', 'value'),
    State('criterio_division', 'value'),
    State('criterion', 'value'),
    State('splitter', 'value'),
    State('max_depth', 'value'),
    State('min_samples_split', 'value'),
    State('min_samples_leaf', 'value'))
def clasificacion(n_clicks, X_Clase, Y_Clase, criterio_division, criterion, splitter, max_depth, min_samples_split, min_samples_leaf):
    if n_clicks is not None:
        global ClasificacionAD
        global X_Clase2
        X_Clase2 = X_Clase
        global Y_Clase2
        Y_Clase2 = Y_Clase
        X = np.array(df[X_Clase])
        Y = np.array(df[Y_Clase])
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
        from sklearn import model_selection

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                        test_size = criterio_division,
                                                                                        random_state = 0,
                                                                                        shuffle = True)

        #Se entrena el modelo a partir de los datos de entrada
        ClasificacionAD = DecisionTreeClassifier(criterion = criterion,
                                                splitter = splitter,
                                                max_depth = max_depth,
                                                min_samples_split = min_samples_split,
                                                min_samples_leaf = min_samples_leaf,
                                                random_state = 0)
        ClasificacionAD.fit(X_train, Y_train)

        #Se etiquetan las clasificaciones
        global Y_Clasificacion
        Y_Clasificacion = ClasificacionAD.predict(X_validation)
        Valores = pd.DataFrame(Y_validation, Y_Clasificacion)

        #Se calcula la exactitud promedio de la validación
        global exactitud
        exactitud = ClasificacionAD.score(X_validation, Y_validation)
        
        #Matriz de clasificación
        ModeloClasificacion1 = ClasificacionAD.predict(X_validation)
        Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(), 
                                        ModeloClasificacion1, 
                                        rownames=['Reales'], 
                                        colnames=['Clasificación'])

        VP = Matriz_Clasificacion1.iloc[0,0]
        FP = Matriz_Clasificacion1.iloc[1,0]
        FN = Matriz_Clasificacion1.iloc[0,1]
        VN = Matriz_Clasificacion1.iloc[1,1]

        criterio = ClasificacionAD.criterion
        splitter_report = ClasificacionAD.splitter
        profundidad = ClasificacionAD.get_depth()
        hojas = ClasificacionAD.get_n_leaves()
        nodos = ClasificacionAD.get_n_leaves() + ClasificacionAD.get_depth()
        tasa_error = 1-ClasificacionAD.score(X_validation, Y_validation)

        # Importancia de las variables
        importancia = pd.DataFrame({'Variable': list(df[X_Clase].columns),
                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)

        # Graficamos la importancia de las variables
        fig2 = px.bar(importancia, x='Variable', y='Importancia', color='Importancia', color_continuous_scale='Bluered', text='Importancia')
        fig2.update_layout(title_text='Importancia de las variables', xaxis_title="Variables", yaxis_title="Importancia")
        fig2.update_traces(texttemplate='%{text:.2}', textposition='outside')
        fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig2.update_layout(legend_title_text='Importancia de las variables')

        #CURVA ROC
        Y_validation2 = pd.DataFrame(Y_validation) # Convertimos los valores de la variable Y_validation a un dataframe
        # Reeemplazamos los valores de la variable Y_validation2 por 0 y 1:
        # Checamos si el dataframe tiene dos valores únicos y si esos son 1 y 0
        if len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 0 and Y_validation2[0].unique()[1] == 1 or len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 1 and Y_validation2[0].unique()[1] == 0:
            pass
        else:
            Y_validation2 = Y_validation2.replace([Y_validation2[0].unique()[0],Y_validation2[0].unique()[1]],[1,0])

        # Graficamos la curva ROC con Plotly
        y_score1 = ClasificacionAD.predict_proba(X_validation)[:,1]

        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(Y_validation2, y_score1)
        # Graficamos la curva ROC con Plotly
        fig3 = px.area(title='Curva ROC. Árbol de Decisión. AUC = '+ str(auc(fpr, tpr).round(4)) )
        fig3.add_scatter(x=fpr, y=tpr, mode='lines', name='AD (Clasificación)', fill='tonexty')
        fig3.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Black", dash="dash"))
        fig3.update_layout(yaxis_title='True Positive Rate', xaxis_title='False Positive Rate')


        # Generamos en texto el árbol de decisión
        from sklearn.tree import export_text
        r = export_text(ClasificacionAD, feature_names=list(df[X_Clase].columns))
        
        return html.Div([
            html.H2("Clasifiaciones", style={'text-align': 'center'}),
            dbc.Row([
                dbc.Col([
                    dbc.Alert('Verdaderos Positivos (VP): ' + str(VP)),
                    dbc.Alert('Falsos Positivos (FP): ' + str(FP)),
                ], width=4),
                dbc.Col([
                    dbc.Alert('Falsos Negativos (FN): ' + str(FN)),
                    dbc.Alert('Verdaderos Negativos (VN): ' + str(VN)),
                ], width=4),
                ], justify="center"),

        ]), html.Div([
            html.H2("Reporte - Árbol", style={'text-align': 'center'}),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
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
                                    html.Td(criterio),
                                    html.Td(splitter_report),
                                    html.Td(profundidad),
                                    html.Td(str(max_depth)),
                                    html.Td(min_samples_split),
                                    html.Td(min_samples_leaf),
                                    html.Td(nodos),
                                    html.Td(ClasificacionAD.get_n_leaves()),
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

            html.H2("Reporte - Efectividad", style={'text-align': 'center'}),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Clasificación"),
                                html.Th("Clasificación - clase: " + str(classification_report(Y_validation, Y_Clasificacion).split()[4])),
                                html.Th("Clasificación - clase: " + str(classification_report(Y_validation, Y_Clasificacion).split()[9])),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Exactitud: " + str(round(exactitud*100,2)) + '%'),
                                    html.Td("Precisión: " + str(round(float(VP/(VP+FP))*100,5)) + '%'),
                                    html.Td("Precisión: " + str(round(float(VN/(VN+FN))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Tasa de error: " + str(round(tasa_error*100,2)) + '%'),
                                    html.Td("Sensibilidad: " + str(round(float(VP/(VP+FN))*100,5)) + '%'),
                                    html.Td("Sensibilidad: " + str(round(float(VN/(VN+FP))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Verdaderos: " + str((Y_validation == Y_Clasificacion).sum())),
                                    html.Td("Especificidad: " + str(round(float(VN/(VN+FP))*100,5)) + '%'),
                                    html.Td("Especificidad: " + str(round(float(VP/(VP+FN))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Falsos: " + str((Y_validation != Y_Clasificacion).sum())),
                                    html.Td("Score: " + str(round(float(2*VP/(2*VP+FP+FN))*100,5)) + '%'),
                                    html.Td("Score: " + str(round(float(2*VN/(2*VN+FN+FP))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Totales: " + str(Y_validation.size)),
                                    html.Td("Muestras: " + str(classification_report(Y_validation, Y_Clasificacion).split()[8])),
                                    html.Td("Muestras: " + str(classification_report(Y_validation, Y_Clasificacion).split()[13])),
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
            ),

            html.H2("Relevancia de las variables", style={'text-align': 'center'}),
        ]), fig2, fig3, dbc.Alert(r, color="success", style={'whiteSpace': 'pre-line'}, className="mb-3"), html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='values_X1_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[0],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[0])),
                    dbc.Input(id='values_X2_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[1],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[1])),
                    dbc.Input(id='values_X3_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[2],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[2])),
                ], width=4),
                dbc.Col([
                    dbc.Input(id='values_X4_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[3],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[3])),
                    dbc.Input(id='values_X5_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[4],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[4])),
                    dbc.Input(id='values_X6_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[5],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[5])),
                ], width=4),

                dbc.Col([
                    dbc.Input(id='values_X7_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[6],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[6])),
                    dbc.Input(id='values_X8_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[7],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[7])),
                    dbc.Input(id='values_X9_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[8],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[8])),
                ], width=4),
            ]),

        ]), html.Div([
                dbc.Button("Nueva clasificación...", id="collapse-button", className="mb-3", color="primary"),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody([
                        html.Div(id='output-container-button'),
                    ])),
                    id="collapse",
                ),
        ]), html.Div([
            dbc.Button(id='btn', children='Árbol de decisión en formato PDF', color="dark", className="mr-1", style={'width': '100%', 'text-align': 'center'}),
            dcc.Download(id="download-ad-modelos"),
        ]),

    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate


@callback(
    Output('valor-clasificacion-arbol-decision2', 'children'),
    Input('collapse-button', 'n_clicks'),
    State('values_X1_AD_Clasificacion', 'value'),
    State('values_X2_AD_Clasificacion', 'value'),
    State('values_X3_AD_Clasificacion', 'value'),
    State('values_X4_AD_Clasificacion', 'value'),
    State('values_X5_AD_Clasificacion', 'value'),
    State('values_X6_AD_Clasificacion', 'value'),
    State('values_X7_AD_Clasificacion', 'value'),
    State('values_X8_AD_Clasificacion', 'value'),
    State('values_X9_AD_Clasificacion', 'value'),
)
def AD_Clasificacion_Pronostico(n_clicks, values_X1, values_X2, values_X3, values_X4, values_X5, values_X6, values_X7, values_X8, values_X9):
    if n_clicks is not None:
        if values_X1 is None or values_X2 is None or values_X3 is None or values_X4 is None or values_X5 is None or values_X6 is None or values_X7 is None or values_X8 is None or values_X9 is None:
            return html.Div([
                dbc.Alert('Debe ingresar los valores de todas las variables', color="danger")
            ])
        else:
            XPredict = pd.DataFrame([[values_X1, values_X2, values_X3, values_X4, values_X5, values_X6, values_X7, values_X8, values_X9]])

            clasiFinal = ClasificacionAD.predict(XPredict)
            return html.Div([
                dbc.Alert('El valor clasificado con un árbol de decisión que tiene una Exactitud de: ' + str(round(exactitud, 4)*100) + '% es: ' + str(clasiFinal[0]), color="success", style={'textAlign': 'center'})
            ])


@callback(
    Output("modal-body-scroll-ADC", "is_open"),
    [
        Input("open-body-scroll-ADC", "n_clicks"),
        Input("close-body-scroll-ADC", "n_clicks"),
    ],
    [State("modal-body-scroll-ADC", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open



@callback(
    Output("download-ad-modelos", "data"),
    Input("btn", "n_clicks"),
    prevent_initial_call=True,
)
def generar_arbol_svg(n_clicks):
    import graphviz
    from sklearn.tree import export_graphviz

    Elementos = export_graphviz(ClasificacionAD, 
                            feature_names = df[X_Clase2].columns,
                            class_names = df[Y_Clase2].unique().astype(str),
                            filled = True,
                            rounded = True,
                            special_characters = True)
    Arbol = graphviz.Source(Elementos)
    Arbol.format = 'pdf'

    return dcc.send_file(Arbol.render(filename='ArbolAD', view=False))