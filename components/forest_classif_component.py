import base64
import datetime
import io
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
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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

layout = html.Div([
    html.H1('Bosques Aleatorios de clasificación', style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or Select Files'
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
    html.Div(id='output-data-upload-bosques-clasificacion'),
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
            'There was an error processing this file.'
        ])

    return html.Div([
        html.P('Estás trabajando con el archivo: {}'.format(filename)),
        # dcc.Tab(label='Aplicación del algoritmo', style=tab_style, selected_style=tab_selected_style, children=[
                html.H3("Selecciona las variables predictoras", style={'text-align': 'center'}),
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        id='X_Clase_bosque_clasificacion',
                        multi=True,
                    ),

                # Seleccionamos la variable Clase con un Dropdown
                html.H3("Selecciona la variable Clase", style={'text-align': 'center'}),
                dcc.Dropdown(
                    [i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)],
                    id='Y_Clase_bosque_clasificacion',
                    multi=True,
                ),

                html.H3("Selecciona los parametros", style = {'text-align': 'center'}),
                html.Br(),

                dbc.Row([
                    dbc.Col([
                        dcc.Markdown('''**Criterio de División (Tamaño del test %):**'''),
                        dbc.Input(
                            id='criterio_division_BA',
                            value = 0.2, type='number',
                             min=0.2,
                              max=0.3,
                               step=0.01),
                    ], width=3, align='center'),
                    dbc.Col([
                        dcc.Markdown('''**Criterio:**'''),
                        dbc.Select(
                            id='criterion_BAC',
                            options=[
                                {'label': 'Ganancia de información', 'value': 'gini'},
                                {'label': 'Entropia', 'value': 'entropy'},
                            ],
                            value='gini',
                            placeholder="Selecciona el criterio",
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**n_estimators:**'''),
                        dbc.Input(
                            id='n_estimators_BAC',
                            type='number',
                            placeholder='Ingresa el número de árboles',
                            value=100,
                            min=1,
                            max=1000,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**n_jobs:**'''),
                        dbc.Input(
                            id='n_jobs_BAC',
                            type='number',
                            placeholder='None',
                            value=None,
                            min=-1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),


                    dbc.Col([
                        dcc.Markdown('''**max_features:**'''),
                        dbc.Select(
                            id='max_features_BAC',
                            options=[
                                {'label': 'Auto', 'value': 'auto'},
                                {'label': 'sqrt', 'value': 'sqrt'},
                                {'label': 'log2', 'value': 'log2'},
                            ],
                            value='auto',
                            placeholder="Selecciona una opción",
                        ),
                    ], width=2, align='center'),

                    
                    dbc.Col([
                        dcc.Markdown('''**Max_depth:**'''),
                        dbc.Input(
                            id='max_depth_BAC',
                            type='number',
                            placeholder='None',
                            value=None,
                            min=1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**Min_samples_split:**'''),
                        dbc.Input(
                            id='min_samples_split_BAC',
                            type='number',
                            placeholder='Selecciona el min_samples_split',
                            value=2,
                            min=1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**Min_samples_leaf:**'''),
                        dbc.Input(
                            id='min_samples_leaf_BAC',
                            type='number',
                            placeholder='Selecciona el min_samples_leaf',
                            value=1,
                            min=1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**max_leaf_nodes:**'''),
                        dbc.Input(
                            id='max_leaf_nodes_BAC',
                            type='number',
                            placeholder='None',
                            value=None,
                            min=1,
                            max=1000,
                            step=1,
                        ),
                    ], width=2, align='left'),
            
                ], align='left'),

                html.Hr(),

                # Estilizamos el botón con Bootstrap
                dbc.Button("Aplicar algoritmo",  color="primary", className="mr-1", id='submit-button-clasificacion-bosques', style={'text-align': 'center' ,'width': '20%'}),

                html.Hr(),

                # Mostramos la matriz de confusión
                html.Div(id='matriz-bosque-clasificacion'),

                html.Hr(),

                # Mostramos el reporte de clasificación
                html.Div(id='clasificacion-bosque-clasificacion'),

                # Mostramos la importancia de las variables
                html.H3("Importancia de las variables", style={'text-align': 'center'}),
                dcc.Graph(id='importancia-bosque-clasificacion'),

                html.Hr(),

                html.H3("Curva ROC", style={'text-align': 'center'}),
                dcc.Graph(id='roc-bosque-clasificacion'),

                html.Div(id='arbol_estimador_BAC'),
                html.Div(id='arbol_estimador_BAC2'),
            # ]),

            # dcc.Tab(label='Nuevas Clasificaciones', style=tab_style, selected_style=tab_selected_style, children=[
                html.H3("Introduce los datos de las nuevas clasificaciones", style={'text-align': 'center'}),
                html.Hr(),
                html.Div(id='output-clasificacion-BAC'),
                html.Hr(),
                html.Div(id='valor-clasificacion-BAC'),
                html.Div(id='valor-clasificacion-BAC2'),
                
            # ]),


        ])
    # ]) 

@callback(Output('output-data-upload-bosques-clasificacion', 'children'),
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
    Output('indicator_graphic_bosques', 'figure'),
    Input('xaxis_column_bosque_clasificacion', 'value'),
    Input('yaxis_column_bosque_clasificacion', 'value'),
    Input('caxis_column_bosque_clasificacion', 'value'))
def update_graph(xaxis_column, yaxis_column, caxis_column):
    dff = df
    dff[caxis_column] = dff[caxis_column].astype('category')
    fig = px.scatter(dff, x=xaxis_column, y=yaxis_column, color=caxis_column, title='Gráfico de dispersión',symbol=caxis_column,marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(showlegend=True, xaxis_title=xaxis_column, yaxis_title=yaxis_column,
                    font=dict(family="Courier New, monospace", size=18, color="black"),legend_title_text=caxis_column)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    return fig

@callback(
    Output('matriz-bosque-clasificacion', 'children'),
    Output('clasificacion-bosque-clasificacion', 'children'),
    Output('importancia-bosque-clasificacion', 'figure'),
    Output('roc-bosque-clasificacion', 'figure'),
    Output('output-clasificacion-BAC', 'children'),
    Output('valor-clasificacion-BAC', 'children'),
    Output('arbol_estimador_BAC', 'children'),
    Output('arbol_estimador_BAC2', 'children'),
    Input('submit-button-clasificacion-bosques','n_clicks'),
    State('X_Clase_bosque_clasificacion', 'value'),
    State('Y_Clase_bosque_clasificacion', 'value'),
    State('criterio_division_BA', 'value'),
    State('criterion_BAC', 'value'),
    State('n_estimators_BAC', 'value'),
    State('n_jobs_BAC', 'value'),
    State('max_features_BAC', 'value'),
    State('max_depth_BAC', 'value'),
    State('min_samples_split_BAC', 'value'),
    State('min_samples_leaf_BAC', 'value'),
    State('max_leaf_nodes_BAC', 'value'))
def clasificacion(n_clicks, X_Clase, Y_Clase, criterio_division, criterion, n_estimators, n_jobs, max_features, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes):
    if n_clicks is not None:
        global X_Clase2
        global n_estimators2
        X_Clase2 = X_Clase
        n_estimators2 = n_estimators

        X = np.array(df[X_Clase])
        Y = np.array(df[Y_Clase])

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn import model_selection

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                        test_size = criterio_division,
                                                                                        random_state = 0,
                                                                                        shuffle = True)

        #Se entrena el modelo a partir de los datos de entrada
        global ClasificacionBA
        ClasificacionBA = RandomForestClassifier(criterion = criterion,
                                                n_estimators = n_estimators,
                                                n_jobs = n_jobs,
                                                max_features = max_features,
                                                max_depth = max_depth,
                                                min_samples_split = min_samples_split,
                                                min_samples_leaf = min_samples_leaf,
                                                max_leaf_nodes = max_leaf_nodes, random_state = 0)
        ClasificacionBA.fit(X_train, Y_train)

        #Se etiquetan las clasificaciones
        Y_Clasificacion = ClasificacionBA.predict(X_validation)
        Valores = pd.DataFrame(Y_validation, Y_Clasificacion)

        #Se calcula la exactitud promedio de la validación
        global exactitud
        exactitud = accuracy_score(Y_validation, Y_Clasificacion)
        
        #Matriz de clasificación
        ModeloClasificacion = ClasificacionBA.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                        ModeloClasificacion, 
                                        rownames=['Reales'], 
                                        colnames=['Clasificación'])
        
        VP = Matriz_Clasificacion.iloc[0,0]
        FP = Matriz_Clasificacion.iloc[1,0]
        FN = Matriz_Clasificacion.iloc[0,1]
        VN = Matriz_Clasificacion.iloc[1,1]

        criterio = ClasificacionBA.criterion
        precision = classification_report(Y_validation, Y_Clasificacion).split()[10]
        tasa_error = 1-ClasificacionBA.score(X_validation, Y_validation)
        sensibilidad = classification_report(Y_validation, Y_Clasificacion).split()[11]
        especificidad = classification_report(Y_validation, Y_Clasificacion).split()[6]

        # Importancia de las variables
        importancia = pd.DataFrame({'Variable': list(df[X_Clase].columns),
                            'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)

        # Graficamos la importancia de las variables
        fig2 = px.bar(importancia, x='Variable', y='Importancia', color='Importancia', color_continuous_scale='Bluered', text='Importancia')
        fig2.update_layout(title_text='Importancia de las variables', xaxis_title="Variables", yaxis_title="Importancia")
        fig2.update_traces(texttemplate='%{text:.2}', textposition='outside')
        fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig2.update_layout(legend_title_text='Importancia de las variables')

        #CURVA ROC
        Y_validation2 = pd.DataFrame(Y_validation) # Convertimos los valores de la variable Y_validation a un dataframe
        # Reeemplazamos los valores de la variable Y_validation2 por 0 y 1:
        if len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 0 and Y_validation2[0].unique()[1] == 1 or len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 1 and Y_validation2[0].unique()[1] == 0:
            pass
        else:
            Y_validation2 = Y_validation2.replace([Y_validation2[0].unique()[0],Y_validation2[0].unique()[1]],[1,0])

        # Graficamos la curva ROC con Plotly
        y_score1 = ClasificacionBA.predict_proba(X_validation)[:,1]

        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(Y_validation2, y_score1)
        # Graficamos la curva ROC con Plotly
        fig3 = px.area(title='Curva ROC. Bosque aleatorio. AUC = '+ str(auc(fpr, tpr).round(4)) )
        fig3.add_scatter(x=fpr, y=tpr, mode='lines', name='Bosque Aleatorio', fill='tonexty')
        fig3.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Black", dash="dash"))
        fig3.update_layout(yaxis_title='True Positive Rate', xaxis_title='False Positive Rate')

        # Generamos en texto el árbol de decisión
        Estimador = ClasificacionBA.estimators_[1] # SE TIENE QUE ELEGIR POR EL USUARIO
        from sklearn.tree import export_text
        r = export_text(Estimador, feature_names=list(df[X_Clase].columns))


        return html.Div([
            html.H2(["", html.H3("Matriz de clasificación")]),
            dbc.Row([
                dbc.Col([
                    dbc.Alert('Verdaderos Positivos (VP): ' + str(VP), color="info"),
                    dbc.Alert('Falsos Positivos (FP): ' + str(FP), color="info"),
                ], width=4),
                dbc.Col([
                    dbc.Alert('Falsos Negativos (FN): ' + str(FN), color="info"),
                    dbc.Alert('Verdaderos Negativos (VN): ' + str(VN), color="info"),
                ], width=4),
                ], justify="center"), 
        ]), html.Div([
            html.H2(["", html.H3("Reporte del bosque obtenido")]),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Criterion"),
                                html.Th("n_estimators"),
                                html.Th("n_jobs"),
                                html.Th("max_features"),
                                html.Th("Max_depth"),
                                html.Th("Min_samples_split"),
                                html.Th("Min_samples_leaf"),
                                html.Th("Max_leaf_nodes"),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(criterio),
                                    html.Td(str(n_estimators)),
                                    html.Td(str(n_jobs)),
                                    html.Td(str(max_features)),
                                    html.Td(str(max_depth)),
                                    html.Td(str(min_samples_split)),
                                    html.Td(str(min_samples_leaf)),
                                    html.Td(str(max_leaf_nodes)),
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

            html.H2(["", html.H3("Reporte de la efectividad del algoritmo obtenido")]),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Reporte de clasificación"),
                                html.Th("Reporte de clasificación para la clase: " + str(classification_report(Y_validation, Y_Clasificacion).split()[4])),
                                html.Th("Reporte de clasificación para la clase: " + str(classification_report(Y_validation, Y_Clasificacion).split()[9])),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Exactitud (Accuracy): " + str(round(exactitud*100,2)) + '%', style={'color': 'green'}),
                                    html.Td("Precisión: " + str(round(float(VP/(VP+FP))*100,5)) + '%'),
                                    html.Td("Precisión: " + str(round(float(VN/(VN+FN))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Tasa de error (Misclassification Rate): " + str(round(tasa_error*100,2)) + '%', style={'color': 'red'}),
                                    html.Td("Sensibilidad (Recall, Sensitivity, True Positive Rate): " + str(round(float(VP/(VP+FN))*100,5)) + '%'),
                                    html.Td("Sensibilidad (Recall, Sensitivity, True Positive Rate): " + str(round(float(VN/(VN+FP))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Verdaderos: " + str((Y_validation == Y_Clasificacion).sum()), style={'color': 'green'}),
                                    html.Td("Especificidad (Specificity, True Negative Rate): " + str(round(float(VN/(VN+FP))*100,5)) + '%'),
                                    html.Td("Especificidad (Specificity, True Negative Rate): " + str(round(float(VP/(VP+FN))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Falsos: " + str((Y_validation != Y_Clasificacion).sum()), style={'color': 'red'}),
                                    html.Td("F1-Score: " + str(round(float(2*VP/(2*VP+FP+FN))*100,5)) + '%'),
                                    html.Td("F1-Score: " + str(round(float(2*VN/(2*VN+FN+FP))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Totales: " + str(Y_validation.size)),
                                    html.Td("Número de muestras: " + str(classification_report(Y_validation, Y_Clasificacion).split()[8])),
                                    html.Td("Número de muestras: " + str(classification_report(Y_validation, Y_Clasificacion).split()[13])),
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


        ]), fig2, fig3, html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='values_X1_BA_Clasificacion', type="number", placeholder=df[X_Clase].columns[0],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[0])),
                    dbc.Input(id='values_X2_BA_Clasificacion', type="number", placeholder=df[X_Clase].columns[1],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[1])),
                    dbc.Input(id='values_X3_BA_Clasificacion', type="number", placeholder=df[X_Clase].columns[2],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[2])),
                ], width=4),
                dbc.Col([
                    dbc.Input(id='values_X4_BA_Clasificacion', type="number", placeholder=df[X_Clase].columns[3],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[3])),
                    dbc.Input(id='values_X5_BA_Clasificacion', type="number", placeholder=df[X_Clase].columns[4],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[4])),
                    dbc.Input(id='values_X6_BA_Clasificacion', type="number", placeholder=df[X_Clase].columns[5],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[5])),
                ], width=4),

                dbc.Col([
                    dbc.Input(id='values_X7_BA_Clasificacion', type="number", placeholder=df[X_Clase].columns[6],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[6])),
                    dbc.Input(id='values_X8_BA_Clasificacion', type="number", placeholder=df[X_Clase].columns[7],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[7])),
                    dbc.Input(id='values_X9_BA_Clasificacion', type="number", placeholder=df[X_Clase].columns[8],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[8])),
                ], width=4),
            ]),

        ]), html.Div([
                dbc.Button("Haz click para mostrar la clasificación...", id="collapse-button-BA", className="mb-3", color="primary"),
        ]),  
    
    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate

@callback(
    Output('valor-clasificacion-BAC2', 'children'),
    Input('collapse-button-BA', 'n_clicks'),
    State('values_X1_BA_Clasificacion', 'value'),
    State('values_X2_BA_Clasificacion', 'value'),
    State('values_X3_BA_Clasificacion', 'value'),
    State('values_X4_BA_Clasificacion', 'value'),
    State('values_X5_BA_Clasificacion', 'value'),
    State('values_X6_BA_Clasificacion', 'value'),
    State('values_X7_BA_Clasificacion', 'value'),
    State('values_X8_BA_Clasificacion', 'value'),
    State('values_X9_BA_Clasificacion', 'value'),
)
def AD_Clasificacion_Pronostico(n_clicks, values_X1, values_X2, values_X3, values_X4, values_X5, values_X6, values_X7, values_X8, values_X9):
    if n_clicks is not None:
        if values_X1 is None or values_X2 is None or values_X3 is None or values_X4 is None or values_X5 is None or values_X6 is None or values_X7 is None or values_X8 is None or values_X9 is None:
            return html.Div([
                dbc.Alert('Debe ingresar los valores de todas las variables', color="danger")
            ])
        else:
            
            XPredict = pd.DataFrame([[values_X1, values_X2, values_X3, values_X4, values_X5, values_X6, values_X7, values_X8, values_X9]])

            clasiFinal = ClasificacionBA.predict(XPredict)
            return html.Div([
                dbc.Alert('El valor clasificado con un Bosque Aleatorio que tiene una Exactitud de: ' + str(round(exactitud, 4)*100) + '% es: ' + str(clasiFinal[0]), color="success", style={'textAlign': 'center'})
            ])


@callback(
    Output('arbol_elegido_BAC', 'children'),
    Input('arbol_estimador_BAC', 'value'),
)
def generar_arbol(arbol_elegido_BAC):
    try:
        EstimadorBAC = ClasificacionBA.estimators_[arbol_elegido_BAC]

        # Generamos en texto el árbol de decisión
        from sklearn.tree import export_text
        r = export_text(EstimadorBAC, feature_names=list(df[X_Clase2].columns))

        return html.Div([
            dbc.Alert(r, color="success", style={'whiteSpace': 'pre-line'}, className="mb-3")
        ])
    except:
        return html.Div([
            dbc.Alert('El árbol elegido no existe, intenta con otro estimador...', color="danger", style={'textAlign': 'center'})
        ])



@callback(
    Output("modal-body-scroll-BAC-Arbol", "is_open"),
    [
        Input("open-body-scroll-BAC-Arbol", "n_clicks"),
        Input("close-body-scroll-BAC-Arbol", "n_clicks"),
    ],
    [State("modal-body-scroll-BAC-Arbol", "is_open")],
)
def toggle_modal_BAC(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
