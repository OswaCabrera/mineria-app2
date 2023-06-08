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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

# load_figure_template("plotly_white")

# df = pd.read_csv("C:\Users\oswal\Documents\DecimoSemestre\DataMining\Data\miami-housing.csv")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
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



layout = html.Div([
    html.H1('Análisis de Componentes Principal', style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Para empezar seleccione un archivo o arrastrelo y sueltelo aquí'
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
        multiple=True,
        accept='.csv, .txt, .xls, .xlsx'
    ),
    html.Div(id='output-data-upload-acp'), # output-datatable
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
        dcc.Tab(label='Analisis Correlacional', children=[
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
            )
        ]),        
        dcc.Markdown('''**Selecciona un método de estandarización**'''),
        dbc.Select(
            id='select-escale',
            options=[
                {'label': 'StandardScaler', 'value': "StandardScaler()"},
                {'label': 'MinMaxScaler', 'value': "MinMaxScaler()"},
            ],
            value="StandardScaler()",
            placeholder="Selecciona el tipo de estandarización",
        ),
        dcc.Markdown('''**Selecciona el número de componentes principales**'''),
        dbc.Input(
            id='n_components',
            type='number',
            placeholder='None',
            value=None,
            min=1,
            max=100,
        ),

        dcc.Markdown('''** Selecciona el porcentaje de relevancia. Recomendamos un valor entre 0.75 y 0.9**'''),
        dbc.Input(
            id='relevancia',
            type='number',
            placeholder='Ingrese el porcentaje de relevancia (0 - 1)',
        ),

        html.Br(),

        dbc.Button("Aplicar los componentes principales", color="warning", className="mr-1", id='submit-button-standarized', style={'width': '20%'}),

        html.Hr(),

        # dcc.Tabs([
            html.H3('Matriz de datos estandarizada', style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='DataTableStandarized',
                    columns=[{"name": i, "id": i} for i in df.select_dtypes(include=['float64', 'int64']).columns],
                    page_size=8,
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(207, 250, 255)', 'color': 'black'},
                    style_header={'backgroundColor': 'rgb(45, 93, 255)', 'fontWeight': 'bold', 'color': 'black', 'border': '1px solid black'},
                    style_table={'height': '300px', 'overflowY': 'auto'},
                    style_data={'border': '1px solid black'}
                ),
            html.H3('Varianza explicada en porcentaje (%)', style={'textAlign': 'center'}),
                dcc.Graph(
                    id='varianza-explicada',
                ),
            html.H3('Número de componentes principales y la varianza acumulada', style={'textAlign': 'center'}),
                dcc.Graph(
                    id='varianza',
                ),
            html.H3('Cargas de las variables', style={'textAlign': 'center'}),
                dcc.Graph(
                    id='FigComponentes',
                ),
            html.Button("Download CSV", id="btn_csv",
                        style={'textAlign': 'center', 'width': '30%', 'margin': 'auto'}
            ),
            dcc.Download(id="download-dataframe-csv2"),
            
        # ])
    ]) #Fin del layout

@callback(
    Output("download-dataframe-csv2", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(df.to_csv, "mydf.csv")

@callback(Output('output-data-upload-acp', 'children'),
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
    Output('DataTableStandarized','data'),
    Output('varianza-explicada', 'figure'),
    Output('varianza', 'figure'),
    Output('FigComponentes', 'figure'),
    Input('submit-button-standarized','n_clicks'),
    State('select-escale', 'value'),
    State('n_components', 'value'),
    State('relevancia', 'value'),
)
def calculoPCA(n_clicks, estandarizacion, n_componentes, relevancia):
    if n_clicks is not None:
        global MEstandarizada1
        df_numeric = df.select_dtypes(include=['float64', 'int64'])
        if estandarizacion == "StandardScaler()":
            MEstandarizada1 = StandardScaler().fit_transform(df_numeric) # Se estandarizan los datos
        elif estandarizacion == "MinMaxScaler()":
            MEstandarizada1 = MinMaxScaler().fit_transform(df_numeric)
        
        MEstandarizada = pd.DataFrame(MEstandarizada1, columns=df_numeric.columns) # Se convierte a dataframe

        pca = PCA(n_components=n_componentes).fit(MEstandarizada) # Se calculan los componentes principales
        Varianza = pca.explained_variance_ratio_

        for i in range(0, Varianza.size):
            varAcumulada = sum(Varianza[0:i+1])
            if varAcumulada >= relevancia:
                varAcumuladaACP = (varAcumulada - Varianza[i])
                numComponentesACP = i - 1
                break
        
        # Se grafica la varianza explicada por cada componente en un gráfico de barras en Plotly:
        fig = px.bar(x=range(1, Varianza.size +1), y=Varianza*100, labels=dict(x="Componentes Principales", y="Varianza explicada (%)"), title='Varianza explicada por cada componente')
        # A cada barra se le agrega el porcentaje de varianza explicada
        for i in range(1, Varianza.size +1):
            fig.add_annotation(x=i, y=Varianza[i-1]*100, text=str(round(Varianza[i-1]*100, 2)) + '%',
            # Se muestran por encima de la barra:
            yshift=10, showarrow=False, font_color='black')
        # Se agrega una gráfica de línea de la varianza explicada que pase por cada barra:
        fig.add_scatter(x=np.arange(1, Varianza.size+1, step=1), y=Varianza*100, mode='lines+markers', name='Varianza explicada',showlegend=False)
        # Mostramos todos los valores del eje X:
        fig.update_xaxes(tickmode='linear')
        
        fig2 = px.line(x=np.arange(1, Varianza.size+1, step=1), y=np.cumsum(Varianza))
        fig2.update_layout(title='Varianza acumulada en los componentes',
                            xaxis_title='Número de componentes',
                            yaxis_title='Varianza acumulada')
        # Se resalta el número de componentes que se requieren para alcanzar el 90% de varianza acumulada
        fig2.add_shape(type="line", x0=1, y0=relevancia, x1=numComponentesACP+1, y1=relevancia, line=dict(color="Red", width=2, dash="dash"))
        fig2.add_shape(type="line", x0=numComponentesACP+1, y0=0, x1=numComponentesACP+1, y1=varAcumuladaACP, line=dict(color="Green", width=2, dash="dash"))
        # Se muestra un punto en la intersección de las líneas
        fig2.add_annotation(x=numComponentesACP+1, y=varAcumuladaACP, text=str(round(varAcumuladaACP*100, 1))+f'%. {numComponentesACP+1} Componentes', showarrow=True, arrowhead=1)
        # Se agregan puntos en la línea de la gráfica
        fig2.add_scatter(x=np.arange(1, Varianza.size+1, step=1), y=np.cumsum(Varianza), mode='markers', marker=dict(size=10, color='blue'), showlegend=False, name='# Componentes')
        # Se le agrega el área bajo la curva
        fig2.add_scatter(x=np.arange(1, Varianza.size+1, step=1), y=np.cumsum(Varianza), fill='tozeroy', mode='none', showlegend=False, name='Área bajo la curva')
        fig2.update_xaxes(range=[1, Varianza.size]) # Se ajusta al tamaño de la gráfica
        fig2.update_xaxes(tickmode='linear')
        fig2.update_yaxes(range=[0, 1.1], 
                        tickmode='array',
                        tickvals=np.arange(0, 1.1, step=0.1))

        # 6
        CargasComponentes = pd.DataFrame(abs(pca.components_), columns=df_numeric.columns)
        CargasComponentess=CargasComponentes.head(numComponentesACP+1) 

        fig3 = px.imshow(CargasComponentes.head(numComponentesACP+1), color_continuous_scale='RdBu_r')
        fig3.update_layout(title='Cargas de los componentes', xaxis_title='Variables', yaxis_title='Componentes')
        # Agregamos los valores de las cargas en la gráfica (Si es mayor a 0.5, de color blanco, de lo contrario, de color negro):
        fig3.update_yaxes(tickmode='linear')
        for i in range(0, CargasComponentess.shape[0]):
            for j in range(0, CargasComponentess.shape[1]):
                if CargasComponentess.iloc[i,j] >= 0.5:
                    color = 'white'
                else:
                    color = 'black'
                fig3.add_annotation(x=j, y=i, text=str(round(CargasComponentess.iloc[i,j], 4)), showarrow=False, font=dict(color=color))
        

        return MEstandarizada.to_dict('records'), fig, fig2, fig3
    
    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate






