import base64, datetime, io, dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Container import Container

from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
from components.eda_component import Eda
from components.pca_component import Pca_Propio



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE, "/assets/styles.css"])

#componentes

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Guía", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Inicia Sesión", href="#"),
                dbc.DropdownMenuItem("Registrate", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="Autenticate",
        ),
    ],
    brand="MineriApp",
    brand_href="#",
    color="primary",
    dark=True,
)



inputs = html.Div(
    [
        dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Arrastra tu archivo o ',
            html.A('Selecciona Archivo')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'color': 'black'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload')

    ],
)

def parse_contents(contents, filename, date):
    global df

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),

        html.Div([
            html.Div(children='Tus datos son: '),
            dash_table.DataTable(
                data=df.to_dict('records'),
                page_size=10,
            )

        ]),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        # html.Div('Raw Content'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })
        html.P("Ahora que quieres hacer con ellos?"),
        menu
    ])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

presentacion = html.Div([
    html.H1("Bienvenido a MineriApp", className="text-center text-primary"),
    html.H4("Esta aplicación te ayudará a realizar minería de datos de una manera sencilla y rápida.", className="text-center"),
])

app.layout = html.Div(
    children=[
        navbar,
        presentacion,
        html.Div(
            style={'maxWidth': '80%', 'margin': '0 auto'},
            children=[
                inputs,
                html.Div(id='children', style={"marginLeft": "20px"}),
                html.Div(id='output-data-upload'),
            ]
        )
    ]
)


#Create a menu of tree buttons
menu  = html.Div([
            dbc.ButtonGroup([
            dbc.Button("EDA",  id="eda-button", n_clicks=0),
            dbc.Button("PCA", id="pca-button"),
            dbc.Button("Bosques", id="bosques-button"),
        ]),
            html.Div([
            html.Div(id="eda-resultado"),
            html.Div(id="pca-resultado"),
            html.Div(id="bosques-resultado"),
        ]),
    ])

@app.callback(
    Output("pca-resultado", "children"),
    [Input("pca-button", "n_clicks")],
)
def show_pca(n_clicks):
    if n_clicks > 0:
        outPCA = Pca_Propio(df)
        return [
            html.Div([
                html.H2("Análisis de Componentes Principales "),
                html.H4("Datos escalados: "),
                dash_table.DataTable(
                    data = outPCA.getEscala('StandardScaler').to_dict('records'),
                    page_size=10,
                    columns=[{'name': i, 'id': i} for i in outPCA.getEscala('StandardScaler').columns]
                ),
                dcc.Graph(
                    figure = outPCA.getGraficaVarianzaExplicada()
                ),
                dcc.Graph(
                    figure = outPCA.graficaVarianzaAcumulada()
                ),
            ])
        ]
    else:
        return ''

@app.callback(
    Output("eda-resultado", "children"),
    [Input("eda-button", "n_clicks")],
)
def show_eda(n_clicks):
    if n_clicks > 0:
        out = Eda(df)
        return [
            html.Div([
                html.H2("Análisis Exploratorio de Datos "),
                html.P(str(out.columnasYfilas())),
                html.H4("Tipos de datos: "),
                dash_table.DataTable(
                    data = out.getTypes().to_dict('records'),
                    page_size=10,
                    columns=[{'name': i, 'id': i} for i in out.getTypes().columns]
                ),
                html.H4("Valores Nulos: "),
                dash_table.DataTable(
                    data = out.getNulos().to_dict('records'),
                    page_size=10,
                    columns=[{'name': i, 'id': i} for i in out.getNulos().columns]
                ),
                html.H4("Datos estadísticos: "),
                dash_table.DataTable(
                    data = out.getDescribe().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in out.getDescribe().columns]
                ),
                html.H4("Histogramas: "),
                html.Div([
                    "Selecciona la o las variables para mostrar su histograma:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos por defecto todas las columnas numéricas, a partir de la segunda
                        value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][1:1],
                        id='value-histograma-eda',
                        multi=True
                    ),
                    dcc.Graph(id='histograma-eda'),
                ]),
            ]),
            html.H4("Matriz de correlación: "),
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
        ]
    else:
        return ''

@app.callback(
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


if __name__ == '__main__':
    app.run_server(debug=True)
