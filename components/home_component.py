import dash_bootstrap_components as dbc
from dash import html

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("MineriAPP", className="display-4"),
                        html.P("La minería de datos es un campo de estudio que se centra en descubrir patrones, relaciones y conocimientos útiles a partir de conjuntos de datos masivos. Con el aumento exponencial en la generación y almacenamiento de datos en los últimos años, la minería de datos se ha vuelto cada vez más relevante y poderosa en diversas áreas, desde el comercio electrónico hasta la medicina y la investigación científica.", style={'text-align': 'justify'}),
                        html.P("Uno de los desafíos en la minería de datos es la aplicación de diferentes algoritmos y técnicas para extraer información valiosa de los conjuntos de datos. Aquí es donde entra en juego la interfaz gráfica de usuario, una herramienta que permite a los usuarios interactuar con los algoritmos de manera intuitiva y visual.", style={'text-align': 'justify'}),
                        html.P("La GUI proporciona una forma amigable y accesible de aplicar una variedad de algoritmos en el proceso de minería de datos. Algunos de los algoritmos comunes utilizados en esta área incluyen el Análisis Exploratorio de Datos (EDA), el Análisis de Componentes Principales (PCA), así como los árboles de decisión y los bosques aleatorios.", style={'text-align': 'justify'}),
                    ],
                    width={"size": 6, "order": 1},
                    className="intro-column",
                ),
                dbc.Col(
                    dbc.Carousel(
                        items=[
                            {"key": "1", "src": "/static/images/carrusel1.jpg"},
                            {"key": "2", "src": "/static/images/carrusel2.jpg"},
                            {"key": "3", "src": "/static/images/carrusel3.jpg"},
                        ],
                        controls=True,
                        indicators=True,
                        interval=2000,
                    ),
                    width={"size": 6, "order": 2},
                    className="carousel-column",
                ),
            ],
            className="mb-5 mt-5",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardImg(src="https://media.licdn.com/dms/image/C4E0DAQEuPRNnsUAxGA/learning-public-crop_288_512/0/1635327651295?e=2147483647&v=beta&t=tG_wj5gly-8N_jAMnxpFyZE3W8Fl8fgphYpqLuLFNFc", top=True, style={"height": "200px"}),
                                dbc.CardBody(
                                    [
                                        html.H5("EDA", className="card-title"),
                                        html.P("Análisis Exploratorio de Datos (EDA) es una técnica que ayuda a entender la estructura de los datos y descubrir patrones, distribuciones y relaciones significativas. Ayuda a identificar características importantes y a preparar los datos para el modelado y la toma de decisiones.", style={'text-align': 'justify'}),
                                    ]
                                ),
                            ],
                            className="mb-4 card-effect",
                        ),
                    ],
                    width={"size": 4, "order": 1},
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardImg(src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRm6hU0agptmSjKyxdDZYojzngn_NAby_qiRb6m2A5iR7kkQiiUQ8bhQgWE4R_yyLfDYt8&usqp=CAU", top=True, style={"height": "200px"}),
                                dbc.CardBody(
                                    [
                                        html.H5("PCA", className="card-title"),
                                        html.P("El Análisis de Componentes Principales (PCA) es una técnica de reducción de dimensionalidad que ayuda a identificar las variables más relevantes en un conjunto de datos. Permite simplificar la representación de los datos mientras mantiene la mayor cantidad de información posible.", style={'text-align': 'justify'}),
                                    ]
                                ),
                            ],
                            className="mb-4 card-effect",
                        ),
                    ],
                    width={"size": 4, "order": 2},
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardImg(src="https://fhernanb.github.io/libro_mod_pred/libro_mod_pred_files/figure-html/unnamed-chunk-27-1.png", top=True, style={"height": "200px"}),
                                dbc.CardBody(
                                    [
                                        html.H5("Árboles de Clasificación", className="card-title"),
                                        html.P("Los árboles de clasificación son modelos de aprendizaje automático que utilizan decisiones basadas en reglas para clasificar ejemplos en categorías. Son fáciles de interpretar y pueden manejar tanto variables numéricas como categóricas, lo que los hace útiles en diversos problemas de clasificación.", style={'text-align': 'justify'}),
                                    ]
                                ),
                            ],
                            className="mb-4 card-effect",
                        ),
                    ],
                    width={"size": 4, "order": 3},
                ),
            ],
            className="justify-content-center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardImg(src="https://fhernanb.github.io/libro_mod_pred/images/ilustracion_arb_regresion.png", top=True, style={"height": "200px"}),
                                dbc.CardBody(
                                    [
                                        html.H5("Árboles de Regresión", className="card-title"),
                                        html.P("Los árboles de regresión son modelos de aprendizaje automático utilizados para predecir valores numéricos. Dividen los datos en subconjuntos utilizando reglas de división y asignan una predicción numérica a cada subconjunto. Son útiles cuando se busca estimar o predecir valores continuos.", style={'text-align': 'justify'}),
                                    ]
                                ),
                            ],
                            className="mb-4 card-effect",
                        ),
                    ],
                    width={"size": 4, "order": 1},
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardImg(src="https://fhernanb.github.io/libro_mod_pred/images/rand_forest_clas_reg.png", top=True, style={"height": "200px"}),
                                dbc.CardBody(
                                    [
                                        html.H5("Bosques de Clasificación", className="card-title"),
                                        html.P("Los bosques de clasificación son conjuntos de árboles de decisión que trabajan en conjunto para realizar predicciones más precisas. Cada árbol en el bosque toma una decisión y la clasificación final se determina por votación o promedio de las predicciones de todos los árboles.", style={'text-align': 'justify'}),
                                    ]
                                ),
                            ],
                            className="mb-4 card-effect",
                        ),
                    ],
                    width={"size": 4, "order": 2},
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardImg(src="https://fhernanb.github.io/libro_mod_pred/images/rand_forest_clas_reg.png", top=True, style={"height": "200px"}),
                                dbc.CardBody(
                                    [
                                        html.H5("Bosques de Regresión", className="card-title"),
                                        html.P("Los bosques de regresión son conjuntos de árboles de regresión que trabajan en conjunto para realizar predicciones más precisas de valores numéricos. Cada árbol en el bosque estima un valor y el resultado final se calcula mediante promedio o votación ponderada.", style={'text-align': 'justify'}),
                                    ]
                                ),
                            ],
                            className="mb-4 card-effect",
                        ),
                    ],
                    width={"size": 4, "order": 3},
                ),
            ],
            className="justify-content-center",
        ),
    ],
    className="container main-container",
)