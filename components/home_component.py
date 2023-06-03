import dash_bootstrap_components as dbc
from dash import html

layout = html.Div(
    [
        html.H1("Bienvenido a MineriApp", className="display-4"),
        html.P("Esta es la página de inicio de MineriApp, donde puedes encontrar información y enlaces a otras secciones de la aplicación."),
        dbc.Carousel(
            items=[
                {
                    "key": "1",
                    "src": "/static/images/eda.png",
                    "header": "EDA",
                    "caption": "Análisis Exploratorio de Datos",
                    "img_style": {"width": "100%", "height": "350px"},  # Ajusta el ancho al 100% y establece una altura fija de 200px
                },
                {
                    "key": "2",
                    "src": "/static/images/pca.jpg",
                    "header": "PCA",
                    "caption": "Análisis de Componentes Principales",
                    "img_style": {"width": "100%", "height": "350px"},  # Ajusta el ancho al 100% y establece una altura fija de 200px
                },
                {
                    "key": "3",
                    "src": "/static/images/arbol.webp",
                    "header": "Árboles de Decisión",
                    "caption": "Análisis de Árboles de Decisión",
                    "img_style": {"width": "100%", "height": "350px"},  # Ajusta el ancho al 100% y establece una altura fija de 200px
                },
            ],
            variant="dark",
            interval=2000,
            style={'width': '30%'},
            className="bg-opacity-25 p-2 m-auto align-middle bg-primary text-dark fw-bold rounded",
        )
    ],
    className="container mt-5"
)
