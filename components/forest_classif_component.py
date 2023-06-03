from dash import html


layout = html.Div(
    children=[
        html.H1("¡Hola desde el componente!"),
        html.P("Este es un componente independiente en Dash."),
    ]
)