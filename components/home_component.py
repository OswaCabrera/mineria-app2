import dash_bootstrap_components as dbc
from dash import html

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Título", className="display-4"),
                        html.P("Texto de introducción"),
                    ],
                    width={"size": 6, "order": 1},
                    className="intro-column",
                ),
                dbc.Col(
                    dbc.Carousel(
                        items=[
                            {"key": "1", "src": "/static/images/slide1.svg"},
                            {"key": "2", "src": "/static/images/slide2.svg"},
                            {"key": "3", "src": "/static/images/slide3.svg"},
                        ],
                        controls=True,
                        indicators=True,
                        interval=3000,
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
                                dbc.CardImg(src="/static/images/card1.jpg", top=True),
                                dbc.CardBody(
                                    [
                                        html.H5("Card 1", className="card-title"),
                                        html.P("Texto de la card 1"),
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
                                dbc.CardImg(src="/static/images/card2.jpg", top=True),
                                dbc.CardBody(
                                    [
                                        html.H5("Card 2", className="card-title"),
                                        html.P("Texto de la card 2"),
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
                                dbc.CardImg(src="/static/images/card3.jpg", top=True),
                                dbc.CardBody(
                                    [
                                        html.H5("Card 3", className="card-title"),
                                        html.P("Texto de la card 3"),
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
