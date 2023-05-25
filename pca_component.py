import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
# Para generar y almacenar los gráficos dentro del cuaderno
import plotly.graph_objects as go
import json
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Pca_Propio():
    def __init__(self, df):
        self.df = df

    def getEscala(self, escala):
        global MEstandarizada
        if escala == 'StandardScaler':
            estandarizar = StandardScaler()
            MEstandarizada = estandarizar.fit_transform(self.df)
        elif escala == 'MinMaxScaler':
            estandarizar = MinMaxScaler()
            MEstandarizada = estandarizar.fit_transform(self.df)
        dataFrameMEstandarizada = pd.DataFrame(MEstandarizada, columns=self.df.columns).reset_index().rename(columns={"index": "Column"})
        dataFrameMEstandarizada['Column'] = dataFrameMEstandarizada['Column'].astype(str)
        return dataFrameMEstandarizada

    def getGraficaVarianza(self):
        global MEstandarizada
        varianza = 0
        fig = go.Figure()
        for i in range(1, varianza.size + 1):
            fig.add_trace(go.Bar(x=[i], y=[varianza[i-1]*100],marker_color=colors[i % len(colors)], legendgroup=f'Componente {i}', name=f'Componente {i}'))

        fig.update_layout(
            title='Varianza explicada por cada componente',
            xaxis=dict(title="Componentes Principales"),
            yaxis=dict(title="Varianza explicada (%)")
        )

        # Se muestra el porcentaje de varianza de cada componente encima de su respectiva barra
        for i in range(1, varianza.size + 1):
            fig.add_annotation(x=i, y=varianza[i - 1] * 100, text=str(round(varianza[i - 1] * 100, 2)) + '%', yshift=10, showarrow=False, font_color='black')

        # Se agrega un scatter que pase por la varianza de cada componente
        fig.add_scatter(x=np.arange(1, varianza.size + 1, step=1), y=varianza * 100, mode='lines+markers', name='Varianza explicada', showlegend=False)

        # Eje X: valores
        fig.update_xaxes(tickmode='linear')
        return fig



