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

    def getEscala(self, escala = 'StandardScaler' ):
        df_numeric = self.df.select_dtypes(include=['float64', 'int64'])
        if escala == 'StandardScaler':
            estandarizar = StandardScaler()
        elif escala == 'MinMaxScaler':
            estandarizar = MinMaxScaler()
        MEstandarizada = estandarizar.fit_transform(self.df)
        dataFrameMEstandarizada = pd.DataFrame(MEstandarizada, columns=self.df.columns).reset_index().rename(columns={"index": "Column"})
        # dataFrameMEstandarizada['Column'] = dataFrameMEstandarizada['Column'].astype(str)
        # mat_stand_dataframe = estandarizar_datos(df_numeric, escala)
        self.pca = PCA(n_components=8).fit(dataFrameMEstandarizada)
        self.varianza = self.pca.explained_variance_ratio_
        return dataFrameMEstandarizada

    def getGraficaVarianzaExplicada(self):
        colors = px.colors.qualitative.Plotly
        fig = go.Figure()
        for i in range(1, self.varianza.size + 1):
            fig.add_trace(go.Bar(x=[i], y=[self.varianza[i-1]*100],marker_color=colors[i % len(colors)], legendgroup=f'Componente {i}', name=f'Componente {i}'))

        fig.update_layout(
            title='Varianza explicada por componente',
            xaxis=dict(title="Componentes Principales"),
            yaxis=dict(title="Varianza (%)")
        )

        # Se muestra el porcentaje de varianza de cada componente encima de su respectiva barra
        for i in range(1, self.varianza.size + 1):
            fig.add_annotation(x=i, y=self.varianza[i - 1] * 100, text=str(round(self.varianza[i - 1] * 100, 2)) + '%', yshift=10, showarrow=False, font_color='black')

        # Se agrega un scatter que pase por la varianza de cada componente
        fig.add_scatter(x=np.arange(1, self.varianza.size + 1, step=1), y=self.varianza * 100, mode='lines+markers', name='Varianza explicada', showlegend=False)

        # Eje X: valores
        fig.update_xaxes(tickmode='linear')
        return fig



