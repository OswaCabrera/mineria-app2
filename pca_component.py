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
            title='Varianza explicada',
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

    def graficaVarianzaAcumulada(self):
        for i in range(0, self.varianza.size):
            self.varAcumulada = sum(self.varianza[0:i+1])
            if self.varAcumulada >= 0.89:
                self.varAcumuladaPCA = (self.varAcumulada - self.varianza[i])
                self.numComponentesPCA = i - 1
                break
        # varianza_acumulada = grafico_varianza_acumulada(Varianza, varAcumuladaPCA, numComponentesPCA, relevancia)
        
        fig = go.Figure()

        x_range = np.arange(1, self.varianza.size + 1, step=1)
        y_range = np.cumsum(self.varianza)

        fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines+markers', marker=dict(size=10, color='blue'), name='Núm. Componente'))

        fig.update_layout(title='Varianza acumulada',
                        xaxis_title='Número de componentes',
                        yaxis_title='Varianza acumulada')

        fig.add_shape(type="line", x0=1, y0=0.89, x1=8 + 1, y1=0.89, line=dict(color="Red", width=2, dash="dash"))
        fig.add_shape(type="line", x0=8 + 1, y0=0, x1=8 + 1, y1=self.varAcumulada, line=dict(color="Green", width=2, dash="dash"))

        fig.add_annotation(x=8 + 1, y=self.varAcumulada, text=str(round(self.varAcumulada * 100, 1)) + f'%. {8 + 1} Componentes', showarrow=True, arrowhead=1)

        fig.add_trace(go.Scatter(x=x_range, y=y_range, fill='tozeroy', mode='none', name='Área bajo la curva', fillcolor='rgba(0, 147, 255, 0.44)'))

        fig.update_xaxes(range=[1, self.varianza.size], tickmode='linear')
        fig.update_yaxes(range=[0, 1.1],
                        tickmode='array',
                        tickvals=np.arange(0, 1.1, step=0.1))
        return fig




