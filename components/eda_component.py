import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
# Para generar y almacenar los gráficos dentro del cuaderno
import plotly.graph_objects as go
import json

class Eda():
    def __init__(self, df):
        self.df = df
        self.data = 'Hola desde EDA'
        self.dataTypes = self.df.dtypes.to_dict()

    def columnasYfilas(self):
        return 'Filas: ' + str(self.df.shape[0]) + ' y Columnas: ' + str(self.df.shape[1])

    def getTypes(self):
        dtypes_df = pd.DataFrame(self.df.dtypes, columns=["Data Type"]).reset_index().rename(columns={"index": "Column"})
        dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)  # Convertir los tipos de datos a strings
        return dtypes_df

    def getNulos(self):
        dnull_df = pd.DataFrame(self.df.isnull().sum(), columns=["Number of Missing Values"]).reset_index().rename(columns={"index": "Column"})
        dnull_df['Number of Missing Values'] = dnull_df['Number of Missing Values'].sum()
        return dnull_df

    def getDescribe(self):
        ddescribe_df = self.df.describe().reset_index().rename(columns={"index": "Stat"})
        ddescribe_df['Stat'] = ddescribe_df['Stat'].astype(str)
        return ddescribe_df

    # Aún no agarra XD 
    def getInfo(self):
        dinfo_df = pd.DataFrame(self.df.info(), columns=["Data Type"]).reset_index().rename(columns={"index": "Colum"})
        return dinfo_df
    
    def getColumna(self, columna):
        return self.df[columna]
    
    def create_categorical_bar_charts(self):
        categorical_columns = self.df.select_dtypes(include='object').columns
        bar_charts = []
        for col in categorical_columns:
            if self.df[col].nunique() < 10:
                counts = self.df[col].value_counts()
                bar_chart = go.Bar(x=counts.index, y=counts.values, name=col)
                bar_charts.append(bar_chart)
        # Crear un objeto go.Figure con las gráficas de barras y un diseño personalizado
        figure = go.Figure(data=bar_charts, layout=go.Layout(title='Distribución de variables categóricas', xaxis=dict(title='Categoría'), yaxis=dict(title='Frecuencia'), hovermode='closest'))
        return figure
    