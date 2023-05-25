import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
# Para generar y almacenar los gráficos dentro del cuaderno
import plotly.graph_objects as go
import json

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Pca():
    def __init__(self, df):
        self.df = df

    def getEscala(self, escala):
        if escala == 'StandardScaler':
            estandarizar = StandardScaler()
            MEstandarizada = estandarizar.fit_transform(self.df)
        elif escala == 'MinMaxScaler':
            estandarizar = MinMaxScaler()
            MEstandarizada = estandarizar.fit_transform(self.df)
        return MEstandarizada