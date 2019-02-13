import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def readData():
    #carrega os dados do arquivo
    data = pd.read_csv('ex2data2.txt', header = None ,names=['Test 1', 'Test 2', 'Resultado'])

    #separa o dataset em dois: X com duas colunas e y com uma
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    #converte os datasets em arrays
    X = np.array(X.values)
    y = np.array(y.values)

    #Retorna as duas arrays e também o conjunto de dados obtido do arquivo, com cabeçalhos
    return(X, y, data)

 

def mapFeature(X):
     poly = PolynomialFeatures(degree = 6 )
     Z = poly.fit_transform(X)

     return Z


