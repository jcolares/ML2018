import numpy as np
import scipy.optimize as opt
from custo_reglog import custo_reglog
from sigmoide import sigmoide

def gd_reglog(theta, X, y):
    #Converte os dados recebidos em matrizes
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    #Armazena na variável parametros a quantidade de itens existentes na matriz theta.
    parametros = int(theta.ravel().shape[1])
    #Cria a array grad, preenchida com zeros e a mesma quantidade de itens que há em parâmetros.
    grad = np.zeros(parametros)
    #Calcula a distância (erro) entre os valores obtidos na hiótese thetaX e os valores corretos (y) do conjunto de dadaos de treinamento.
    erro = sigmoide(X * theta.T) - y
    #Calcula a derivada parcial de X
    for i in range(parametros):
        term = np.multiply(erro, X[:,i]) 
        grad[i] = np.sum(term) / len(X) 
    #Retorna o valor calculado
    return grad

