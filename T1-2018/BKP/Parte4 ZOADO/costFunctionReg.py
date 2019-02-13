#import pandas as pd
#import numpy as np

import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from mapFeature import readData
from mapFeature import mapFeature
#from costFunctionReg import costFunctionReg
#from costFunctionReg import gd_reglog_reg  

def sigmoide(z):
    return (1.00 / (1.00 + np.exp(-z)))



def costFunctionReg(theta, X, y, lbd):
    # Conversão dos parâmetros da função em matrizes
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)


    # Quantidade de exemplos (m)
    m = len(X)

    #hipótese (h(x))
    h = sigmoide(X * theta.T)
    
    # Cálculo do custo     
    custo1 = np.multiply(y, np.log(h))
    custo0 = np.multiply((1-y), np.log(1-h))
    custo = -1/m * np.sum(custo1 + custo0) 

    # Calculo da parcela de regularização
    reg = lbd / 2*m * np.sum(np.power(theta, 2))   ##sum((theta ** 2)) 

    # Retorna o custo regularizado
    #return custo + reg 
    custo_reg = custo + reg 



    #def grad_reglog(X, y, theta):
    #Converte os dados recebidos em matrizes
    #theta = np.matrix(theta)
    #X = np.matrix(X)
    #y = np.matrix(y)

    #Armazena na variável parametros a quantidade de itens existentes na matriz theta.
    parametros = np.size(theta)
    
    #Cria a array grad, preenchida com zeros e a mesma quantidade de itens que há em parâmetros.
    grad = np.zeros(parametros)
    
    #Calcula a distância (erro) entre os valores obtidos na hiótese thetaX e os valores corretos (y) do conjunto de dadaos de treinamento.
    erro = sigmoide(X * theta.T) - y
    
    #Calcula a derivada parcial de X
    for j in range(parametros):
        term = np.multiply(erro, X[:,j]) 
        grad[j] = np.sum(term) / len(X) 
    
    #Retorna o valor calculado
    return custo_reg, grad


def gd_reglog_reg(theta, X, y, lbd):
    import scipy.optimize as opt
    # Calcula o gradiente descendente e armazena na variÃ¡vel result
    #result = opt.fmin_tnc(func=costFunctionReg, x0=theta, args=(X, y, lbd), disp=0)
    result = opt.fmin_tnc(func=costFunctionReg, args=([X, y, lbd]), x0=theta, disp=0)
    result = np.array(result[0])
    theta_otim = result
    return theta_otim


######################3333


# Lê os dados do arquivo 
X, y, data = readData()

#Plota o gráfico
#código aqui

#Inicializa o hiperparâmetro Lambda
lbd = 0

#Engenharia de parâmetros: chama a função mapFeature, 
# que aumenta as características (X) para 28 colunas
X = mapFeature(X)

#Teste da função do custo e gradiente 
theta = np.zeros(28)
lbd = 0
#print("theta ótimo: ", end="" )
#print(gd_reglog_reg(theta, X, y, lbd))


