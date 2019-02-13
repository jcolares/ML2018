import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
#from mapFeature import readData
#from mapFeature import mapFeature
#from costFunctionReg import costFunctionReg
#from costFunctionReg import gd_reglog_reg  

def sigmoide(z):
    return (1.00 / (1.00 + np.exp(-z)))


def costFunctionReg(theta, X, y, lbd):
    # Conversão dos parÃ¢metros da função em matrizes
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    # Quantidade de exemplos (m)
    m = len(X)

    #hipótese (h(x))
    h = sigmoide(X * theta.T)
    
    # CÁCLULO DO CUSTO REGULARIZADO
    # Cálculo do custo     
    '''
    custo1 = np.multiply(y, np.log(h))
    custo0 = np.multiply((1-y), np.log(1-h))
    custo = np.sum(custo1 + custo0) 
    '''
    grad0 = np.multiply(-1 * y, np.log(sigmoide(X * theta.T)))
    grad1 = np.multiply((1 - y), np.log(1.00 - sigmoide(X * theta.T)))
    custo =  np.sum(grad0 - grad1) / m

    # Calculo da parcela de regularização
    #theta1 = np.matrix(theta[:,1:])
    #print(theta1)
    reg = lbd / 2*m * np.sum(np.power(theta[:,1:], 2))   ##sum((theta ** 2)) 

    # Retorna o custo regularizado
    custo_reg =  (1/m) * (custo + reg )

    #--->return custo + reg 
    # teste --> custo_reg = -(1/m) * (np.sum(custo1 + custo0) + reg) 

    #CÁLCULO DO GRADIENTE
    #Armazena na variável parametros a quantidade de itens existentes na matriz theta.
    parametros = np.size(theta)
    
    #Cria a array grad, preenchida com zeros e a mesma quantidade de itens que em parametros.
    grad = np.zeros(parametros)
    
    #Calcula a distancia (erro) entre os valores obtidos na hipotese (theta.X) 
    # e os valores corretos (y) do conjunto de dadaos de treinamento.
    erro = sigmoide(X * theta.T) - y
    
    #Calcula a derivada parcial de X 
    for j in range(parametros):
        term = np.multiply(erro, X[:,j]) 
        grad[j] = np.sum(term) / len(X) 
    

    #RETORNA OS VALORES CALCULADOS PARA CUSTO REGULARIZADO E GRADIENTE
    return custo_reg, grad


'''
def computeCost(theta, X, y, lbd):
    # Conversão dos parÃ¢metros da função em matrizes
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    # Quantidade de exemplos (m)
    m = len(X)

    #hipótese (h(x))
    h = sigmoide(X * theta.T)
    
    # CÁCLULO DO CUSTO REGULARIZADO
    # Cálculo do custo     

    custo1 = np.multiply(y, np.log(h))
    custo0 = np.multiply((1-y), np.log(1-h))
    custo = np.sum(custo1 + custo0) 
  
    grad0 = np.multiply(-1 * y, np.log(sigmoide(X * theta.T)))
    grad1 = np.multiply((1 - y), np.log(1.00 - sigmoide(X * theta.T)))
    custo =  np.sum(grad0 - grad1) / m

    # Calculo da parcela de regularização
    #theta1 = np.matrix(theta[:,1:])
    #print(theta1)
    reg = lbd / 2*m * np.sum(np.power(theta[:,1:], 2))   ##sum((theta ** 2)) 

    # Retorna o custo regularizado
    custo_reg =  (1/m) * (custo + reg )
    return custo_reg
'''


#Hypothesis function and cost function for logistic regression#Hypoth 
def h(mytheta,myX): #Logistic hypothesis function
    return sigmoide(np.dot(myX,mytheta))


def computeCost(mytheta,myX,myy,mylambda = 0.): 
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    y is a matrix with m- rows and 1 column
    Note this includes regularization, if you set mylambda to nonzero
    For the first part of the homework, the default 0. is used for mylambda
    """
    m = len(myX)
    #note to self: *.shape is (rows, columns)
    term1 = np.dot(-np.array(myy).T,np.log(h(mytheta,myX)))
    term2 = np.dot((1-np.array(myy)).T,np.log(1-h(mytheta,myX)))
    regterm = (mylambda/2) * np.sum(np.dot(mytheta[1:].T,mytheta[1:])) #Skip theta0
    return float( (1./m) * ( np.sum(term1 - term2) + regterm ) )