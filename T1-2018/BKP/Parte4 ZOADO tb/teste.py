import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from costFunctionReg import computeCost

def lerDados():
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



def optimizeRegularizedTheta(theta, X, y, lbd):
    from scipy import optimize
    result = optimize.minimize(computeCost, theta, args=(X, y, lbd),  method='BFGS', options={"maxiter":500, "disp":True} )
    return np.array([result.x]), result.fun
    ##theta, mincost = optimizeRegularizedTheta(initial_theta,mappedX,y)


X, y, dados = lerDados()

#Calcula o custo e o gradiente, conforme item 4.3 do enunciado
Xmap = mapFeature(X[:,0],X[:,1])
theta = np.ones(28)
lbd = 0
custo = computeCost(theta, Xmap, y, lbd)
print("Valor do custo: ",custo)
print("Obtido com lambda = ", lbd, " e theta = ", theta)
print(" ")
'''
#C�lculo dos valores �timos de theta, segunda parte do item 4.3.
theta_ot, custo = optimizeRegularizedTheta(theta, Xmap, y, lbd)
#good# theta_ot = otimizarTheta(theta, Xmap, y, lbd)
#print("Valores otimizados de theta: ", theta_ot)
#custo, grad = costFunctionReg(theta_ot, Xmap, y, lbd)
#print("Valor do custo: ",custo)
#print("Obtidos com lambda = ", lbd)
#print(" ")

# Impress�o do esbo�o da fronteira de decis�o
plotDecisionBoundary(X, y, dados, theta_ot)
'''
