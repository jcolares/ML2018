import pandas as pd
import numpy as np

def sigmoide(z):
    return (1.00 / (1.00 + np.exp(-z)))



def costFunctionReg(X, y, lbd, theta):
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
    return custo + reg 



def grad_reglog(X, y, theta):
    #Converte os dados recebidos em matrizes
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

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
    return grad
    


def gd_reglog_reg(X, y, lbd, theta):
    import scipy.optimize as opt

    #theta = np.array(theta)
    
    # Calcula o gradiente descendente e armazena na variÃ¡vel result
    result = opt.fmin_tnc(func=costFunctionReg, args=(X, y, lbd), x0=theta, fprime=grad_reglog(X,y,theta), disp=0)
    result = np.array(result[0])
    theta_otim = result
    return theta_otim



