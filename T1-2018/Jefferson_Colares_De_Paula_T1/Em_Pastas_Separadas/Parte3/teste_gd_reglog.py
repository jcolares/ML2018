import numpy as np  
import pandas as pd  
import scipy.optimize as opt
from custo_reglog import custo_reglog 
from custo_reglog import normalizar_caracteristicas
from gd_reglog import gd_reglog
from sigmoide import sigmoide
from predizer_aprovacao import predizer
from predizer_aprovacao import acuracia

# Carregando os dados do dataset e armazendo em um array. 
data = pd.read_csv('ex2data1.txt', header=None, names=['Prova 1', 'Prova 2', 'Aprovado'])  

#Separa as colunas do dataset em Características e Rótulos
X = data.iloc[:,0:2]  
y = data.iloc[:,2:3]

#Normaliza as características 
X_norm, X_mean, X_stdev, y_norm, y_mean, y_stdev  = normalizar_caracteristicas(X, y)

#Converte X e y em matrizes (sem normalização)
X = np.matrix(np.c_[np.ones((X.shape[0],1)), X])
y = np.matrix(y)

# Insere coluna de 1’s no conjunto de dados X
X_norm = np.c_[np.ones((X.shape[0],1)), X_norm]

# Inicia o vetor de parâmetros theta
theta = np.array([1, 4, 3],ndmin=2)

# Calcula o gradiente descendente e armazena na variável result
result = opt.fmin_tnc(func=custo_reglog, x0=theta, fprime=gd_reglog, args=(X_norm, y), disp=0)
result = np.array(result[0])
theta = result
# Calcula o custo utiilizando os parãmetros encontrados 
##print(custo_reglog(result[0], X_norm, y))

# Item 3.2.4 - Predição da probabilidade de aprovação
# de um aluno com as notas 45 e 85 (Esperado: 0.80)
notas = np.array([1, (45 - X_mean[0]) / X_stdev[0], (85 - X_mean[1]) / X_stdev[1]])
pred = sigmoide(notas.dot(theta))
print("Probabilidade de aprovação com notas 45 e 85: ", pred)

# Item 3.2.4 - Predição e Acurácia
acuracia(X_norm, y, theta)

