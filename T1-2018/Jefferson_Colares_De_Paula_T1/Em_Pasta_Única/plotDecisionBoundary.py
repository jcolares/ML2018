import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from scipy import optimize

#carrega os dados do arquivo
dados = pd.read_csv('ex2data2.txt', header = None ,names=['Test 1', 'Test 2', 'Resultado'])

#Separa os dados obtidos do arquivo em 2 conjuntos (dataframes), 
# de acordo com o conteúdo da coluna "Resultado".  
positivo = dados[dados['Resultado'].isin([1])]  
negativo = dados[dados['Resultado'].isin([0])]

#separa o dataset em dois: X com duas colunas e y com uma
cols = dados.shape[1]
X = dados.iloc[:,0:cols-1]
y = dados.iloc[:,cols-1:cols]

#converte os datasets em arrays
X = np.array(X.values)
y = np.array(y.values)

#Inicializa parâmetros da função de custo
Xmap = mapFeature(X[:,0],X[:,1])
thetaInicial = np.zeros(28)
lambd = 1

#Executa função de otimização que utiliza a nossa função de custo costFunctionReg
minima = optimize.minimize(costFunctionReg, thetaInicial, args=(Xmap, y, lambd),  method='BFGS', options={"maxiter":500, "disp":False} ) 
theta = np.array(minima.x)

#Gera dados para plotar a fronteira de decisão
xs = np.linspace(-1,1.5,50)
ys = np.linspace(-1,1.5,50)
zs = np.zeros((len(xs),len(ys)))

#Cria os pares xy usados no gráfico
for i in range(len(xs)):
    for j in range(len(ys)):
        Xplot = mapFeature(np.array([xs[i]]),np.array([ys[j]]))
        zs[i][j] = np.dot(theta,Xplot.T)
zs = zs.transpose()    

#Plota o gráfico de dispersão conforme o item 4.1 do enunciado.
#Lambda = 0
fig = plt.figure()
ax = fig.subplots()  
ax.axis([-1, 1.5, -1, 1.5])
ax.scatter(positivo['Test 1'], positivo['Test 2'], s=50, c='k', marker='+', label='y=1')  
ax.scatter(negativo['Test 1'], negativo['Test 2'], s=50, c='y', marker='o', label='y=0')  
ax.legend()  
ax.set_xlabel('Microchip Test 1')  
ax.set_ylabel('Microchip Test 2')  
titulo = 'Com Lambda = ' + str(lambd)
nomeArquivo = 'plot4.3.' + str(lambd) + '.png'
plt.title(titulo)

#Plota a fronteira de decisão
plt.contour(xs, ys, zs, [0])
fig.savefig(nomeArquivo)
plt.show()    
