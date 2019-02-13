import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from custo_reglin_uni import custo_reglin_uni

#carrega os dados do arquivo
dataset = pd.read_csv('ex1data1.txt', header = None)

#acrescenta uma coluna preenchida com 1s
dataset.insert(0, 'Ones', 1)

#separa o dataset em dois: X com duas colunas e y com uma
cols = dataset.shape[1]
X = dataset.iloc[:,0:cols-1]
y = dataset.iloc[:,cols-1:cols]

#converte os datasets em arrays
X = np.array(X.values)
y = np.array(y.values)


# Valores de theta0 e theta1 informados no enunciado do trabalho
theta0 = np.arange(-10, 10, 0.01)
theta1 = np.arange(-1, 4, 0.01)

# Cria uma array bidimensional J inicializada com zeros
J = np.zeros((len(theta0), len(theta1)))

# Preenche J com valores crescentes e sucessivos de theta0 e theta1
for i in range(len(theta0)):
    for j in range(len(theta1)):
        t = [[theta0[i]], [theta1[j]]]
        J[i,j] = custo_reglin_uni(X, y, t)

# Transpõe J devido as funções contour/meshgrid
J = np.transpose(J)

#combina cada valor de theta0 com o correspondente em theta1 (um grid)
theta0, theta1 = np.meshgrid(theta0, theta1)

# Plota a função de custo utilizando levels como logspace. Range -1 ~ 4 devido ao
# range de theta1 e 20 pois o theta0 tem 20 valores (-10 até 10)
fig = plt.figure()
#fig, ax = plt.subplots()
ax = fig.subplots()
ax.contour(theta0, theta1, J, levels=np.logspace(-1, 4, 20), color='blue')
#ax.plot(theta[0,0], theta[1,0], 'rx')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.savefig('plot1.2.png')
plt.show()