import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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

J = np.zeros((len(theta0), len(theta1)))

for i in range(len(theta0)):
    for j in range(len(theta1)):
        t =[[theta0[i]],[theta1[j]]]
        J[i,j] = custo_reglin_uni(X, y, t)

J = np.transpose(J)
#theta0, theta1 = np.meshgrid(theta0, theta1)

# Comandos necessários para o matplotlib plotar em 3D
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plotando o gráfico de superfície
theta0, theta1 = np.meshgrid(theta0, theta1) 
surf = ax.plot_surface(theta0, theta1, J)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.savefig('plot1.3.png')
plt.show()