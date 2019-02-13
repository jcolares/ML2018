
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from mapFeature import readData
from mapFeature import mapFeature
#from costFunctionReg import costFunctionReg
from costFunctionReg import gd_reglog_reg  
from costFunctionReg import sigmoide


# Lê os dados do arquivo 
X, y, data = readData()

#Separa os dados obtidos do arquivo em 2 conjuntos (dataframes), 
# de acordo com o conteúdo da coluna "Resultado".  
positivo = data[data['Resultado'].isin([1])]  
negativo = data[data['Resultado'].isin([0])]

#Prepara dados para plotar a decision boundary
'''
Xfeat = mapFeature(X)
theta = np.zeros(28)
lbd = 0
tetha_otim = gd_reglog_reg(theta, Xfeat, y, lbd)
tetha_otim = np.array(tetha_otim)
v = np.array(np.arange(-1, 1.5, 0.1))
w = np.array(np.arange(-1, 1.5, 0.1))
VV, WW = np.meshgrid(v, w)
print(np.shape(WW))
#Z = np.multiply(v, w)
Z = np.concatenate((v, w), axis =0)
Z = Z.reshape(2, 25)
print("Z ",np.shape(Z))
#vv,ww = np.meshgrid(v, w, sparse=True)
Zfeat  = mapFeature(Z.T)
print(np.shape(Zfeat))
print("tetha ",np.shape(tetha_otim))
Zoto = Zfeat.dot(tetha_otim).T
print("shape ", np.shape(Zoto), Zoto)
print(np.shape(v))
#z[i][j] = np.dot(mapFeature(np.array([v[i]]),np.array([w[j]])),theta) 
#Z = mapFeature(Z)
#Z = Z.dot(tetha_otim.T)
#print (v, w, z)

fig = plt.figure()
ax = fig.subplots()
ax.axis([-1, 1.5, -1, 1.5])
#ax.contour(v, w, ,Z)
ax.contour(VV, WW)
plt.show()


v = np.linspace(-1, 1.5, 0.1)
w = np.linspace(-2, 1.5, 0.1)
vw = np.meshgrid(v, w)
print(vw)

Xfeat = mapFeature(X)
theta = np.zeros(28)
lbd = 0
tetha_otim = gd_reglog_reg(theta, Xfeat, y, lbd)
h = Xfeat.dot(tetha_otim)
b = np.around(1 / (1 + np.exp(-h)), decimals=1)
b = np.matrix(b).T
X = np.concatenate((X,b), axis=1)

'''
Xfeat = mapFeature(X)
theta = np.zeros(28)
lbd = 0
'''
theta_otim = gd_reglog_reg(theta, Xfeat, y, lbd)
theta_otim = np.array(theta_otim)

#Plota o gráfico de dispersão conforme o item 4.1 do enunciado.
fig = plt.figure()
ax = fig.subplots()  
ax.axis([-1, 1.5, -1, 1.5])
ax.scatter(positivo['Test 1'], positivo['Test 2'], s=50, c='k', marker='+', label='y=1')  
ax.scatter(negativo['Test 1'], negativo['Test 2'], s=50, c='y', marker='o', label='y=0')  
ax.plot(0,0, 'rx')
ax.legend()  
ax.set_xlabel('Microchip Test 1')  
ax.set_ylabel('Microchip Test 2')  

# Create grid space
u = np.linspace(-1,1.5,50)
v = np.linspace(-1,1.5,50)
z = np.zeros((len(u),len(v)))

# Evaluate z = theta*x over values in the gridspace
for i in range(len(u)):
    for j in range(len(v)):
        z[i][j] = np.dot(mapFeature(np.array([u[i] ,v[j]]).reshape(-1, 1) ),theta_otim)
        #z[i][j] = np.dot(mapFeature(np.array([u[i]]),np.array([v[j]])),theta)
# Plot contour
ax.contour(u,v,z,levels=[0])
#fig.savefig('plot4.4.png')
fig.show()
'''

#CoLA
"""
def plotBoundary(mytheta, myX, myy, mylambda=0.):
Function to plot the decision boundary for arbitrary theta, X, y, lambda value
Inside of this function is feature mapping, and the minimization routine.
It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
And for each, computing whether the hypothesis classifies that point as
True or False. Then, a contour is drawn with a built-in pyplot function.
"""
#theta, mincost = optimizeRegularizedTheta(mytheta,myX,myy,mylambda)
theta_o, custo = costFunctionReg(theta, Xfeat, y, lbd)
xvals = np.linspace(-1,1.5,50)
yvals = np.linspace(-1,1.5,50)
zvals = np.zeros((len(xvals),len(yvals)))
for i in xrange(len(xvals)):
    for j in xrange(len(yvals)):
        myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
        zvals[i][j] = np.dot(theta,myfeaturesij.T)
zvals = zvals.transpose()

u, v = np.meshgrid( xvals, yvals )
mycontour = plt.contour( xvals, yvals, zvals, [0])
#Kind of a hacky way to display a text on top of the decision boundary
myfmt = { 0:'Lambda = %d'%mylambda}
plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
plt.title("Decision Boundary")

plt.figure(figsize=(12,10))
plt.subplot(221)
plotData()
