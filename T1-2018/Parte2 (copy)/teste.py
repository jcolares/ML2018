import os, sys
import pandas as pd
import numpy as np
from normalizacao import normalizar_caracteristica
from numpy.linalg import norm
from custo_reglin_multi import custo_reglin_multi
from gd_reglin_multi import gd

x = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
x = [[1000,2000,3000],[1,2,3],[1,2,3],[1,2,3]]
#y = [[3,4],[3,4],[3,4],[3,4],[3,4]]
th = [0,1,10]
y = [6,6,6,6]
m= len(x)

x = np.array(x)
th = np.array(th)
y = np.array(y)
alpha = 0.01
epochs = 500

X_norm, y_norm, mean_X, std_X, mean_y, std_y = normalizar_caracteristica(x, y) 
x = X_norm

j = custo_reglin_multi(x,y,th)

print(" ")
print("Custo inicial, com theta = ",th," : ", j)
print(" ")

gd = gd(x, y, alpha, epochs, th)

print("Custo final e valores de theta após ",epochs, " epochs:")
print(gd)
print(" ")
