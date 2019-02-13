#coding=ISO8859-1
#declara��o de encoding para possibilitar a utiliza��o desse arquivo vom o Linux

import os, sys
import numpy as np
from custo_reglin_uni import custo_reglin_uni
# }

dadosX = [1,2,3,4,5]
dadosY = [1,2,3,4,5]
X = np.array(dadosX)
Y = np.array(dadosY)
theta = 1
print(custo_reglin_uni(X,Y,theta))

'''
Se a fun��o de custo estiver correta, o resutado que ela retornar� ser� 0.00,
pois a dist�ncia entre duas arrays iguais, com um theta de valor neutro (theta = 1), tem que ser zero.
Qualquer valor diferente indica erro de c�lculo na fun��o de custo.
