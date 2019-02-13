#### crednn.py 
#### Classificação de crédito /

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import itertools
import sklearn.preprocessing as prep 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Carrega os conjuntos de dados de teste e treinamento
traindata = pd.read_table('credtrain.txt', header=None)
testdata = pd.read_table('credtest.txt', header=None)

# Converte os conjutos de teste e treinamento em arrays (X e y)
X_train = np.array(traindata.iloc[:,0:11])
y_train = np.array(traindata.iloc[:,11])
X_test = np.array(testdata.iloc[:,0:11])
y_test = np.array(testdata.iloc[:,11])

print(np.sum(y_train), len(y_train))
print(np.sum(y_test), len(y_test))

'''


# Normaliza as features do conjunto de treinamento
X_train_prep = prep.normalize(X_train, axis = 0, return_norm=True) 
X_train_normalizd = X_train_prep[0]
X_train_norm = X_train_prep[1]

# Normaliza as features do conjunto de teste
X_test_normalizd = np.divide(X_test, X_train_norm)

print(X_train[0:19,3])
print(X_train_normalizd[0:19,3])
'''