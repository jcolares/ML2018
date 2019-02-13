#### crednn.py 
#### Classificação de crédito /

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import itertools
import sklearn.preprocessing as prep 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def plot_cm(cm):
    ## Figura
    # Configura os rótulos dos eixos 
    plt.xticks([0,1],['0','1'])
    plt.yticks([0,1],['0','1'])
    plt.xlabel('Predição')
    plt.ylabel('Real')
    # Configura os valores de cada classe
    thresh = cm.max()/2  # limite usado para definir quando o texto será claro ou escuro
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.title('Confusion matrix do classificador')
    # Configura as cores do gráfico
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, cmap=cmap) 
    plt.show()

def plot_learning_curve(loss_curve, cm, titulo):
    acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
    ## Figura
    plt.figure(figsize=[7.5,4])
    plt.suptitle(titulo)
    plt.subplot(121)
    plt.title('Curva de aprendizado')
    plt.xlabel('Iterações')
    plt.ylabel('Custo')
    plt.plot(loss_curve, label='loss')

    plt.subplot(122)
    # Configura os rótulos dos eixos 
    plt.xticks([0,1],['0','1'])
    plt.yticks([0,1],['0','1'])
    plt.xlabel('Predição')
    plt.ylabel('Real')
    # Configura os valores de cada classe
    thresh = cm.max()/2  # limite usado para definir quando o texto será claro ou escuro
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.title('Confusion matrix (Teste)' +'\n'+ 'Acurácia: %4.2f'% acc)
    # Configura as cores do gráfico
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, cmap=cmap)     
    plt.tight_layout()  
    plt.subplots_adjust(top=0.8)  
    plt.show()


def main():
    # Carrega os conjuntos de dados de teste e treinamento
    traindata = pd.read_table('credtrain.txt', header=None)
    testdata = pd.read_table('credtest.txt', header=None)

    # Converte os conjutos de teste e treinamento em arrays (X e y)
    X_test = np.array(testdata.iloc[:,0:11])
    y_test = np.array(testdata.iloc[:,11])

    X_train = np.array(traindata.iloc[:,0:11])
    y_train = np.array(traindata.iloc[:,11])

    # Normaliza as features do conjunto de treinamento
    X_train_prep = prep.normalize(X_train, axis = 0, return_norm=True) 
    X_train_normalizd = X_train_prep[0]
    X_train_norm = X_train_prep[1]

    # Normaliza as features do conjunto de teste
    X_test_normalizd = np.divide(X_test, X_train_norm)
    
    # Ajuste de parâmetros
    solver = 'adam'
    tx_reg = 0
    tx_apr = 0.005
    neuronios = 100

    # Treinamento do modelo (1)
    clf = MLPClassifier(solver=solver,
                        alpha=tx_reg, # taxa de regularizacao
                        learning_rate_init=tx_apr, # taxa de aprendizado
                        learning_rate='constant', # taxa de aprendizado
                        hidden_layer_sizes=neuronios,   
                        max_iter=300) 
    modelo = clf.fit(X_train_normalizd, y_train)
    loss_curve = clf.loss_curve_

    # Predicoes utilzando conjunto de teste
    #y_pred_train = modelo.predict(X_train_normalizd)
    y_pred_test = modelo.predict(X_test_normalizd)

    cm = confusion_matrix(y_test, y_pred_test)
 
    titulo = 'solver=%s, neurônios=%i, tx_regularização=%s, tx_aprendizado=%s' % (solver, neuronios, tx_reg, tx_apr)    
    plot_learning_curve(loss_curve, cm, titulo)




if __name__ == "__main__":
	main()