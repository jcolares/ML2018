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
    thresh = cm.max()/2  # limite usado para definir quando o texto será branco ou preto
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
    plt.figure(figsize=[9,4])
    plt.suptitle(titulo)
    plt.subplot(121)
    plt.title('Curva de aprendizado (Treinamento)')
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
    thresh = cm.max()/2  # limite usado para definir quando o texto será branco ou preto
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.title('Confusion matrix (Teste) - Acurácia: %4.2f'% acc)
    # Configura as cores do gráfico
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, cmap=cmap)     
  
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
    
    # Treina o modelo (1)
    clf = MLPClassifier(solver='adam',
                        alpha=0.2, # taxa de regularizacao
                        learning_rate_init=0.005, # taxa de aprendizado
                        learning_rate='adaptive', # taxa de aprendizado
                        hidden_layer_sizes=(60,), 
                        max_iter=300) 

    clf.fit(X_train_normalizd, y_train)
    loss_curve = clf.loss_curve_

    # Predicoes utilzando conjunto de teste
    y_pred_test = clf.predict(X_test_normalizd)


    cm = confusion_matrix(y_test, y_pred_test)
 
    plot_learning_curve(loss_curve, cm, 'adam, 60 neurons, reg=0.2, apr=0.005')
    

    # Treina o modelo (2)
    clf = MLPClassifier(solver='sgd',
                        alpha=0.001, # taxa de regularizacao
                        learning_rate_init= 0.2, # taxa de aprendizado
                        learning_rate='adaptive', # taxa de aprendizado
                        hidden_layer_sizes=(40,), 
                        max_iter=300)
    clf.fit(X_train_normalizd, y_train)
    loss_curve = clf.loss_curve_

    # Predicoes utilzando conjunto de teste
    y_pred_test = clf.predict(X_test_normalizd)

    cm = confusion_matrix(y_test, y_pred_test)
 
    plot_learning_curve(loss_curve, cm, 'sgd, 40 neurons, reg=0.001, apr=0.2')



if __name__ == "__main__":
	main()