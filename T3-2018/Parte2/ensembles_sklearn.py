# AM-ensembles
# July 26, 2018

## Script adaptado de https://github.com/jaimeps/adaboost-implementation/blob/master/adaboost.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""
def print_error_rate(err):  
    print ('Error rate: Training: %.4f - Test: %.4f' % err)

""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), get_error_rate(pred_test, Y_test)

""" ADABOOST IMPLEMENTATION ================================================="""
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_i])]
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), get_error_rate(pred_test, Y_test)

""" NEW ADABOOST IMPLEMENTATION ================================================="""
def adaboost_ada(Y_train, X_train, Y_test, X_test, M):
    ada = AdaBoostClassifier(n_estimators=M, learning_rate=1)
    ada = ada.fit(X_train, Y_train)
    score_train = ada.score(X_train, Y_train)
    score_test = ada.score(X_test, Y_test)

    # Return error rate in train and test set
    return(1 - score_train, 1 - score_test)


""" BAGGING IMPLEMENTATION ================================================="""
def adaboost_bag(Y_train, X_train, Y_test, X_test, M):
    bag = BaggingClassifier(n_estimators=M) 
    bag = bag.fit(X_train, Y_train)
    score_train = bag.score(X_train, Y_train)
    score_test = bag.score(X_test, Y_test)

    # Return error rate in train and test set
    return(1 - score_train, 1 - score_test)


"""PLOT FUNCTION ==========================================================="""
def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6), color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Error rate vs number of iterations', fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')
    plt.show()

"""PLOT FUNCTION ==========================================================="""
def new_plot_error_rate(er_train, er_test, er_train_ada, er_test_ada):
    df_error = pd.DataFrame([er_train, er_test, er_train_ada, er_test_ada]).T
    df_error.columns = ['Training', 'Test', 'Scikit Training', 'Scikit Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6), color = ['lightblue', 'darkblue','lightgreen', 'darkgreen'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Error rate vs number of iterations', fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')
    plt.show()    

""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
    # Read data
    x, y = make_hastie_10_2()
    df = pd.DataFrame(x)
    df['Y'] = y

    # Split into training and test set
    train, test = train_test_split(df, test_size = 0.2) # separa 20% do dataset para testes
    X_train, Y_train = train.iloc[:,:-1], train.iloc[:,-1] # a última coluna vai para Y-train, as demais para X_train
    X_test, Y_test = test.iloc[:,:-1], test.iloc[:,-1] # idem, no conjunto de teste
    print('passei 1')
    
    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
    print('passei 2')
    
    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    er_train, er_test = [er_tree[0]], [er_tree[1]]

    x_range = range(10, 410, 10)
    for i in x_range:
        er_i = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
        er_train.append(er_i[0])
        er_test.append(er_i[1])
    print('passei 3')

    # Compare error rate vs number of iterations
    plot_error_rate(er_train, er_test)

    
    # Adaboost ada - utilizando AdaBoostClassifier
    er_train_ada, er_test_ada = [er_tree[0]], [er_tree[1]]
    new_range = range(10, 401, 10)
    for i in new_range:
        er_i = adaboost_ada(Y_train, X_train, Y_test, X_test, i)
        er_train_ada.append(er_i[0])
        er_test_ada.append(er_i[1])

    # Compare error rate vs number of iterations
    new_plot_error_rate(er_train, er_test, er_train_ada, er_test_ada)

    
    # Adaboost bagging - implementação do AdaBoostBaggingClassifier
    er_train_bag, er_test_bag = [], []
    new_range = range(10, 401, 10)
    for i in new_range:
        er_i = adaboost_bag(Y_train, X_train, Y_test, X_test, i)
        er_train_bag.append(er_i[0])
        er_test_bag.append(er_i[1])

    # Compare error rate vs number of iterations    
    plot_error_rate(er_train_bag, er_test_bag)
