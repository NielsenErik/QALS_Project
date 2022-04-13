#!/usr/local/bin/python3

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression


def getAccuracy(best_subset, inputMatrix, inputVector, isQubo):
    x_tmp = inputMatrix
    rows, _ = x_tmp.shape
    pos = np.asarray(best_subset)
    if (isQubo == True):
        print("Running accuracy score QUBO")
        columns = len(pos[0])
    else:
        print("Running accuracy score RFECV")
        columns = len(pos)
    tmp_x = x_tmp[:,pos]
    x = tmp_x.reshape(rows, columns)
    y = inputVector
    sss = StratifiedShuffleSplit(n_splits=10000, test_size=0.5, random_state=0)
    sss.get_n_splits(x, y)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    logReg = LogisticRegression(random_state=0).fit(x_train, y_train)
    #print(logReg.predict(x_test))
    print(logReg.score(x_test, y_test))
    score = logReg.score(x_test, y_test)
    return score, columns

'''data = german_credit_data()
matrix = rescaledDataframe(data)
vector = vector_V(data)

qubo = qubo_Matrix(0.3, matrix, vector)

qubo_array=QUBOsolver(48, 0.99, matrix, vector, 1, 1000, simulation=True)
rfecv = RFECV_solver(matrix, vector)
print(qubo_array)
print(rfecv)

scoreQubo, feature_nQ = getAccuracy(qubo_array, matrix, vector, True)
scoreRfecv, feature_nR = getAccuracy(rfecv, matrix, vector, False)

print(" QUBO = ", scoreQubo, " Feature number = ", feature_nQ, " RFECV = ", scoreRfecv, " Feature number = ", feature_nR)'''