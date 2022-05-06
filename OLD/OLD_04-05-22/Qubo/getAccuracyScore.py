#!/usr/local/bin/python3

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression


def getAccuracy(best_subset, inputMatrix, inputVector, isQubo, isRFECV):
    x_tmp = inputMatrix
    rows, _ = x_tmp.shape
    pos = np.asarray(best_subset)
    if (isQubo == True):
        print("Running accuracy score QUBO")
        columns = len(pos[0])
    elif(isRFECV == True):
        print("Running accuracy score RFECV")
        columns = len(pos)
    else:
        #print("Running accuracy score Random Max")
        tmp = np.where(best_subset>0)
        pos = np.asarray(tmp)
        columns = len(pos[0])
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
    if(isQubo == True or isRFECV == True):
        print(logReg.score(x_test, y_test))
    score = logReg.score(x_test, y_test)
    return score, columns
