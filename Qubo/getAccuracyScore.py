#!/usr/local/bin/python3

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from .utils import print_step


def getAccuracy(best_subset, inputMatrix, inputVector, isQubo, isRFECV , isAllFeature = False):
    x_tmp = inputMatrix
    rows, _ = x_tmp.shape
    pos = best_subset
    if (isQubo == True):
        print_step("Running accuracy score QUBO", "QUBO")
    elif(isRFECV == True):
        print_step("Running accuracy score RFECV", "RFECV")
    elif(isAllFeature == True):
        print_step("Running accuracy score for all feature", "ALL")         
    else:
        print_step("Running accuracy score QALS", "QALS")
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

    score = logReg.score(x_test, y_test)
    if (isQubo == True):
        buffer = str(score)
        print_step("Score: "+buffer, "QUBO")
        print_step("Done with score", "QUBO")
    elif(isRFECV == True):
        buffer = str(score)
        print_step("Score: "+buffer, "RFECV")
        print_step("Done with score", "RFECV")
    elif(isAllFeature == True):
        buffer = str(score)
        print_step("Score: "+buffer, "ALL")
        print_step("Done with score", "ALL")     
    else:
        buffer = str(score)
        print_step("Score: "+buffer, "QALS")
        print_step("Done with score", "QALS") 
         
    
    return score, columns
