#!/usr/local/bin/python3

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

def RFECV_solver(inputMatrix, inputVector):
     #inputMatrix = matrix from rescaledDataframe()
    #inputVector = vector from vector_V()
    #this function does Recursive feature selection with
    #cross validation as described in the papaer
    print("Running RFECV")
    x = inputMatrix
    y = inputVector
    logReg = LogisticRegression(max_iter=2000)
    selector = RFECV(logReg, step = 1, cv = 3)
    selector = selector.fit(x, y)
    indexList = selector.get_support()
    featureList = np.where(indexList)[0]
    result = np.asarray(featureList)
    
    return result