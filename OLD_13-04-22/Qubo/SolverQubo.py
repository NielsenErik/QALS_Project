#!/usr/local/bin/python3

import numpy as np 
import pandas as pd

from Qubo_Matrix import qubo_Matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from Subset_generator import subset_Vector
from Data_Rescaler import german_credit_data, rescaledDataframe, vector_V

def subset_array_generator_per_k (n, dim, k):
    subset_array = np.zeros((n, dim))
    for i in range(n):
        subset_array[i]=subset_Vector(dim, k)
    return subset_array

def qubo_function (x, Q):
    f = np.matmul(np.matmul(-x, Q), x.T)
    return f

def qubo_solver_per_K(n, dim, k, alpha, inputData):
    #n = number of subsets, dim = matrix dimension, k = subset cardinality, alpha = weighting in the problem
    #inputData = data already scaled
    x = subset_array_generator_per_k(n, dim, k)
    Q = qubo_Matrix(alpha, inputData)
    function = np.zeros(n)
    for i in range(n):
        function[i] = qubo_function(x[i], Q)
    result = np.argmin(function)
    x_value = x[result]
    y = function[result]
    return x_value, y


def RFECV_solver():
    x = rescaledDataframe(german_credit_data())
    y = vector_V(german_credit_data())
    logReg = LogisticRegression()
    selector = RFECV(logReg, step = 1, cv = 3)
    selector = selector.fit(x, y)
    indexList = selector.get_support()
    featureList = np.where(indexList)[0]
    result = np.asarray(featureList)
    return result


    
def getResultForQubo(qubo_array):
    x_tmp = rescaledDataframe(german_credit_data())

    rows, _ = x_tmp.shape
    pos = np.where(qubo_array>0)
    pos = np.asarray(pos)
    _, column = pos.shape
    tmp_x = x_tmp[:,pos]
    x = tmp_x.reshape(rows, column)
    y = vector_V(german_credit_data())
    sss = StratifiedShuffleSplit(n_splits=10000, test_size=0.5, random_state=0)

    sss.get_n_splits(x, y)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    logReg = LogisticRegression(random_state=0).fit(x_train, y_train)
    #print(logReg.predict(x_test))
    print(logReg.score(x_test, y_test))
    score = logReg.score(x_test, y_test)
    return score

def getResultRFECV(RFE_array):
    x_tmp = rescaledDataframe(german_credit_data())
    rows, _ = x_tmp.shape
    pos = np.asarray(RFE_array)
    columns = len(RFE_array)
    tmp_x = x_tmp[:,pos]
    x = tmp_x.reshape(rows, columns)
    y = vector_V(german_credit_data())
    sss = StratifiedShuffleSplit(n_splits=10000, test_size=0.5, random_state=0)
    sss.get_n_splits(x, y)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    logReg = LogisticRegression(random_state=0).fit(x_train, y_train)
    #print(logReg.predict(x_test))
    print(logReg.score(x_test, y_test))
    score = logReg.score(x_test, y_test)
    return score

def print_test_Result():
    data = german_credit_data()
    print("QUBO K = 20")
    qubo_result1, f_value1 = qubo_solver_per_K(10000, 48, 20, 0.977, data)
    print(qubo_result1)
    print(f_value1)  
    getResultForQubo(qubo_result1)
    print(" ")
    print("QUBO K = 24")
    data1 = german_credit_data()
    qubo_result, f_value = qubo_solver_per_K(10000, 48, 24, 0.977, data1)
    print(qubo_result)
    print(f_value)  
    getResultForQubo(qubo_result)
    print(" ")
    print("RFECV")
    featureList = RFECV_solver()
    resRFE = getResultRFECV(featureList)
    print(featureList)
    print(resRFE)
    

def print_test_ResultFor(start, end):
    data = german_credit_data()
    
    for i in range(start, end):    
        print("QUBO K = ", i+1)
        qubo_result, f_value = qubo_solver_per_K(1000, 48, i, 0.977, german_credit_data())
        print(qubo_result)
        print(f_value)  
        getResultForQubo(qubo_result)
        print(" ")
    
    print("RFECV")
    featureList = RFECV_solver()
    resRFE = getResultRFECV(featureList)
    print(featureList)
    print(resRFE)
    
print_test_ResultFor(22,25)

