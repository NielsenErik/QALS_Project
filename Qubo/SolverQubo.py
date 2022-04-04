#!/usr/local/bin/python3
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler
from dwave.system.samplers.dwave_sampler import DWaveSampler
import dwave_networkx as dnx
import networkx as nx
import neal
import time
import numpy as np 
import pandas as pd
from sqlalchemy import func
from Qubo_Matrix import qubo_Matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from Subset_generator import subset_Vector
from Data_Rescaler import german_credit_data, rescaledDataframe, vector_V
import random

def subset_array_generator_per_k (n, dim, k):
    subset_array = np.zeros((n, dim))
    for i in range(n):
        subset_array[i]=subset_Vector(dim, k)
    return subset_array

def qubo_function (x, Q):
    f = -x.dot(Q).dot(x.T)
    return f

def qubo_solver_per_K(n, dim, k, alpha, inputData):
    x = subset_array_generator_per_k(n, dim, k)
    Q = qubo_Matrix(alpha, inputData)
    function = np.zeros(n+1)
    for i in range(n):
        function[i] = qubo_function(x[i], Q)
    result = np.argmin(function)
    x_value = x[result]
    y = function[result]
    return result, x_value, y

def qubo_solver(n, dim, alpha, inputData):
    results_array = np.zeros((dim, n))
    for i in range(dim):
        results_array[i] = qubo_solver_per_K(n, dim, i, alpha, inputData)
    return results_array
    
def getResult(qubo_array):
    x_tmp = rescaledDataframe(german_credit_data())
    x = qubo_array.T*x_tmp
    print(x)
    y = vector_V(german_credit_data())
    sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.5, random_state=0)
    sss.get_n_splits(x, y)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    logReg = LogisticRegression(random_state=0).fit(x_train, y_train)
    #print(logReg.predict(x_test))
    print(logReg.score(x_test, y_test))
    
data = german_credit_data()

#tmp = subset_array_generator_per_k(1000,48,24)
#print(tmp)
'''ì'''
pos, qubo_result, f_value = qubo_solver_per_K(100, 48, 24, 0.977, data)
#qubo_solver(1000, 48, 0.977, german_credit_data() )
#print(result)  
getResult(qubo_result)