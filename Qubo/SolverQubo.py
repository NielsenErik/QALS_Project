import random
import numpy as np 
import pandas as pd
from Qubo import qubo_Matrix
from scipy.optimize import minimize 
from Subset_generator import subset_Vector
from Data_Rescaler import german_credit_data

def subset_array_generator (n, dim):
    tmp = np.zeros((n, dim))
    for i in range(n):
        tmp[i]=subset_Vector(dim,random.randrange(dim))
    subset_array = np.array(tmp)
    return subset_array

def qubo_Solver(n,alpha):
    q = qubo_Matrix(alpha, german_credit_data())
    rows, column = q.shape
    x = subset_array_generator(n, column)
    for i in range( len(x)):
        f = x[i].T*q*x[i]
    print(f)

qubo_Solver(100000, 0.3)