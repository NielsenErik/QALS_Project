#!/usr/local/bin/python3
import numpy as np
from Correletion_Matrix_and_Vector import column_Correlation
from Data_Rescaler import rescaledDataframe, vector_V, german_credit_data
from Subset_generator import subset_Vector

def qubo_function (x_subset, alpha):
    try:
        if(alpha<0 and alpha >1):
            raise ValueError
        else:
            input_Matrix = rescaledDataframe(german_credit_data())
            v_array = vector_V(german_credit_data())
            rho_vector_V, rho_column = column_Correlation(input_Matrix, v_array)
            dim = len(rho_vector_V)
            a = 0
            b = 0
            for i in range(dim):
                a+=x_subset[i]*abs(rho_vector_V[i])
                for j in range(dim):
                    if(j != i):
                        b += x_subset[i]*x_subset[j]*abs(rho_column[i,j])      
            qubo = -(alpha*a-(1-alpha)*b)      
    except ValueError:
        print("Alpha value error")
    return qubo
    
def test (nIteration, alpha, dim, k):
    for i in range(nIteration):
        print(i+1, '. ',qubo_function(subset_Vector(dim, k), alpha))

i= 0        
print(test(10, 0.6, 48, 28))