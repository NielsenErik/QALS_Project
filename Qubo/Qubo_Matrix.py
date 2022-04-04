#!/usr/local/bin/python3
import numpy as np
from Correletion_Matrix_and_Vector import column_Correlation
from Data_Rescaler import rescaledDataframe, vector_V, german_credit_data
from Subset_generator import subset_Vector


def qubo_Matrix (alpha, inputData):
    ''' this function is to generate Q matrix for the qubo problem'''
    
    '''Data preprocessing and correlation matrix and vector'''
    
    input_Matrix = rescaledDataframe(inputData)
    v_array = vector_V(inputData)
    rho_vector_V, rho_column = column_Correlation(input_Matrix, v_array)
    
    '''Qubo initiaization and creation'''
    
    dim = len(rho_vector_V)
    qubo = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if(j != i):
                qubo[i,j] = -(1-alpha)*rho_column[i,j]
            else:
                qubo[i,j] = alpha*rho_vector_V[i]  
                  
    return qubo

def test_qubo(alpha, inputData):
    #Funzione di test per creare un QUBO matrix
    b = qubo_Matrix(alpha, inputData)
    print(b)
    '''a_file = open("matrix_Q.txt", "w")
    np.savetxt(a_file, b)
    a_file.close()'''
    
#test_qubo(0.977, german_credit_data())
    
'''def test_qubo_function():
    #Funzione di test per creare una funzione e trovare il minimo
    print("Computing")
    x = subset_Vector(48, 35)
    qubo = qubo_Matrix(0.3, german_credit_data())    
    fqx = -x.T*qubo*x
    y = np.argmin(fqx)
    print("Argmin di f(x) = x.T*Q*x: ", y)
    a_file = open("fqx.txt", "w")
    np.savetxt(a_file, fqx)
    a_file.close()
    print("DONE")

        
for i in range (10):
    test_qubo_function()'''
