#!/usr/local/bin/python3

from scipy import stats
import numpy as np

def column_Correlation(inputData, v_Vector):
    #inputData = data_Rescaled(german_credit_data())
    #v_Vector = vectro_V(german_credit_data())
    #Anyway this functions works with any input matrix and 
    #relatice classifing vector
    
    rows, columns = inputData.shape
    
    '''Initialize correlation column matrix and correlation vector'''
    
    corrColumnsMatrix = np.zeros((columns,columns))
    corrColumnsV = np.zeros(columns)
    
    '''Calculate correlation between each column and also between quality vector'''
    
    v = np.asarray(v_Vector.astype(float))    
    for i in range(columns):
        x = np.asarray(inputData[:,i].astype(float))
        tmpVect, tmp= stats.spearmanr(x,v)
        corrColumnsV[i] = tmpVect
        for j in range(columns):            
            y = np.asarray(inputData[:,j].astype(float))
            tmpMatrix , tmp = stats.spearmanr(x,y)  
            corrColumnsMatrix[i,j] = tmpMatrix
            corrColumnsMatrix[j,i] = tmpMatrix
    #corrColumnsV = np.absolute(corrColumnsV)
    #corrColumnsMatrix = np.absolute(corrColumnsMatrix)      
    
    return corrColumnsV, corrColumnsMatrix

def qubo_Matrix (alpha, inputMatrix, inputVector):
    #alpha = weighting needed in the QUBO formulation
    #inputMatrix = matrix from rescaledDataframe()
    #inputVector = vector from vector_V()
    ''' this function is to generate Q matrix for the qubo problem'''
    
    '''Data preprocessing and correlation matrix and vector'''
 
    rho_vector_V, rho_column = column_Correlation(inputMatrix, inputVector)
    rho_vector_V = np.absolute(rho_vector_V)
    rho_column = np.absolute(rho_column)
    '''Qubo initiaization and creation'''
    dim = len(rho_vector_V)
    qubo = np.zeros((dim,dim))
    for i in range(dim):
        qubo[i,i] = -alpha*rho_vector_V[i]
        for j in range(dim):
            if(j != i):
                qubo[i,j] = (1-alpha)*rho_column[i,j]
    return qubo
