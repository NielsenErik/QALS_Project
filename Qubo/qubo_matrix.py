#!/usr/local/bin/python3

from scipy import stats
import numpy as np

def column_Correlation(inputData, inputArray):
    #inputData = data_Rescaled(german_credit_data())
    #v_Vector = vectro_V(german_credit_data())
    #Anyway this functions works with any input matrix and 
    #relatice classifing vector
    
    rows, columns = inputData.shape
    
    '''Initialize correlation column matrix and correlation vector'''
    
    correlation_w_features = np.zeros((columns,columns))
    correlation_w_label = np.zeros(columns)
    
    '''Calculate correlation between each column and also between quality vector'''
    
    v = np.asarray(inputArray.astype(float))    
    for i in range(columns):
        x = np.asarray(inputData[:,i].astype(float))
        tmpVect, _= stats.spearmanr(x,v)
        correlation_w_label[i] = tmpVect
        for j in range(columns):            
            y = np.asarray(inputData[:,j].astype(float))
            tmpValue , _ = stats.spearmanr(x,y)  
            correlation_w_features[i,j] = tmpValue
            correlation_w_features[j,i] = tmpValue
    #corrColumnsV = np.absolute(corrColumnsV)
    #corrColumnsMatrix = np.absolute(corrColumnsMatrix)      
    
    return correlation_w_label, correlation_w_features

def qubo_Matrix (alpha, inputData, inputArray):
    #alpha = weighting needed in the QUBO formulation
    #inputMatrix = matrix from rescaledDataframe()
    #inputVector = vector from vector_V()
    ''' this function is to generate Q matrix for the qubo problem'''
    
    '''Data preprocessing and correlation matrix and vector'''
 
    rho_vector_V, rho_column = column_Correlation(inputData, inputArray)
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
