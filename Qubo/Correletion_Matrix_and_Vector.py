#!/usr/local/bin/python3
from sqlalchemy import false
from scipy import stats
from Data_Rescaler import rescaledDataframe, vector_V, german_credit_data
import numpy as np

def column_Correlation(inputData, v_Vector):
    '''This function is made to find correlation between the columns, and each column
        with quality vector V describe in the paper.
        the paper say tha for simplicity in convinient to use peason correlation
        Pearson correlation '''
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
    print(corrColumnsMatrix, " ", corrColumnsMatrix.shape)
    print(corrColumnsV)
    return corrColumnsV, corrColumnsMatrix
'''Done ro column i column j and ro column j e vector v'''

'''data = german_credit_data()
dataMatrix = rescaledDataframe(data)
v = vector_V(data)
column_Correlation(dataMatrix, v)'''