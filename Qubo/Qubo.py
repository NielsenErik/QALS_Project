#!/usr/local/bin/python3
from sqlalchemy import false
from Data_Rescaler import rescaledDataframe
from Data_Rescaler import vector_V
from Data_Rescaler import german_credit_data
import numpy as np
import pandas as pd

def column_Correlation(inputData, v_Vector):
    '''This function is made to find correlation between the columns, and each column
        with quality vector V describe in the paper.
        the paper say tha for simplicity in convinient to use peason correlation
        Pearson correlation '''
    rows, columns = inputData.shape
    corrColumnsMatrix = np.zeros((columns,columns))
    corrColumnsV = np.zeros(columns)
    v = np.asarray(v_Vector.astype(float))    
    for i in range(columns):
        x = np.asarray(inputData[:,i].astype(float))
        tmpVect= np.corrcoef(x,v)
        corrColumnsV[i] = tmpVect[0,1]     
        for j in range(columns):            
            y = np.asarray(inputData[:,j].astype(float))
            tmpMatrix = np.corrcoef(x, y)               
            corrColumnsMatrix[i,j] = tmpMatrix[1,0]
    print(corrColumnsMatrix, " ", corrColumnsMatrix.shape)
    print(corrColumnsV)
    return corrColumnsV, corrColumnsMatrix
'''Done ro column i column j and ro column j e vector v'''