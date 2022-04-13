#!/usr/local/bin/python3
from tkinter.font import names
from warnings import catch_warnings
from Subset_generator import subset_Vector
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing

def testData():
    columnNames = ['1','2','3','4','5']
    data = pd.read_csv("Data_folder/Test_data/test.csv", names= columnNames)
    return data

def binarizingWGetDummies (inputData):
    '''Column in need to be bynarized: 1,2,'''
    '''The first binary indicator in each group was removed (for k indicators, only k − 1 are independent).
        So must drop first column of each group converted on one-hot in binarizing'''
    catData = inputData[['1']]
    outputData = pd.get_dummies(catData, drop_first=True)
    return outputData       
        

def normalizing (inputData):
    '''Column in need to be normalize: 4,5'''
    numData = inputData[['2','3']]
    scaler = preprocessing.StandardScaler().fit(numData)
    scaler.mean_
    scaler.scale_
    tmp = scaler.transform(numData)
    outputData = pd.DataFrame(tmp)
    return outputData

def classifizing (inputData):
    '''Column as classifier: 5'''
    
    inputData.loc[inputData["4"]=='A', "4"]=0
    inputData.loc[inputData["4"]=='B', "4"]=1
    outputData = inputData[['4']]

    return outputData

def vector_V (inputData):
    inputData.loc[inputData["5"]==1, "5"]=0
    inputData.loc[inputData["5"]==2, "5"]=1
    v = inputData['5'].to_numpy()      
    return v

def rescaledDataframe (inputData):
    catData = binarizingWGetDummies(inputData)
    numData = normalizing(inputData)
    classData = classifizing(inputData)
    tmp = catData.join(numData).join(classData)
    outputData = tmp.to_numpy()
    return outputData

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
    return corrColumnsV, corrColumnsMatrix

def qubo_Matrix (alpha, inputData):
    
    input_Matrix = rescaledDataframe(inputData)
    v_array = vector_V(inputData)
    rho_vector_V, rho_column = column_Correlation(input_Matrix, v_array)
    dim = len(rho_vector_V)
    qubo = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if(j != i):
                qubo[i,j] = (1-alpha)*rho_column[i,j]
            else:
                qubo[i,j] = alpha*rho_vector_V[i]    
    return qubo

def test_qubo(alpha, inputData):
    b = qubo_Matrix(alpha, inputData)
    print(b)
    a_file = open("matrix.txt", "w")
    np.savetxt(a_file, b)
    a_file.close()
    
def qubo_function():
    x = subset_Vector(5, 2)
    qubo = qubo_Matrix(0.3, testData())
    
    fqx = -x.T*qubo*x
    y = np.argmin(fqx)
    print(y)
    a_file = open("matrix.txt", "w")
    np.savetxt(a_file, fqx)
    a_file.close()
    print(fqx)


qubo_function()