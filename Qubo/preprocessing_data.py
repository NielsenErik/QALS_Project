#!/usr/local/bin/python3

import pandas as pd
from sklearn import preprocessing
'''This preprocessing methods works only with german credit
data, the steps used are explained in the paper'''
def binarizingWGetDummies (inputData):
    #inputData = german_credi_data()
    '''Column in need to be bynarized: 1,3,4,6,7,9,10,12,14,15,17'''
    '''The first binary indicator in each group was removed (for k indicators, only k − 1 are independent).
        So must drop first column of each group converted on one-hot in binarizing'''
    catData = inputData[['1','3','4','6','7','9','10','12','14','15','17']]
    outputData = pd.get_dummies(catData, drop_first=True)
    return outputData       
        

def normalizing (inputData):
    #inputData = german_credi_data()
    
    '''Column in need to be normalize: 2,5,8,11,13,16,18'''
    numData = inputData[['2','5','8','11','13','16','18']]
    scaler = preprocessing.StandardScaler().fit(numData)
    scaler.mean_
    scaler.scale_
    tmp = scaler.transform(numData)
    outputData = pd.DataFrame(tmp)
    return outputData

def classifizing (inputData):
    #inputData = german_credi_data()
    '''Column as classifier: 19,20
        Attribute 19: (qualitative)
	      Telephone
	      A191 : none
	      A192 : yes, registered under the customers name

        Attribute 20: (qualitative)
	      foreign worker
	      A201 : yes
	      A202 : no'''
    
    inputData.loc[inputData["19"]=="A191", "19"]=0
    inputData.loc[inputData["19"]=="A192", "19"]=1
    inputData.loc[inputData["20"]=="A202", "20"]=0
    inputData.loc[inputData["20"]=="A201", "20"]=1
    outputData = inputData[['19','20']]
    return outputData

def vector_V (inputData):
    #inputData = german_credi_data()
    #v is the classifying vector of good/bad credit
    #it is the name used in the paper for qubo 
    #formulation
    
    print("Creating classifying vector of good/bad credit")
    
    inputData.loc[inputData["21"]==1, "21"]=0
    inputData.loc[inputData["21"]==2, "21"]=1
    v = inputData['21'].to_numpy()      
    return v

def rescaledDataframe (inputData):
    
    #inputData = german_credi_data()
    #output is
    
    print("Preprocessing Data")
    
    catData = binarizingWGetDummies(inputData)
    numData = normalizing(inputData)
    classData = classifizing(inputData)
    tmp = catData.join(numData).join(classData)
    outputData = tmp.to_numpy()
    return outputData
