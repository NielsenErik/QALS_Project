#!/usr/local/bin/python3
from warnings import catch_warnings
import pandas as pd
import numpy as np
from sklearn import preprocessing

'''The data consists of 20 features (7 numerical, 13 categorical) and a binary classification (good credit or bad credit).
There are 1000 rows, of which 700 are ‘‘good’’ and 300 are ‘‘bad’’. The data is intended for use with a cost matrix, where
giving credit to a bad applicant is five times as bad as not giving credit to a good applicant. In this study, however, we
were concerned mainly with the relative ‘‘predictive power’’ of the feature subsets, so the cost matrix was not used.
The data was prepared as follows:
• The german.data file from UCI was imported into a Jupyter (iPython) notebook as a pandas DataFrame and given
column headers with names from the accompanying german.doc file.
• The categorial variables were converted to ‘‘one-hot’’ binary indicators using the DictVectorizer class from
scikit-learn.
• The first binary indicator in each group was removed (for k indicators, only k − 1 are independent).
• All of the numerical features were scaled to mean zero and variance one.
• The classification variable was transformed to 0 = good, 1 = bad.
In this way it was possible to get 48 feature from the 20 already exisisting'''


'''Decided to use get dummies because by searching seems much faster and had much less trouble than use DictVectorize libraries
    as was easly able to drop first binary indicator.
    in any cases the worki'function is here below without any dropping table 
    .
    .
    def binarizing (inputData):
    #Column in need to be bynarized: 1,3,4,6,7,9,10,12,14,15,17
    v = DictVectorizer(sparse=False)
    catData = inputData[['1','3','4','6','7','9','10','12','14','15','17']].to_dict('records')
    tmp = v.fit_transform(catData)
    output = pd.DataFrame.from_dict(tmp)
    output.columns= v.get_feature_names_out(catData)
    return output
    .
    .
    '''
def german_credit_data ():
    try:
        #first transformed german.data into german.csv
        column_Names = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"]
        #column_Names = ['Status of existing checking account','Duration in month','Credit history','Purpose','Credit amount','Savings account/bonds','Present employment since','Installment rate in percentage of disposable income','Personal status and sex','Other debtors / guarantors','Present residence since','Property','Age in years','Other installment plans','Housing','Number of existing credits at this bank','Job','Number of people being liable to provide maintenance for','Telephone','foreign worker']
        dataframe = pd.read_csv("Data_folder/German/german.csv", names=column_Names)
    except:
        print("Import dataframe error")
    return dataframe
    
def binarizingWGetDummies (inputData):
    '''Column in need to be bynarized: 1,3,4,6,7,9,10,12,14,15,17'''
    '''The first binary indicator in each group was removed (for k indicators, only k − 1 are independent).
        So must drop first column of each group converted on one-hot in binarizing'''
    catData = inputData[['1','3','4','6','7','9','10','12','14','15','17']]
    outputData = pd.get_dummies(catData, drop_first=True)
    return outputData       
        

def normalizing (inputData):
    '''Column in need to be normalize: 2,5,8,11,13,16,18'''
    numData = inputData[['2','5','8','11','13','16','18']]
    scaler = preprocessing.StandardScaler().fit(numData)
    scaler.mean_
    scaler.scale_
    tmp = scaler.transform(numData)
    outputData = pd.DataFrame(tmp)
    return outputData

def classifizing (inputData):
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
    inputData.loc[inputData["21"]==1, "21"]=0
    inputData.loc[inputData["21"]==2, "21"]=1
    v = inputData['21'].to_numpy()      
    return v

def rescaledDataframe (input):
    catData = binarizingWGetDummies(input)
    numData = normalizing(input)
    classData = classifizing(input)
    tmp = catData.join(numData).join(classData)
    outputData = tmp.to_numpy()
    return outputData

