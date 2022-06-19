#!/usr/local/bin/python3

from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from Qubo.noisy_data import generate_noisy_data, generate_noisy_feature
from Qubo.import_data import german_credit_data
from Qubo.preprocessing_data import rescaledDataframe_German, vector_V_German
def generateSintheticDataset(nSamples, nFeatures):
    X, y = make_classification(
        n_samples=nSamples,
        n_features=nFeatures,
        n_informative=(nFeatures-int(0.6*nFeatures)),
        n_redundant=int(.1*nFeatures),
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    tmp = np.insert(X, nFeatures, y, axis = 1)
    dataset = pd.DataFrame(tmp)
    data_name = "dataset_nf_"+str(nFeatures)+".csv"
    return dataset, data_name

def generate_noisy_datsets_German(start, end):
    data, data_name = german_credit_data()
    inputMatrix, matrix_Len = rescaledDataframe_German(data)
    inputVector = vector_V_German(data)
    for i in range(start,end):
        generate_noisy_data(inputMatrix, inputVector, (i+1)*0.01, matrix_Len, data_name)
        generate_noisy_feature(inputMatrix, inputVector, i+1, matrix_Len, data_name)
    
def n_datsates():
    nf = [75,100,125,150,175,200,250,300,500,750,1000]
    for i in nf:
        
        dataset, dataset_name = generateSintheticDataset(1500, i)
        path = "./Qubo/Data_folder/Synthetic_data/"+dataset_name
        dataset.to_csv(path, index=False) 

#generate_noisy_datsets_German(9,10) 
n_datsates()


