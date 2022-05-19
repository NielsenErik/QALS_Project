#!/usr/local/bin/python3

from re import X
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

def generateSintheticDataset(nSamples, nFeatures):
    X, y = make_classification(
        n_samples=nSamples,
        n_features=nFeatures,
        n_informative=nFeatures-5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    
    tmp = np.insert(X, nFeatures, y, axis = 1)
    dataset = pd.DataFrame(tmp)
    data_name = "dataset_nf_"+str(nFeatures)+".csv"
    return dataset, data_name

for i in range(100):
    k = 10+i*10
    dataset, dataset_name = generateSintheticDataset(5000, k)
    path = "./Qubo/Data_folder/Synthetic_data/"+dataset_name
    dataset.to_csv(path)
