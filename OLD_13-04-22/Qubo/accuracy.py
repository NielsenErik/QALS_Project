import numpy as np 
import pandas as pd

from Qubo_Matrix import qubo_Matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from Subset_generator import subset_Vector
from Data_Rescaler import german_credit_data, rescaledDataframe, vector_V

def getResultRFECV(RFE_array):
    x_tmp = rescaledDataframe(german_credit_data())
    rows, _ = x_tmp.shape
    pos = np.asarray(RFE_array)
    columns = len(RFE_array)
    tmp_x = x_tmp[:,pos]
    x = tmp_x.reshape(rows, columns)
    y = vector_V(german_credit_data())
    sss = StratifiedShuffleSplit(n_splits=10000, test_size=0.5, random_state=0)
    sss.get_n_splits(x, y)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    logReg = LogisticRegression(random_state=0).fit(x_train, y_train)
    #print(logReg.predict(x_test))
    print(logReg.score(x_test, y_test))
    score = logReg.score(x_test, y_test)
    return score