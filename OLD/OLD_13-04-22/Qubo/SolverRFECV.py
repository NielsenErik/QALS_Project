
import numpy as np
from Qubo_Matrix import qubo_Matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from Data_Rescaler import german_credit_data, rescaledDataframe, vector_V


def RFECV_solver():
    x = rescaledDataframe(german_credit_data())
    y = vector_V(german_credit_data())
    logReg = LogisticRegression()
    selector = RFECV(logReg, step = 1, cv = 3)
    selector = selector.fit(x, y)
    indexList = selector.get_support()
    featureList = np.where(indexList)[0]
    result = np.asarray(featureList)
    return result