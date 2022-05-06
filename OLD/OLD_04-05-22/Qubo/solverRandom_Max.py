from .getAccuracyScore import getAccuracy
from .random_Subset_Generator import subset_array_generator_per_k
import numpy as np

def bestRandomSubset(start,dim, n_arrays, inputMatrix, inputVector):
    #dim = dimension of problem
    #n_arrays =number of subset per k wanted
    print("Running Random Max, it will take a while")
    best_scores = np.zeros(dim).astype(float)
    best_sub = np.zeros((dim,dim))
    for i in range(start, dim-1):
        print("Start dim: ", i)
        x = subset_array_generator_per_k(n_arrays, dim, i)
        score = np.zeros(n_arrays).astype(float)
        for j in range(n_arrays):
            score[j], _ = getAccuracy(x[j], inputMatrix, inputVector, False, False)
        pos = np.argmax(score)
        best_scores[i] = score[pos]        
        best_sub[i] = x[pos]
        print("Done dim: ", i)
    tot_pos = np.argmax(best_scores)
    tot_best_score = best_scores[tot_pos]
    tmp = best_sub[tot_pos]
    tot_best_sub = np.asarray(np.where(tmp>0))
    tot_best_len = len(tot_best_sub[0])
    
    print(tot_best_score, " ", tot_best_len)
    
    return tot_best_score, tot_best_len, tot_best_sub
    
