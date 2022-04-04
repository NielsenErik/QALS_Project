#!/usr/local/bin/python3
import numpy as np
import random

def subset_Generator (dim, k):
    ''' this F generates a vector of size k (number of feature of subset) 
        and dim(max number of feature'''
    try: 
        if(k<0 and k>dim):
            raise ValueError
        else:
            tmp = random.sample(range(dim), k)
            x = np.array(tmp)
    except ValueError:
        print("Dim and K size error")
    return x    

def subset_Vector (dim, k):
    tmp = subset_Generator(dim, k)
    x = np.zeros(dim)
    for i in range(dim):
        for j in range(k):
            if( i == tmp[j]):
                x[i]=int(1)
    return x

