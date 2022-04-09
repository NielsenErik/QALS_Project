#!/usr/local/bin/python3

import dimod
import numpy as np 
import pandas as pd
import dimod
from SolverQubo import getResultForQubo

from Qubo_Matrix import qubo_Matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from Subset_generator import subset_Vector
from Data_Rescaler import german_credit_data, rescaledDataframe, vector_V
import dwave_networkx as dnx
import networkx as nx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

def annealer(qubo, sampler, k=1):
    #qubo = get_Q, sampler = sampler dWave, k = number of reads in sample_qubo
    response = sampler.sample_qubo(qubo, num_reads = k)
    print(response)
    return list(response.first.sample.values())

def qubo_function (x, Q):
    f = np.matmul(np.matmul(-x, Q), x.T)
    return f

def subset_array_generator_per_k (n, dim, k):
    subset_array = np.zeros((n, dim))
    for i in range(n):
        subset_array[i]=subset_Vector(dim, k)
    return subset_array

def generate_pegasus(n):
    #n = number of nodes in pegasus graph
    P = dnx.pegasus_graph(16)
    tmp = nx.to_numpy_matrix(P)
    
    graph = np.zeros((n, n))
    for i in range(n):
        graph.append((i, i))
        
        for j in range (n):
            if(tmp.item(i,j)):
                graph.append((i,j))
    
    return graph

def get_Nodes(sampler, n):
    #sampler = Dwave_Sampler, n = number of nodes
    nodes = dict()
    tmp = list(sampler.nodelist)
    nodelist = list()
    
    for i in range(n):
        nodelist.append(tmp[i])
    
    for i in nodelist:
        nodes[i] = list()
        
    for nodeA , nodeB in sampler.edgelist:
        if (nodeA in nodelist and nodeB in nodelist):
            nodes[nodeA].append(nodeB)
            nodes[nodeB].append(nodeA)
    
    if(len(nodes) != n):
        i = 1
        while(len(nodes) != n):
            nodes[tmp[n+i]] = list()
    
    return nodes

def get_Q(q, A):
    #Function to map Q basing on Dwave topology A,
    #q = qubo_matrix, A  = get_nodes(...)
    n = len(q)
    Q = dict()
    support = dict(zip(A.keys(), np.arange(n)))
    for i in list(A.keys()):
        k = support[i]
        Q[i, i] = q[k][k]
        for j in A[i]:
            l = support[j]
            Q[i,j] = q[k][l]
    
    return Q    

def getResultForQubo(qubo_array):
    x_tmp = rescaledDataframe(german_credit_data())

    rows, _ = x_tmp.shape
    tmp = np.asarray(qubo_array)
    pos = np.where(tmp>0)
    pos = np.asarray(pos)
    _, column = pos.shape
    tmp_x = x_tmp[:,pos]
    x = tmp_x.reshape(rows, column)
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

def solve(n, Q, number_iteration, k = 1):
    #n = dimension of problem
    sampler = DWaveSampler({'topology__type':'pegasus'})
    A = get_Nodes(sampler, n)   
    
    qubo = get_Q(Q, A)
    f = np.zeros(number_iteration)
    x = np.zeros((number_iteration, len(Q)))
    x = annealer(qubo, sampler, k)
    for i in range(number_iteration):
        x[i] = annealer(qubo, sampler, 1)
        x[i] = np.asarray(x[i])
        f[i] = qubo_function(x[i], Q)
        
    '''k = np.zeros((number_iteration, len(Q)))   
    for i in range(number_iteration):
        
        k[i] = np.where(x[i]>0)
        k[i] = np.asarray(k)
        print(k[i].shape)'''
    res = np.argmin(f)
    print(f[res])
    print(x[res])
    
    return x[res]

data = german_credit_data()
qubo = qubo_Matrix(0.977, data)


qubo_array=solve(48, qubo, 10, 1)
getResultForQubo(qubo_array)