#!/usr/local/bin/python3

import dimod
import numpy as np
import pandas as pd
from sqlalchemy import false, true
from SolverQubo import getResultForQubo

from Qubo_Matrix import qubo_Matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from Subset_generator import subset_Vector
from Data_Rescaler import german_credit_data, rescaledDataframe, vector_V
import neal
import dwave_networkx as dnx
import networkx as nx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

def annealer(qubo, sampler, k=1):
    #qubo = get_Q, sampler = sampler dWave, k = number of reads in sample_qubo
    response = sampler.sample_qubo(qubo, num_reads = k)
    csv_report = response.to_pandas_dataframe()
    csv_report.to_csv("annealer.csv")
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
    print("Generating Pegasus Graph")
    P = dnx.pegasus_graph(16)
    tmp = nx.to_numpy_matrix(P)

    rows = []
    columns = []

    for i in range(n):
        rows.append(i)
        columns.append(i)
        for j in range(n):
            if(tmp.item(i,j)):
                rows.append(i)
                columns.append(j)

    return list(zip(rows, columns))

def get_Nodes(sampler, n):
    #sampler = Dwave_Sampler, n = number of nodes
    print("Getting Qubits and Couplers from Dwave")
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

def get_Q(q, A, simulation = true):
    #Function to map Q basing on Dwave topology A,
    #q = qubo_matrix in numpy, A  = get_nodes(...)
    n = len(q)
    Q = dict()
    if(simulation == false):
        print("Mapping QUBO on Dwave's qubit")
        support = dict(zip(A.keys(), np.arange(n)))
        for i in list(A.keys()):
            k = support[i]
            Q[i, i] = q[k][k]
            for j in A[i]:
                l = support[j]
                Q[i,j] = q[k][l]
    else:
        print("Mapping QUBO on Simulation")
        for rows, columns in A:
            Q[rows, columns] = q[rows][columns]

    return Q

def getResultForQubo(qubo_array):
    print("Getting accuracy score from previuos results")
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

def solve(n, Q, number_iteration, k = 1, simulation = true):
    #n = dimension of problem, Q = qubo numpy Matrix, k = number of reads
    #in the annealer, simulation = simulate or run dwave

    if(simulation == false):
        print("Running Dwave")
        sampler = DWaveSampler({'topology__type':'pegasus'})
        A = get_Nodes(sampler, n)
    else:
        print("Running simulation")
        sampler = neal.SimulatedAnnealingSampler()
        A = generate_pegasus(n)

    qubo = get_Q(Q, A, simulation)

    f = np.zeros(number_iteration)
    x = np.zeros((number_iteration, len(Q)))
    for i in range(number_iteration):
        x[i] = annealer(qubo, sampler, k)
        x[i] = np.asarray(x[i])
        f[i] = qubo_function(x[i], Q)
    res = np.argmin(f)
    print(f[res])
    print(x[res])

    return x[res]

data = german_credit_data()
qubo = qubo_Matrix(0.977, data)


qubo_array=solve(48, qubo, 1, 1000, simulation=true)
getResultForQubo(qubo_array)
