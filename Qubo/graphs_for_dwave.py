#!/usr/local/bin/python3


import numpy as np
import dwave_networkx as dnx
import networkx as nx
from dwave.system.samplers import DWaveSampler

def annealer(qubo, sampler, k=1):
    #qubo = get_Q, sampler = sampler dWave, k = number of reads in sample_qubo
    #this function is activating the annealer and getting the results, 
    #works both with D-Wave and with simulation
    response = sampler.sample_qubo(qubo, num_reads = k)
    '''csv_report = response.to_pandas_dataframe()
    csv_report.to_csv("annealer.csv")'''
    return list(response.first.sample.values())

def get_Q(q):
    #q = qubo_matrix in numpy, 
    #if Simulation = False: A  = get_nodes(...)
    #if Simulation = True: A  = generate_pegasus(...)
    #this is made to map matrix Q basing on Dwave topology A
    #has possibility to choose between simulation or real usage
    
    n = len(q)
    Q = dict()
    print("Mapping QUBO on Dwave's qubit")
    for i in range(n):
        Q[i,i] = q[i][i]
        for j in range(n):
            Q[i,j] = q[i][j]
    return Q