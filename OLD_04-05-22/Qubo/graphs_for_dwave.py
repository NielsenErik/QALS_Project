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

def generate_pegasus(n):
    #n = number of nodes in pegasus graph
    #This function generate pegasus graph needed
    #in case of simulation
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
    #This function get the nodes from qubits/couplers 
    #needed in case of D-Wave usage
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

def get_Q(q, A, simulation = True):
    #q = qubo_matrix in numpy, 
    #if Simulation = False: A  = get_nodes(...)
    #if Simulation = True: A  = generate_pegasus(...)
    #this is made to map matrix Q basing on Dwave topology A
    #has possibility to choose between simulation or real usage
    
    n = len(q)
    Q = dict()
    if(simulation == False):
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