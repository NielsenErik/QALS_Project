#!/usr/local/bin/python3


import numpy as np
import dwave_networkx as dnx
import networkx as nx
from dwave.system.samplers import DWaveSampler
from .utils import print_step


def annealer(theta, sampler, k, time=False):
    #qubo = get_Q, sampler = sampler dWave, k = number of reads in sample_qubo
    #this function is activating the annealer and getting the results, 
    #works both with D-Wave and with simulation
    if time:
        start = time.time()        
    response = sampler.sample_qubo(theta, num_reads=k)    
    if time:
        print_step(f"Time: {time.time()-start}")    
    return list(response.first.sample.values())

def hybrid_solver(Q, sampler):
    
    response = sampler.sample_qubo(Q) 
     
    return list(response.first.sample.values())

def get_Q(q, simulation = True):
    #q = qubo_matrix in numpy, 
    #if Simulation = False: A  = get_nodes(...)
    #if Simulation = True: A  = generate_pegasus(...)
    #this is made to map matrix Q basing on Dwave topology A
    #has possibility to choose between simulation or real usage
    
    n = len(q)
    Q = dict()
    if(simulation == False):
        print_step("Mapping QUBO matrix on Dwave's qubit", "QUBO")
    else:
        print_step("Mapping QUBO matrix on Simulation", "QUBO")
    for i in range(n):
        Q[i,i] = q[i][i]
        for j in range(n):
            Q[i,j] = q[i][j]
    return Q


def generate_chimera(n):
    print_step("Generating chimera graph", "QALS")
    G = dnx.chimera_graph(16)
    tmp = nx.to_dict_of_lists(G)
    rows = []
    cols = []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        for j in tmp[i]:
            if(j < n):
                rows.append(i)
                cols.append(j)

    return list(zip(rows, cols))

def generate_pegasus(n):
    print_step("Generating pegasus graph", "QALS")
    G = dnx.pegasus_graph(16)

    tmp = nx.to_numpy_matrix(G)
    
    rows = []
    cols = []
           
    for i in range(n):
        rows.append(i)
        cols.append(i)
        for j in range(n):
            if(tmp.item(i,j)):
                rows.append(i)
                cols.append(j)
      
    return list(zip(rows, cols))

def get_Nodes(sampler, n):
    #sampler = Dwave_Sampler, n = number of nodes
    #This function get the nodes from qubits/couplers 
    #needed in case of D-Wave usage
    print_step("Getting Qubits and Couplers from Dwave for QALS")
    nodes = dict()
    tmp = list(sampler.nodelist)
    nodelist = list()
    for i in range(n):
        try:
            nodelist.append(tmp[i])
        except IndexError:
            input(f"Error when reaching {i}-th element of tmp {len(tmp)}") 

    for i in nodelist:
        nodes[i] = list()

    for node_1, node_2 in sampler.edgelist:
        if node_1 in nodelist and node_2 in nodelist:
            nodes[node_1].append(node_2)
            nodes[node_2].append(node_1)

    if(len(nodes) != n):
        i = 1
        while(len(nodes) != n):
            nodes[tmp[n+i]] = list()

    return nodes