#!/usr/local/bin/python3

import neal
import dwave_networkx as dnx
import networkx as nx
from dwave.system.samplers import DWaveSampler
import dimod
import numpy as np
import pandas as pd
from .graphs_for_dwave import annealer, generate_pegasus, get_Nodes, get_Q
from .qubo_matrix import qubo_Matrix


def qubo_function (x, Q):
    f = np.matmul(np.matmul(-x, Q), x.T)
    return f

def QUBOsolver(n, alpha, inputMatrix, inputVector, number_iteration, k = 1, simulation = True):
    #n = dimension of problem, Q = qubo numpy Matrix, k = number of reads
    #in the annealer, simulation = simulate or run dwave
    #alpha = weighting needed in the QUBO formulation
    #inputMatrix = matrix from rescaledDataframe()
    #inputVector = vector from vector_V()
    #This function is the main solver, has both simulation and real usage.
    #It is set normally on simulation to avoid unwanted
    #time consuption
    print("Genearating Qubo Matrix")
    Q_matrix = qubo_Matrix(alpha, inputMatrix, inputVector)
    if(simulation == False):
        print("Running Dwave")
        sampler = DWaveSampler({'topology__type':'pegasus'})
        A = get_Nodes(sampler, n)
    else:
        print("Running simulation")
        sampler = neal.SimulatedAnnealingSampler()
        A = generate_pegasus(n)

    qubo = get_Q(Q_matrix, A, simulation)

    f = np.zeros(number_iteration)
    x = np.zeros((number_iteration, len(Q_matrix)))
    print("Running annealer")
    for i in range(number_iteration):
        x[i] = annealer(qubo, sampler, k)
        print("Done with annealer measure: ", i+1)
        x[i] = np.asarray(x[i])
        f[i] = qubo_function(x[i], Q_matrix)
    res = np.argmin(f)
    numerical_x = np.asarray(np.where(x[res]>0))
    return numerical_x
