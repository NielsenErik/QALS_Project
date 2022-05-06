#!/usr/local/bin/python3

import neal
import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import numpy as np
import pandas as pd
from .graphs_for_dwave import annealer, get_Q
from .qubo_matrix import qubo_Matrix


def qubo_function (x, Q):
    f = np.matmul(np.matmul(-x, Q), x.T)
    return f

def QUBOsolver(n, alpha, inputMatrix, inputVector, k = 1, simulation = True):
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
        sampler = EmbeddingComposite(DWaveSampler({'topology__type':'pegasus'}))
    else:
        print("Running simulation")
        sampler = neal.SimulatedAnnealingSampler()
    qubo = get_Q(Q_matrix, simulation)
    
    print("Running annealer")
    x = annealer(qubo, sampler, k)
    print("Done with annealer")
    x = np.asarray(x)
    #f= qubo_function(x, Q_matrix)
    numerical_x = np.asarray(np.where(x>0))
    return numerical_x
