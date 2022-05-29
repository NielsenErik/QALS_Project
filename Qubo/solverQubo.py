#!/usr/local/bin/python3

import neal
import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import dimod
import numpy as np
import pandas as pd

from .utils import print_step
from .graphs_for_dwave import annealer, get_Q, hybrid_solver
from .qubo_matrix import qubo_Matrix

def QUBOsolver(n, alpha, inputMatrix, inputVector, k = 1, simulation = True):
    #n = dimension of problem, Q = qubo numpy Matrix, k = number of reads
    #in the annealer, simulation = simulate or run dwave
    #alpha = weighting needed in the QUBO formulation
    #inputMatrix = matrix from rescaledDataframe()
    #inputVector = vector from vector_V()
    #This function is the main solver, has both simulation and real usage.
    #It is set normally on simulation to avoid unwanted
    #time consuption
    print_step("Genearating Qubo Matrix", "QUBO")
    Q_matrix = qubo_Matrix(alpha, inputMatrix, inputVector)
    qubo = get_Q(Q_matrix, simulation)
    if(simulation == False):
        print_step("Running Dwave", "QUBO")
        #sampler = EmbeddingComposite(DWaveSampler({'topology__type':'pegasus'}))
        sampler = LeapHybridSampler()
        print_step("Running annealer", "QUBO")
        x = hybrid_solver(qubo, sampler)
        #x = annealer(qubo, sampler, k)
        
    else:
        print_step("Running simulation", "QUBO")
        sampler = neal.SimulatedAnnealingSampler()
        print_step("Running annealer", "QUBO")
        x = annealer(qubo, sampler, k)
        
    print_step("Done with annealer", "QUBO")
    x = np.asarray(x)
    numerical_x = np.asarray(np.where(x>0)).flatten()
    return numerical_x
