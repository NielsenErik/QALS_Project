#!/usr/local/bin/python3

import numpy as np
import random
import neal

from dwave.system import DWaveSampler, EmbeddingComposite
from .graphs_for_dwave import get_Q, annealer, generate_pegasus, get_Nodes, get_Theta
from .qubo_matrix import qubo_Matrix
from .graphs_for_dwave import get_Nodes, generate_pegasus
#These function are from Andrea Bonomi project in QALS, all credits to him and check his work 
#at his github repo: https://github.com/bonom/Quantum-Annealing-for-solving-QUBO-Problems

def qubo_function (x, Q):
    f = np.matmul(np.matmul(x, Q), x.T)
    return f

def make_decision(probability):
    return random.random() < probability

def random_shuffle(a):
    keys = list(a.keys())
    values = list(a.values())
    random.shuffle(values)
    return dict(zip(keys, values))


def shuffle_vector(v):
    n = len(v)
    
    for i in range(n-1, 0, -1):
        j = random.randint(0,i) 
        v[i], v[j] = v[j], v[i]

def shuffle_map(m):
    
    keys = list(m.keys())
    shuffle_vector(keys)
    
    i = 0

    for key, item in m.items():
        it = keys[i]
        ts = item
        m[key] = m[it]
        m[it] = ts
        i += 1

def fill(m, perm, _n):
    n = len(perm)
    if (n != _n):
        n = _n
    filled = np.zeros(n, dtype=int)
    for i in range(n):
        if i in m.keys():
            filled[i] = perm[m[i]]
        else:
            filled[i] = perm[i]

    return filled


def inverse(perm, _n):
    n = len(perm)
    if(n != _n):
        n = _n
    inverted = np.zeros(n, dtype=int)
    for i in range(n):
        inverted[perm[i]] = i

    return inverted


def map_back(z, perm):
    n = len(z)
    inverted = inverse(perm, n)

    z_ret = np.zeros(n, dtype=int)

    for i in range(n):
        z_ret[i] = int(z[inverted[i]])

    return z_ret

def g(Q, A, oldperm, p, sim):
    #Q = qubo matrix
    #A = graph topology
    #oldperm = oldpermutation
    #p = probability
    #sim = simulation or not
    
    n = len(Q)
    m = dict()
    for i in range(n):
        if make_decision(p):
            m[i] = i
    

    m = random_shuffle(m)
    perm = fill(m, oldperm, n)
    inversed = inverse(perm, n)
    
    Theta = dict()
    if (sim):
        for row, col in A:
            k = inversed[row]
            l = inversed[col]
            Theta[row, col] = Q[k][l]
    else:
        support = dict(zip(A.keys(), np.arange(n))) 
        for key in list(A.keys()):
            k = inversed[support[key]]
            Theta[key, key] = Q[k][k]
            for elem in A[key]:
                l = inversed[support[elem]]
                Theta[key, elem] = Q[k][l]
              
    return Theta, perm

def qals_Solver(qubo, dim, n_reads, lambda_zero, p_delta, N, q, d_min, i_max, N_max, simulation = True):
    #qubo = qubo matrix
    #dim = dimension of problem
    #n_reads = number reads
    #lambda_zero = lambda_zero
    #p_delta = delta probability, minimum probability of permutation modification
    #N = probability decreasing rate
    #q = candidate perturbation probability 
    
    if(simulation == False):
        sampler = DWaveSampler({'topology__type':'pegasus'})
        A = get_Nodes(sampler, dim)
    else:
        sampler = neal.SimulatedAnnealingSampler()
        A = generate_pegasus(dim)
        
    I = np.identity(dim)
    p = 1
    Theta_one, m_one = g(qubo, A, np.arange(dim), p, simulation)
    Theta_two, m_two = g(qubo, A, np.arange(dim), p, simulation)
    
    z_one = map_back(annealer(Theta_one, sampler, n_reads), m_one)
    z_two = map_back(annealer(Theta_two, sampler, n_reads), m_two)
    
    f_one = qubo_function(qubo, z_one).item()
    f_two = qubo_function(qubo, z_two).item()
    
    if (f_one < f_two):
        z_star = z_one
        f_star = f_one
        m_star = m_one
        z_prime = z_two
    else:
        z_star = z_two
        f_star = f_two
        m_star = m_two
        z_prime = z_one
        
    if (f_one != f_two):
        S = (np.outer(z_prime, z_prime) - I) + np.diagflat(z_prime)
    else:
        S = np.zeros((dim, dim))
    
    e = 0
    d = 0
    i = 1
    lam = lambda_zero   
    
    