#!/usr/local/bin/python3

from turtle import color
from dwave.system.samplers import DWaveSampler
from dwave.system.samplers.dwave_sampler import DWaveSampler
import dwave_networkx as dnx
import networkx as nx
import neal
import datetime
import sys
import numpy as np

from sympy import ask
from Qubo.german_credit_data import german_credit_data
from Qubo.preprocessing_data import rescaledDataframe, vector_V

from Qubo.colors import colors
from Qubo.solverQubo import QUBOsolver
from Qubo.solverRFECV import RFECV_solver
from Qubo.getAccuracyScore import getAccuracy
from Qubo.solverRandom_Max import bestRandomSubset
from Qubo.noisy_data import genearate_noisy_data

def main():
    data, data_name = german_credit_data()
    inputMatrix = rescaledDataframe(data)
    inputVector = vector_V(data)
    noise, noisy_vector = genearate_noisy_data(inputMatrix, inputVector, 100, 48)
    '''a_file = open("data.txt", "w")
    np.savetxt(a_file, inputMatrix)
    a_file.close()'''
    b_file = open("noisy_data.txt", "w")
    np.savetxt(b_file, noisy_vector)
    b_file.close()
    
    
if __name__=='__main__':
    main()