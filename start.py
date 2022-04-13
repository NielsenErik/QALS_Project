#!/usr/local/bin/python3

from dwave.system.samplers import DWaveSampler
from dwave.system.samplers.dwave_sampler import DWaveSampler
import dwave_networkx as dnx
import networkx as nx
import neal
import time
from Qubo.german_credit_data import german_credit_data
from Qubo.preprocessing_data import rescaledDataframe, vector_V


from Qubo.solverQubo import QUBOsolver
from Qubo.solverRFECV import RFECV_solver
from Qubo.getAccuracyScore import getAccuracy
from Qubo.graphs_for_dwave import generate_pegasus, get_Nodes, get_Q


def main():
    print("This program aim is optimal feature selection") 
    print("in credit scoring using quantum annealer")
    data = german_credit_data()
    inputMatrix = rescaledDataframe(data)
    inputVector = vector_V(data)
    alpha = 0.977
    
    qubo_array= QUBOsolver(48, alpha, inputMatrix, inputVector, 1 , 10 ,simulation = False)
    rfecv_array = RFECV_solver(inputMatrix, inputVector)
    scoreQubo, feature_nQ = getAccuracy(qubo_array, inputMatrix, inputVector, isQubo= True)
    scoreRfecv, feature_nR = getAccuracy(rfecv_array, inputMatrix, inputVector, isQubo= False)

    print(" QUBO = ", scoreQubo, " Feature number = ", feature_nQ, " RFECV = ", scoreRfecv, " Feature number = ", feature_nR)
    
    
    
    print("Done")

if __name__=='__main__':
    main()