#!/usr/local/bin/python3

from dwave.system.samplers import DWaveSampler
from dwave.system.samplers.dwave_sampler import DWaveSampler
import dwave_networkx as dnx
import networkx as nx
import neal
import time
import sys

from sympy import ask
from Qubo.german_credit_data import german_credit_data
from Qubo.preprocessing_data import rescaledDataframe, vector_V

from Qubo.solverQubo import QUBOsolver
from Qubo.solverRFECV import RFECV_solver
from Qubo.getAccuracyScore import getAccuracy

def ask_for_simulation():
    print("Would you like to trya a simulation or run Dwave?")
    simulation = input("[y] for simulation [n] for dwave: ")
    sim = True
    if(simulation == 'y'):
        sim = True
    elif(simulation == 'n'):
        sim = False
    else:
        print("Wrong answer, terminating program...")
        exit()
    return sim

def main():
    print("This program aim is optimal feature selection") 
    print("in credit scoring using quantum annealer")
    sim = ask_for_simulation()
    data = german_credit_data()
    inputMatrix = rescaledDataframe(data)
    inputVector = vector_V(data)
    alpha = 0.977
    
    qubo_array= QUBOsolver(48, alpha, inputMatrix, inputVector, 1 , 10 ,simulation = sim)
    rfecv_array = RFECV_solver(inputMatrix, inputVector)
    scoreQubo, feature_nQ = getAccuracy(qubo_array, inputMatrix, inputVector, isQubo= True)
    scoreRfecv, feature_nR = getAccuracy(rfecv_array, inputMatrix, inputVector, isQubo= False)

    print(" QUBO = ", scoreQubo, " Feature number = ", feature_nQ, " RFECV = ", scoreRfecv, " Feature number = ", feature_nR)
    
    
    
    print("Done")

if __name__=='__main__':
    main()