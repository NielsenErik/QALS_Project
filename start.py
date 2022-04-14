#!/usr/local/bin/python3

from turtle import color
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

from Qubo.colors import colors
from Qubo.solverQubo import QUBOsolver
from Qubo.solverRFECV import RFECV_solver
from Qubo.getAccuracyScore import getAccuracy

def header_script():
    #just the title and headar that appear on terminal
    print(colors.BOLD, colors.HEADER, "This program aim is optimal feature selection") 
    print("in credit scoring using quantum annealer", colors.ENDC)

def ask_for_simulation():
    #This function ask if it is wanted a simulation
    #or real usage touser before send the problem
    print(" ")
    print(colors.ORANGE ,"Would you like to try a simulation or run Dwave?", colors.ENDC)    
    sim = True
    check = False
    while(check == False):
        print(colors.ORANGE)
        simulation = input("[s] for simulation [d] for dwave [e] for exit program: ")
        print(colors.ENDC)
        if(simulation == 's'):
            sim = True
            check = True
        elif(simulation == 'd'):
            sim = False
            check = True
        elif(simulation == 'e'):
            print(colors.WARNING, "Terminating program...", colors.ENDC)
            exit()
        else:
            print(colors.FAIL, "Wrong answer, try again!", colors.ENDC)
    return sim

def main():
    #main function of the program
    
    header_script()
    
    sim = ask_for_simulation()
    data = german_credit_data()
    inputMatrix = rescaledDataframe(data)
    inputVector = vector_V(data)
    alpha = 0.977
    
    qubo_array= QUBOsolver(48, alpha, inputMatrix, inputVector, 1 , 100 ,simulation = sim)
    rfecv_array = RFECV_solver(inputMatrix, inputVector)
    scoreQubo, feature_nQ = getAccuracy(qubo_array, inputMatrix, inputVector, isQubo= True)
    scoreRfecv, feature_nR = getAccuracy(rfecv_array, inputMatrix, inputVector, isQubo= False)
    
    print(colors.BOLD, colors.HEADER, "RESULTS", colors.ENDC)
    print(colors.RESULT, " QUBO = ", scoreQubo, " Feature number = ", feature_nQ, " RFECV = ", scoreRfecv, " Feature number = ", feature_nR, colors.ENDC)   
    print(colors.BOLD, colors.HEADER, "Done", colors.ENDC)

if __name__=='__main__':
    main()