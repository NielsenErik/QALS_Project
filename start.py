#!/usr/local/bin/python3
from sqlite3 import DatabaseError
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler
from dwave.system.samplers.dwave_sampler import DWaveSampler
import dwave_networkx as dnx
import networkx as nx
import neal
import time
from os import listdir, mkdir, system, name
from os.path import isfile, join, exists
import sys
from Qubo import File_withAll

def main():
    print("This program aim optimal feature selection") 
    print("in credit scoring using quantum annealer")
    data = File_withAll.german_credit_data()
    pos, qubo_result, f_value = File_withAll.qubo_solver_per_K(1000000, 48, 24, 0.977, data)
    File_withAll.getResult(qubo_result)
    #result = SolverQubo.qubo_solver_per_K(1000, 48, 24, 0.977, data)
    print("Done")

if __name__=='__main__':
    main()