#!/usr/local/bin/python3
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler
from dwave.system.samplers.dwave_sampler import DWaveSampler
import dwave_networkx as dnx
import networkx as nx
import neal
import time

def main():
    print("This program aim optimal feature selection") 
    print("in credit scoring using quantum annealer")
    time.sleep(0.5)
    print("Done")

if __name__=='__main__':
    main()