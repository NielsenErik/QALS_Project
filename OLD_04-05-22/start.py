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
from Qubo.import_data import german_credit_data, australian_credit_data, polish_bankrupcy_data
from Qubo.preprocessing_data import rescaledDataframe_German, vector_V_German, vector_V_Australian, rescaledDataframe_Australian, vector_V_Polish, normalizing_Polish

from Qubo.colors import colors
from Qubo.solverQubo import QUBOsolver
from Qubo.solverRFECV import RFECV_solver
from Qubo.getAccuracyScore import getAccuracy
from Qubo.solverRandom_Max import bestRandomSubset
from Qubo.noisy_data import generate_noisy_data, generate_noisy_feature, noisy_feature_detector

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
    check1 = False
    random_max = False
    print(colors.ORANGE ,"Would you like to get score from Random Subsets generator?", colors.ENDC) 
    print(colors.ORANGE ,"It will take a lot of time", colors.ENDC) 
    while(check1 == False):
        print(colors.ORANGE)
        simulation = input("[y] for yes [n] for no: ")
        print(colors.ENDC)
        if(simulation == 'y'):
            random_max = True
            check1 = True
        elif(simulation == 'n'):
            random_max = False
            check1 = True
        else:
            print(colors.FAIL, "Wrong answer, try again!", colors.ENDC)
    return sim, random_max

def outputTxt(fileName, simulation = True):
    f = open(fileName, 'a')
    f.write("###########################################################################\n")
    now = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    f.write(now)
    f.write("\n")
    if(simulation == True):
         f.write("This result are from simulation")
    else:
        f.write("This result are from Dwave QPU")
    return f

def printStartInfos(alpha, dataName, fileDescriptor):
    #aplha = alpha parameter
    #dataName = name of data used
    #fileDescriptor = call file descriptor used in outputTxt
    #simulatioN = if is simulation or Dwave usage

    tmp = "\nData used are: " + str(dataName) + "\n"
    fileDescriptor.write(tmp)
    tmp = "Alpha value used is: " + str(alpha) + "\n\n"
    fileDescriptor.write(tmp)
    
def printResults(fileDescriptor, qubo_array, rfecv_array, score_qubo, score_rfecv, nf_qubo, nf_efecv, scoreRand, feature_nRand, randSub):
    #fileDescriptor = call file descriptor used in outputTxt
    fileDescriptor.write("RESULTS\n\n")
    tmp = "QUBO features are: " + str(qubo_array) + "\n"
    fileDescriptor.write(tmp)
    tmp = "RFECV features are: " + str(rfecv_array) + "\n\n"
    fileDescriptor.write(tmp)
    if(scoreRand != -1):
        tmp = "Random features are: " + str(randSub) + "\n\n"
        fileDescriptor.write(tmp)
    tmp = "QUBO accuracy score = " + str(score_qubo) + " with number of feature = " + str(nf_qubo) + "\n\n"
    fileDescriptor.write(tmp)  
    tmp = "RFECVaccuracy score = " + str(score_rfecv) + " with number of feature = " + str(nf_efecv) + "\n\n"
    fileDescriptor.write(tmp)       
    if(scoreRand != -1):
        tmp = "Random accuracy score = " + str(scoreRand) + " with number of feature = " + str(feature_nRand) + "\n\n"
        fileDescriptor.write(tmp)

def printResults_w_Noisy_samples(noise, fileDescriptor, qubo_array, rfecv_array, score_qubo, score_rfecv, nf_qubo, nf_efecv):
    #fileDescriptor = call file descriptor used in outputTxt
    tmp = "Results with Noisy samples'%' = " + str(noise) +"%\n\n"
    fileDescriptor.write(tmp)
    tmp = "QUBO features are: " + str(qubo_array) + "\n"
    fileDescriptor.write(tmp)
    tmp = "RFECV features are: " + str(rfecv_array) + "\n\n"
    fileDescriptor.write(tmp)
    tmp = "QUBO accuracy score = " + str(score_qubo) + " with number of feature = " + str(nf_qubo) + "\n\n"
    fileDescriptor.write(tmp)  
    tmp = "RFECVaccuracy score = " + str(score_rfecv) + " with number of feature = " + str(nf_efecv) + "\n\n"
    fileDescriptor.write(tmp)     
    
def printResults_w_Noisy_feature(noise, fileDescriptor, qubo_array, rfecv_array, score_qubo, score_rfecv, nf_qubo, nf_efecv, qubo_detector, rfecv_detector):
    #fileDescriptor = call file descriptor used in outputTxt
    tmp = "Results with number of noisy feature' = " + str(noise) +"%\n\n"
    fileDescriptor.write(tmp)
    tmp = "QUBO features are: " + str(qubo_array) + "\n"
    fileDescriptor.write(tmp)
    if(qubo_detector == True):
        tmp = "DETECTED NOISY FEATURE in QUBO\n\n"
        fileDescriptor.write(tmp)
    tmp = "RFECV features are: " + str(rfecv_array) + "\n"
    fileDescriptor.write(tmp)
    if(rfecv_detector == True):
        tmp = "DETECTED NOISY FEATURE in RFECV\n\n"
        fileDescriptor.write(tmp)
    tmp = "QUBO accuracy score = " + str(score_qubo) + " with number of feature = " + str(nf_qubo) + "\n\n"
    fileDescriptor.write(tmp)  
    tmp = "RFECVaccuracy score = " + str(score_rfecv) + " with number of feature = " + str(nf_efecv) + "\n\n"
    fileDescriptor.write(tmp)     
   

    
def main():
    #main function of the program
    
    header_script()
    
    sim, random_max = ask_for_simulation()
    fileOutput = 'outPut.txt'
    fd = outputTxt(fileOutput, sim)
   
    data, data_name = german_credit_data()
    inputMatrix, matrix_Len = rescaledDataframe_German(data)
    inputVector = vector_V_German(data)
    alpha = 0.977
    
    '''
    data, data_name = polish_bankrupcy_data()
    inputMatrix,matrix_Len = normalizing_Polish(data)
    inputVector = vector_V_Polish(data)
    alpha = 0.977
    
    '''
    
    '''
    data, data_name = australian_credit_data()
    inputMatrix, matrix_Len = rescaledDataframe_Australian(data)
    inputVector = vector_V_Australian(data)
    alpha = 0.1
    '''
    
    printStartInfos(alpha, data_name, fd)
    
    scoreRandom = -1
    feature_nRandom = -1
    randomSub = -1
    
    qubo_array= QUBOsolver(matrix_Len, alpha, inputMatrix, inputVector, 10,simulation = sim)
    rfecv_array = RFECV_solver(inputMatrix, inputVector)
    scoreQubo, feature_nQ = getAccuracy(qubo_array, inputMatrix, inputVector, isQubo= True, isRFECV=False)
    scoreRfecv, feature_nR = getAccuracy(rfecv_array, inputMatrix, inputVector, isQubo= False, isRFECV=True)
    if(random_max == True):
        scoreRandom, feature_nRandom, randomSub = bestRandomSubset(20, matrix_Len, 100, inputMatrix, inputVector)
    
    printResults(fd, qubo_array, rfecv_array, scoreQubo, scoreRfecv, feature_nQ, feature_nR, scoreRand = scoreRandom, feature_nRand = feature_nRandom, randSub = randomSub)
    fd.write("////////////////////////////////////////////////////////////////////////////////////\n")
    #Testing methods with a percentage of noisy samples (noisy stemps are the percentage)
    
    noisy_steps = 3

    noisy_scoreQubo = np.zeros(noisy_steps)
    noisy_scoreRfecv = np.zeros(noisy_steps)
    noisy_feature_nQ = np.zeros(noisy_steps)
    noisy_feature_nR = np.zeros(noisy_steps)
    
    for i in range(noisy_steps):
        percentage_step = (i+1)*0.01
        noisy_matrix, noisy_vector, noisy_data_name = generate_noisy_data(inputMatrix, inputVector, percentage_step, matrix_Len, data_name)
        qubo_array_noisy = QUBOsolver(matrix_Len, alpha, noisy_matrix, noisy_vector, 10,simulation = sim) 
        rfecv_array_noisy = RFECV_solver(noisy_matrix, noisy_vector)
    
        noisy_scoreQubo[i], noisy_feature_nQ[i] = getAccuracy(qubo_array_noisy, noisy_matrix, noisy_vector, isQubo= True, isRFECV=False)
        noisy_scoreRfecv[i], noisy_feature_nR[i] = getAccuracy(rfecv_array_noisy, noisy_matrix, noisy_vector, isQubo= False, isRFECV=True)
        printResults_w_Noisy_samples(i+1, fd, qubo_array, rfecv_array, noisy_scoreQubo[i], noisy_scoreRfecv[i], noisy_feature_nQ[i], noisy_feature_nR[i])
    fd.write("////////////////////////////////////////////////////////////////////////////////////\n")
    ##Testing methods with new noisy features
    
    noisy_scoreQubo_feature = np.zeros(noisy_steps)
    noisy_scoreRfecv_feature = np.zeros(noisy_steps)
    noisy_feature_nQ_feature = np.zeros(noisy_steps)
    noisy_feature_nR_feature = np.zeros(noisy_steps)
    
    for i in range(noisy_steps):
        noisy_matrix, noisy_vector, noisy_data_name = generate_noisy_feature(inputMatrix, inputVector, i, matrix_Len, data_name)
        qubo_array_noisy = QUBOsolver(matrix_Len, alpha, noisy_matrix, noisy_vector, 10,simulation = sim) 
        rfecv_array_noisy = RFECV_solver(noisy_matrix, noisy_vector)
        qubo_detector = noisy_feature_detector(qubo_array_noisy, matrix_Len)
        rfecv_detector = noisy_feature_detector(rfecv_array_noisy, matrix_Len)   
        noisy_scoreQubo_feature[i], noisy_feature_nQ_feature[i] = getAccuracy(qubo_array_noisy, noisy_matrix, noisy_vector, isQubo= True, isRFECV=False)
        noisy_scoreRfecv_feature[i], noisy_feature_nR_feature[i] = getAccuracy(rfecv_array_noisy, noisy_matrix, noisy_vector, isQubo= False, isRFECV=True)
        printResults_w_Noisy_feature(i+1, fd, qubo_array, rfecv_array, noisy_scoreQubo_feature[i], noisy_scoreRfecv_feature[i], noisy_feature_nQ_feature[i], noisy_feature_nR_feature[i], qubo_detector, rfecv_detector)
    fd.write("////////////////////////////////////////////////////////////////////////////////////\n")
    
    
    print(colors.BOLD, colors.HEADER, "RESULTS", colors.ENDC)
    print(colors.RESULT, "QUBO = ", scoreQubo, " Feature number = ", feature_nQ)
    print("RFECV = ", scoreRfecv, " Feature number = ", feature_nR, colors.ENDC)
    if(random_max == True):
        print(colors.RESULT,"Random Max = ", scoreRandom, " Feature number = ", feature_nRandom, colors.ENDC)
        
    print(colors.BOLD, colors.HEADER, "RESULTS with NOISY SAMPLES", colors.ENDC)
    
    for i in range(noisy_steps):
        print(colors.BOLD, colors.HEADER, "Noise % = ", i+1, "% ", colors.ENDC)
        print(colors.RESULT, "QUBO = ", noisy_scoreQubo[i], " Feature number = ", noisy_feature_nQ[i])
        print("RFECV = ", noisy_scoreRfecv[i], " Feature number = ", noisy_feature_nR[i], colors.ENDC)
        print(colors.BOLD, colors.HEADER, "Done", colors.ENDC)
    
    print(colors.BOLD, colors.HEADER, "RESULTS with NOISY FEATURE", colors.ENDC)
    print(colors.BOLD, colors.HEADER, "To check if noise was selected look at the output.txt", colors.ENDC)
        
    for i in range(noisy_steps):
        print(colors.BOLD, colors.HEADER, "Noisy feature added  = ", i+1, colors.ENDC)
        print(colors.RESULT, "QUBO = ", noisy_scoreQubo_feature[i], " Feature number = ", noisy_feature_nQ_feature[i])
        print("RFECV = ", noisy_scoreRfecv_feature[i], " Feature number = ", noisy_feature_nR_feature[i], colors.ENDC)
        print(colors.BOLD, colors.HEADER, "Done", colors.ENDC)
    
    
    

if __name__=='__main__':
    main()