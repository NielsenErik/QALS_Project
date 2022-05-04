#!/usr/local/bin/python3

import datetime

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
    
def printResults(fileDescriptor, qubo_array, rfecv_array, score_qubo, score_rfecv, nf_qubo, nf_efecv):
    #fileDescriptor = call file descriptor used in outputTxt
    fileDescriptor.write("RESULTS\n\n")
    tmp = "QUBO features are: " + str(qubo_array) + "\n"
    fileDescriptor.write(tmp)
    tmp = "RFECV features are: " + str(rfecv_array) + "\n\n"
    fileDescriptor.write(tmp)
    tmp = "QUBO accuracy score = " + str(score_qubo) + " with number of feature = " + str(nf_qubo) + "\n\n"
    fileDescriptor.write(tmp)  
    tmp = "RFECVaccuracy score = " + str(score_rfecv) + " with number of feature = " + str(nf_efecv) + "\n\n"
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