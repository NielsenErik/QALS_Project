#!/usr/local/bin/python3
import numpy as np
import signal
import sys

from Qubo.qals import qals_solver
from Qubo.import_data import german_credit_data, australian_credit_data, polish_bankrupcy_data, synthetic_Data
from Qubo.preprocessing_data import vector_V_synthetic, rescaledDataframe_synthetic, rescaledDataframe_German, vector_V_German, vector_V_Australian, rescaledDataframe_Australian, vector_V_Polish, normalizing_Polish
from Qubo.colors import colors
from Qubo.solverQubo import QUBOsolver
from Qubo.solverRFECV import RFECV_solver
from Qubo.getAccuracyScore import getAccuracy
from Qubo.noisy_data import generate_noisy_data, generate_noisy_feature, noisy_feature_detector, generate_random_data
from Qubo.print_on_file import printStartInfos, printResults_w_Noisy_samples, printResults_w_Noisy_feature, printResults, outputTxt, end_file
from Qubo.qubo_matrix import qubo_Matrix
from Qubo.utils import print_step
import time

def signal_handler(sig, frame):
    print(" ")
    print(colors.FAIL,'Interrupting program', colors.ENDC)
    sys.exit(0)

def header_script():
    #just the title and headar that appear on terminal
    print(colors.BOLD, colors.HEADER, "This program aim is optimal feature selection") 
    print("in credit scoring using quantum annealer", colors.ENDC)

def ask_for_simulation():
    #This function ask if it is wanted a simulation
    #or real usage touser before send the problem
    print(colors.ORANGE ,"To INTERRUPT program at any time press CTRL+C", colors.ENDC)   
    print(" ")
    print(colors.ORANGE ,"Would you like to try a simulation or run Dwave?", colors.ENDC)    
    print(colors.ORANGE ,"Simulation might not return the true results for QUBO and QALS", colors.ENDC)    
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

def ask_which_dataset():
    print(colors.ORANGE ,"Wich dataset would you like to do feature selection?", colors.ENDC)    
    print(colors.ORANGE ,"[a] German Credit Data", colors.ENDC)  
    print(colors.ORANGE ,"[b] Australian Credit Data", colors.ENDC)  
    print(colors.ORANGE ,"[c] Polish Bankruptcy Data", colors.ENDC)
    print(colors.ORANGE ,"[d] Synthetic Data", colors.ENDC)    
    print(colors.ORANGE ,"[e] To exit program", colors.ENDC)  
   
    check = False
    while(check == False):
        print(colors.ORANGE)
        answer = input("Which dataset? ")
        print(colors.ENDC)
        if(answer == 'a' or answer == 'b' or answer == 'c' or answer == 'e' or answer == 'd'):
            check = True
        elif(answer == 'e'):
            check = True
            print(colors.WARNING, "Terminating program...", colors.ENDC)
            exit()
        else:
            print(colors.FAIL, "Wrong answer, try again!", colors.ENDC)
            check = False

    return answer  
def choose_dataset(answer):
    if(answer == 'a'):
        data, data_name = german_credit_data()
        inputMatrix, matrix_Len = rescaledDataframe_German(data)
        inputVector = vector_V_German(data)
        alpha = 0.977
    elif(answer == 'c'):
        data, data_name = polish_bankrupcy_data()
        inputMatrix,matrix_Len = normalizing_Polish(data)
        inputVector = vector_V_Polish(data)
        alpha = 0.9  
    
    elif(answer == 'b'):
        data, data_name = australian_credit_data()
        inputMatrix, matrix_Len = rescaledDataframe_Australian(data)
        inputVector = vector_V_Australian(data)
        alpha = 0.05  
    elif(answer == 'd'):
        try:
            print(colors.ORANGE)
            nFeature = input("With how many features? ")
            print(colors.ENDC)
            data, data_name = synthetic_Data(nFeature)
            inputMatrix, matrix_Len = rescaledDataframe_synthetic(data, nFeature)
            inputVector = vector_V_synthetic(data)
            alpha = 0.977
        except:
            print(colors.FAIL+"dataset with "+nFeature+" features doesn't exists"+colors.ENDC)
            
    return data_name, inputMatrix, matrix_Len, inputVector, alpha

def allFeaturesArray(dim):
    tmp = np.ones(dim)
    output = np.asarray(np.where(tmp>0)).flatten()
    return output
    
def main():
    #main function of the program
    
    header_script()
    
    sim = ask_for_simulation()
    answer = ask_which_dataset()
    fileOutput = 'output.txt'
    fd = outputTxt(fileOutput, sim)
    signal.signal(signal.SIGINT, signal_handler)
    ###########################################
    #variables needed globally
    
    n_reads_annealer = 30
    qals_iteration = 10
    qals_n_reads_annealer = 1
    noisy_steps = 3
    data_name, inputMatrix, matrix_Len, inputVector, alpha = choose_dataset(answer=answer)
    data_name_Qals = data_name
    inputMatrix_Qals = inputMatrix  
    matrix_Len_Qals = matrix_Len
    inputVector_Qals= inputVector
    alpha_Qals = alpha
    ##############################################################################################
    all_f_array = allFeaturesArray(matrix_Len)
    scoreDataset, _ = getAccuracy(all_f_array , inputMatrix, inputVector, isQubo = False, isRFECV =False, isAllFeature = True)
    
    printStartInfos(alpha, data_name, fd, scoreDataset)
   
    qubo_array= QUBOsolver(matrix_Len, alpha, inputMatrix, inputVector, n_reads_annealer,simulation = sim)
    rfecv_array = RFECV_solver(inputMatrix, inputVector)
    scoreQubo, feature_nQ = getAccuracy(qubo_array, inputMatrix, inputVector, isQubo= True, isRFECV=False)
    scoreRfecv, feature_nR = getAccuracy(rfecv_array, inputMatrix, inputVector, isQubo= False, isRFECV=True)
    #QALS   
    qubo_qals = qubo_Matrix(alpha_Qals, inputMatrix_Qals, inputVector_Qals)
    z_qals, conv_time = qals_solver(d_min=70, eta_prob_dec_rate=0.01, i_max=qals_iteration, k_n_reads=qals_n_reads_annealer, lambda_zero=3/2, dim_problem=matrix_Len_Qals, N_it_const_prob=10, N_max=100, p_delta=0.1, q_perm_prob=0.2, topology='pegasus', QUBO=qubo_qals, log_DIR='qals_output.txt', sim = sim)
    z_pos = np.asarray(np.where(z_qals>0)).flatten()
    scoreQALS, feature_nQALS = getAccuracy(z_pos, inputMatrix, inputVector, isQubo= False, isRFECV=False)

    printResults(fd, qubo_array, rfecv_array, scoreQubo, scoreRfecv, feature_nQ, feature_nR, z_pos, scoreQALS, feature_nQALS)
    fd.write("////////////////////////////////////////////////////////////////////////////////////\n")
    #Testing methods with a percentage of noisy samples (noisy stemps are the percentage)
    
    

    noisy_scoreQubo = np.zeros(noisy_steps)
    noisy_scoreRfecv = np.zeros(noisy_steps)
    noisy_feature_nQ = np.zeros(noisy_steps)
    noisy_feature_nR = np.zeros(noisy_steps)
    noisy_all_Score = np.zeros(noisy_steps)
    
    for i in range(noisy_steps):
        percentage_step = (i+1)*0.01
        noisy_matrix, noisy_vector, noisy_data_name = generate_noisy_data(inputMatrix, inputVector, percentage_step, matrix_Len, data_name)
        qubo_array_noisy = QUBOsolver(matrix_Len, alpha, noisy_matrix, noisy_vector, n_reads_annealer,simulation = sim) 
        rfecv_array_noisy = RFECV_solver(noisy_matrix, noisy_vector)
        
        noisy_all_Score[i], _ = getAccuracy(all_f_array, noisy_matrix, noisy_vector, isQubo= True, isRFECV=False)
        noisy_scoreQubo[i], noisy_feature_nQ[i] = getAccuracy(qubo_array_noisy, noisy_matrix, noisy_vector, isQubo= True, isRFECV=False)
        noisy_scoreRfecv[i], noisy_feature_nR[i] = getAccuracy(rfecv_array_noisy, noisy_matrix, noisy_vector, isQubo= False, isRFECV=True)
        printResults_w_Noisy_samples(i+1, fd, qubo_array_noisy, rfecv_array_noisy, noisy_scoreQubo[i], noisy_scoreRfecv[i], noisy_feature_nQ[i], noisy_feature_nR[i], noisy_all_Score[i])
        time.sleep(1)
    fd.write("////////////////////////////////////////////////////////////////////////////////////\n")
    ##Testing methods with new noisy features
    
    noisy_scoreQubo_feature = np.zeros(noisy_steps)
    noisy_scoreRfecv_feature = np.zeros(noisy_steps)
    noisy_feature_nQ_feature = np.zeros(noisy_steps)
    noisy_feature_nR_feature = np.zeros(noisy_steps)
    noisy_all_Score = np.zeros(noisy_steps)
    
    for i in range(noisy_steps):
        noisy_matrix, noisy_vector, noisy_data_name = generate_noisy_feature(inputMatrix, inputVector, i+1, matrix_Len, data_name)
        qubo_array_noisy = QUBOsolver(matrix_Len, alpha, noisy_matrix, noisy_vector, n_reads_annealer,simulation = sim) 
        rfecv_array_noisy = RFECV_solver(noisy_matrix, noisy_vector)
        
        qubo_detector = noisy_feature_detector(qubo_array_noisy, matrix_Len)
        rfecv_detector = noisy_feature_detector(rfecv_array_noisy, matrix_Len) 
        
        noisy_all_Score[i], _ = getAccuracy(all_f_array, noisy_matrix, noisy_vector, isQubo= True, isRFECV=False)          
        noisy_scoreQubo_feature[i], noisy_feature_nQ_feature[i] = getAccuracy(qubo_array_noisy, noisy_matrix, noisy_vector, isQubo= True, isRFECV=False)
        noisy_scoreRfecv_feature[i], noisy_feature_nR_feature[i] = getAccuracy(rfecv_array_noisy, noisy_matrix, noisy_vector, isQubo= False, isRFECV=True)
        printResults_w_Noisy_feature(i+1, fd, qubo_array_noisy, rfecv_array_noisy, noisy_scoreQubo_feature[i], noisy_scoreRfecv_feature[i], noisy_feature_nQ_feature[i], noisy_feature_nR_feature[i], qubo_detector, rfecv_detector, noisy_all_Score[i])
        time.sleep(1)
    fd.write("////////////////////////////////////////////////////////////////////////////////////\n")
    
    
    print("/////////////////////////////////////////////////////////////////////////////////////////////")
    
    print(colors.BOLD, colors.HEADER, "RESULTS", colors.ENDC)
    print(colors.RESULT, "All feature = ", scoreDataset)
    print(colors.RESULT, "QUBO = ", scoreQubo, " Feature number = ", feature_nQ)
    print("RFECV = ", scoreRfecv, " Feature number = ", feature_nR, colors.ENDC)
    print(colors.RESULT, "Qals = ", scoreQALS, " Feature number = ", feature_nQALS)
        
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
    
    print("///////////////////////////////////////////////////////////////////////////////")
    end_file(fd)  
    
    

    
if __name__=='__main__':
    main()