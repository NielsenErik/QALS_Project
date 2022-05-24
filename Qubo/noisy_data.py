import numpy as np
import random
import pandas as pd
from .utils import print_step

def generate_noisy_data(inputMatrix, inputVector, noise_dim_percent, dim, input_data_name):
    #inputMatrix = matrix from rescaledDataframe()
    #inputVector = vector from vector_V()
    #noise_dim_percent = percent of sample to make as noise
    #dim = dimension of the problem aka number of feature in the preprocessing matrix
    #input_data_name = name of dataset
    data_name = "Noisy Data from " + input_data_name
    buffer = str(noise_dim_percent*100)
    print_step("Generating "+data_name+" with "+buffer+"%% of noise")
    noise = inputMatrix 
    noisy_vector = inputVector
    noise_dim = int(len(noisy_vector)*noise_dim_percent)
    rows, columns = inputMatrix.shape
    new_array = np.zeros(dim)
    for i in range(noise_dim):
        for j in range(dim):
            random_index = random.randint(0, rows-1)
            random_binary = random.randint(0, 1)
            new_array[j] = inputMatrix[random_index][j]
        noise = np.vstack([noise, new_array])
        noisy_vector = np.append(noisy_vector, random_binary)
    tmp = np.insert(noise, dim, noisy_vector, axis = 1)
    dataset = pd.DataFrame(tmp)
    dataset_name = "German_dataset_noise_samples_"+str(noise_dim_percent)+".csv"
    path = "./Qubo/Data_folder/German/"+dataset_name
    dataset.to_csv(path, index=False)

def generate_noisy_feature(inputMatrix, inputVector, noise_feature_number, dim, input_data_name):
    #inputMatrix = matrix from rescaledDataframe()
    #inputVector = vector from vector_V()
    #noise_feature_number = number of noisy feature
    #dim = dimension of the problem aka number of feature in the preprocessing matrix
    #input_data_name = name of dataset
    data_name = "Noisy Feature from " + input_data_name
    buffer = str(noise_feature_number)
    print_step("Generating "+data_name+" with "+buffer+" feature of noise")
    noise = inputMatrix 
    rows, columns = inputMatrix.shape
    for i in range(noise_feature_number):
        new_column = np.zeros(rows)
        for j in range(rows):
            #new_column[j] = random.randint(0, 1)
            new_column[j] = random.random()
        noise = np.insert(noise, dim+i, new_column, axis=1)
    
    tmp = np.insert(noise, dim+noise_feature_number, inputVector, axis = 1)
    dataset = pd.DataFrame(tmp)
    dataset_name = "German_dataset_noise_feature_"+str(noise_feature_number)+".csv"
    path = "./Qubo/Data_folder/German/"+dataset_name
    dataset.to_csv(path, index=False)
    
def noisy_feature_detector(array, dim):
    alert = False
    if(np.any(array>(dim-1))):
        alert = True
    return alert   
