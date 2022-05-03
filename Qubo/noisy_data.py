import numpy as np
import random

def generate_noisy_data(inputMatrix, inputVector, noise_dim_percent, dim, input_data_name):
    #inputMatrix = matrix from rescaledDataframe()
    #inputVector = vector from vector_V()
    #noise_dim_percent = percent of sample to make as noise
    #dim = dimension of the problem aka number of feature in the preprocessing matrix
    #input_data_name = name of dataset
    
    data_name = "Noisy Data from " + input_data_name
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
    return noise, noisy_vector, data_name

def generate_noisy_feature(inputMatrix, inputVector, noise_feature_number, dim, input_data_name):
    #inputMatrix = matrix from rescaledDataframe()
    #inputVector = vector from vector_V()
    #noise_feature_number = number of noisy feature
    #dim = dimension of the problem aka number of feature in the preprocessing matrix
    #input_data_name = name of dataset
    data_name = "Noisy Feature from " + input_data_name
    noise = inputMatrix 
    rows, columns = inputMatrix.shape
    
    noisy_vector = inputVector
    for i in range(noise_feature_number):
        new_column = np.zeros(rows)
        for j in range(rows):
            new_column[j] = random.randint(0, 1)
        noise = np.insert(noise, -1, new_column, axis=1)
        
    
    return noise, noisy_vector, data_name
    
def noisy_feature_detector(array, dim):
    alert = False

    if(np.any(array>(dim-1))):
        alert = True
    return alert   
