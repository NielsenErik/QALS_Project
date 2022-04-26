import numpy as np
import random
from .german_credit_data import german_credit_data
from .preprocessing_data import rescaledDataframe

def genearate_noisy_data(inputMatrix, inputVector, noise_dim_percent, dim, input_data_name):
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
    print(noisy_vector)
    return noise, noisy_vector, data_name

'''data = german_credit_data()
matrix = rescaledDataframe(data)
genearate_noisy_data(matrix, 5, 48)'''