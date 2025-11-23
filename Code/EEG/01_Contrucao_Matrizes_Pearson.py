import numpy as np
from scipy.stats import pearsonr
from itertools import combinations

'''
x_a: autism group

x_c: control group

Data format x_a and x_c: [Patient ID, Sensor ID, Wave ID, Time Series]

Wave ID: [0:original, 1:trash, 2:delta, 3:theta, 4:alpha, 5:beta, 6:gamma]
'''

xxx = np.load('formatted_data.npy')
x_a = xxx[0]  # autism data
x_c = xxx[1]  # control data

ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']

'''
Pearson autism data
'''

a_pearson = np.zeros((np.shape(x_a)[0], len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))

for paciente in range(np.shape(x_a)[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):
        matriz_a = np.zeros((np.shape(x_a)[1], np.shape(x_a)[1]))
        for i, j in combinations(range(np.shape(x_a)[1]), 2):
            r, _ = pearsonr(x_a[paciente, i, k, :], x_a[paciente, j, k, :])
            d = np.sqrt(2 * (1 - r))
            matriz_a[i, j] = d
            matriz_a[j, i] = d

        a_pearson[paciente, idx] = matriz_a

np.save("Matrizes_Adjacencia/Matrizes_adjacencia_autista_pearson.npy", a_pearson)

'''
Pearson control data
'''

c_pearson = np.zeros((np.shape(x_c)[0], len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):
        matriz_c = np.zeros((np.shape(x_c)[1], np.shape(x_c)[1]))
        for i, j in combinations(range(np.shape(x_c)[1]), 2):
            r, _ = pearsonr(x_c[paciente, i, k, :], x_c[paciente, j, k, :])
            d = np.sqrt(2 * (1 - r))
            matriz_c[i, j] = d
            matriz_c[j, i] = d

        c_pearson[paciente, idx] = matriz_c

np.save("Matrizes_Adjacencia/Matrizes_adjacencia_controle_pearson.npy", c_pearson)

'''
a_pearson/c_pearson: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''
