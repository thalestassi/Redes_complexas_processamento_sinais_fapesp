import numpy as np
from scipy.stats import pearsonr
from itertools import combinations

'''
x_a: autism group

x_c: control group

Data format x_a and x_c: [Patient ID, Sensor ID, Wave ID, Time Series]

Wave ID: [0:original, 1:trash, 2:delta, 3:theta, 4:alpha, 5:beta, 6:gamma]
'''

x_a = np.load('x_autism_new_data.npy')  # autism data
x_c = np.load('x_control_new_data.npy')  # control data


ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']

'''
Pearson autism data
'''

a_pearson = np.zeros((np.shape(x_a)[0], len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
a_pearson_1 = np.zeros((np.shape(x_a)[0], len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))

for paciente in range(np.shape(x_a)[0]):
    print(f"Paciente c {paciente}")
    for k in range(len(ondas)):
        matriz_a = np.zeros((np.shape(x_a)[1], np.shape(x_a)[1]))
        matriz_a_1 = np.zeros((np.shape(x_a)[1], np.shape(x_a)[1]))
        for i, j in combinations(range(np.shape(x_a)[1]), 2):
            serie_i = x_a[paciente, i, k, :]
            serie_j = x_a[paciente, j, k, :]
            if np.all(serie_i == serie_i[0]) or np.all(serie_j == serie_j[0]):
                r = 0.0
            else:
                r, _ = pearsonr(serie_i, serie_j)
            d = np.sqrt(2 * (1 - r))
            matriz_a[i, j] = d
            matriz_a[j, i] = d
            matriz_a_1[i, j] = r
            matriz_a_1[j, i] = r

        a_pearson[paciente, k] = matriz_a
        a_pearson_1[paciente, k] = matriz_a_1

np.save("Matrizes_Adjacencia/Matrizes_adjacencia_autista_pearson_i.npy", a_pearson)
np.save("Matrizes_Adjacencia/Matrizes_adjacencia_autista_pearson_i_1.npy", a_pearson_1)

'''
Pearson para controle
'''

c_pearson = np.zeros((np.shape(x_c)[0], len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
c_pearson_1 = np.zeros((np.shape(x_c)[0], len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    print(f"Paciente c {paciente}")
    for k in range(len(ondas)):
        matriz_c = np.zeros((np.shape(x_c)[1], np.shape(x_c)[1]))
        matriz_c_1 = np.zeros((np.shape(x_c)[1], np.shape(x_c)[1]))
        for i, j in combinations(range(np.shape(x_c)[1]), 2):
            serie_i = x_c[paciente, i, k, :]
            serie_j = x_c[paciente, j, k, :]
            if np.all(serie_i == serie_i[0]) or np.all(serie_j == serie_j[0]):
                r = 0.0  # ou np.nan, se preferir
            else:
                r, _ = pearsonr(serie_i, serie_j)
            d = np.sqrt(2 * (1 - r))
            matriz_c[i, j] = d
            matriz_c[j, i] = d
            matriz_c_1[i, j] = r
            matriz_c_1[j, i] = r
        c_pearson[paciente, k] = matriz_c
        c_pearson_1[paciente, k] = matriz_c_1

np.save("Matrizes_Adjacencia/Matrizes_adjacencia_controle_pearson_i.npy", c_pearson)
np.save("Matrizes_Adjacencia/Matrizes_adjacencia_controle_pearson_i_1.npy", c_pearson_1)

'''
a_pearson/c_pearson: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''
