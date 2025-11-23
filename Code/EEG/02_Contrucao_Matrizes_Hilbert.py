import numpy as np
from scipy.signal import hilbert

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
Hilbert autism data
'''

a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
for paciente in range(np.shape(x_a)[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):
        sinais_hilbert = hilbert(x_a[paciente, :, k], axis=-1)
        fases = np.angle(sinais_hilbert)
        for i in range(np.shape(x_a)[1]):
            for j in range(i + 1, np.shape(x_a)[1]):
                fase_dif = fases[i] - fases[j]
                r = np.abs(np.mean(np.exp(1j * fase_dif)))
                a_hilbert[paciente, idx, i, j] = r
                a_hilbert[paciente, idx, j, i] = r

np.save("Matrizes_Adjacencia/Matrizes_adjacencia_autistas_hilbert.npy", a_hilbert)

''' 
Hilbert control data
'''

c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):
        sinais_hilbert = hilbert(x_c[paciente, :, k], axis=-1)
        fases = np.angle(sinais_hilbert)
        for i in range(np.shape(x_c)[1]):
            for j in range(i + 1, np.shape(x_c)[1]):
                fase_dif = fases[i] - fases[j]
                r = np.abs(np.mean(np.exp(1j * fase_dif)))
                c_hilbert[paciente, idx, i, j] = r
                c_hilbert[paciente, idx, j, i] = r

np.save("Matrizes_Adjacencia/Matrizes_adjacencia_controle_hilbert.npy", c_hilbert)

'''
a_hilbert/c_hilbert: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''
