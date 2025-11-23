import numpy as np
from scipy.signal import hilbert
np.set_printoptions(threshold=np.inf)

'''
x_a: autism group

x_c: control group

Data format x_a and x_c: [Patient ID, Sensor ID, Wave ID, Time Series]

Wave ID: [0:original, 1:trash, 2:delta, 3:theta, 4:alpha, 5:beta, 6:gamma]
'''

x_a = np.load('x_autism_new_data.npy') # autism data
x_c = np.load('x_control_new_data.npy')  # control data



ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']

''' 
Hilbert autism data
'''


a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
for paciente in range(np.shape(x_a)[0]):
    print(f"Paciente {paciente}")
    for k in range(5):
        sinais_hilbert = hilbert(x_a[paciente, :, k], axis=-1)
        fases = np.angle(sinais_hilbert)

        # Calcula a matriz de conectividade
        for i in range(np.shape(x_a)[1]):
            for j in range(i + 1, np.shape(x_a)[1]):
                fase_dif = fases[i] - fases[j]
                r = np.abs(np.mean(np.exp(1j * fase_dif)))
                a_hilbert[paciente, k, i, j] = r
                a_hilbert[paciente, k, j, i] = r  # simetria


np.save("Matrizes_Adjacencia/Matrizes_adjacencia_autistas_hilbert_i.npy", a_hilbert)



''' 
Hilbert control data
'''

c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    print(f"Paciente {paciente}")
    for k in range(5):
        sinais_hilbert = hilbert(x_c[paciente, :, k], axis=-1)
        fases = np.angle(sinais_hilbert)

        # Calcula a matriz de conectividade
        for i in range(np.shape(x_c)[1]):
            for j in range(i + 1, np.shape(x_c)[1]):
                fase_dif = fases[i] - fases[j]
                r = np.abs(np.mean(np.exp(1j * fase_dif)))
                c_hilbert[paciente, k, i, j] = r
                c_hilbert[paciente, k, j, i] = r  # simetria

np.save("Matrizes_Adjacencia/Matrizes_adjacencia_controle_hilbert_i.npy", c_hilbert)


'''
a_hilbert/c_hilbert: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''




