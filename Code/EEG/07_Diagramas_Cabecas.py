import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import hilbert
import pandas as pd
import networkx as nx
np.set_printoptions(threshold=np.inf)

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

eletrodos = ['F7', 'T3', 'T5', 'Fp1', 'F3', 'C3', 'P3',
             'O1', 'F8', 'T4', 'T6', 'Fp2', 'F4', 'C4',
             'P4', 'O2', 'Fz', 'Cz', 'Pz', 'Oz', 'A1',
             'A2']

chs = {'F7': [-0.073, 0.047],
       'T3': [-0.085, 0],
       'T5': [-0.073, -0.047],
       'FP1': [-0.03, 0.08],
       'F3': [-0.04, 0.041],
       'C3': [-0.045, 0],
       'P3': [-0.04, -0.041],
       'O1': [-0.03, -0.08],
       'F8': [0.073, 0.047],
       'T4': [0.085, 0],
       'T6': [0.07, -0.047],
       'FP2': [0.03, 0.08],
       'F4': [0.04, 0.041],
       'C4': [0.045, 0],
       'P4': [0.04, -0.041],
       'O2': [0.03, -0.08],
       'Fz': [0, 0.038],
       'Cz': [0, 0],
       'Pz': [0, -0.038],
       'Oz': [0, -0.09],
       'A1': [-0.1, 0],
       'A2': [0.1, 0]
       }

channels = pd.DataFrame(chs).transpose()

for key in chs.keys():
    chs[key] += [0]



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
        matriz_a = a_hilbert[paciente, idx, :, :]

        valores_flat = matriz_a.flatten()
        indices_maiores = np.argsort(valores_flat)[-40:]
        posicoes_na_matriz = np.zeros((40, 2))
        contador = 0
        indices = []
        for i in indices_maiores:
            posicoes_na_matriz[contador, 0] = i // 22
            posicoes_na_matriz[contador, 1] = i % 22
            contador = contador + 1

        for par_eletrodos in posicoes_na_matriz:
            par_ordenado = tuple(sorted(par_eletrodos))
            if par_ordenado not in indices:
                indices.append(par_ordenado)
        mont = mne.channels.make_dig_montage(chs)
        figura = mont.plot()
        adj_matriz_maiores_sem_peso = np.zeros((len(chs), len(chs)))
        for indice in indices:
            adj_matriz_maiores_sem_peso[int(indice[0]), int(indice[1])] = adj_matriz_maiores_sem_peso[
                int(indice[1]), int(indice[0])] = 1
        G = nx.Graph()
        for key, value in chs.items():
            G.add_node(key, pos=(value[0], value[1]))
        for i in indices:
            node1 = list(chs.keys())[int(i[0])]
            node2 = list(chs.keys())[int(i[1])]
            G.add_edge(node1, node2)
        pos = nx.get_node_attributes(G, 'pos')
        ax = figura.axes[0]
        for edge in G.edges():
            x_values = [pos[edge[0]][0], pos[edge[1]][0]]
            y_values = [pos[edge[0]][1], pos[edge[1]][1]]
            ax.plot(x_values, y_values, color='grey', linewidth=1)
        plt.savefig(f"Redes_Cabeças\Rede_paciente_autista_{paciente}_onda_{ondas[idx]}.png", dpi=300)
        plt.close()



''''' Hilbert para controle
'''''

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
        matriz_c = c_hilbert[paciente, idx, :, :]
        valores_flat = matriz_c.flatten()
        indices_maiores = np.argsort(valores_flat)[-40:]
        posicoes_na_matriz = np.zeros((40, 2))
        contador = 0
        indices = []

        for i in indices_maiores:
            posicoes_na_matriz[contador, 0] = i // 22
            posicoes_na_matriz[contador, 1] = i % 22
            contador = contador + 1

        for par_eletrodos in posicoes_na_matriz:
            par_ordenado = tuple(sorted(par_eletrodos))
            if par_ordenado not in indices:
                indices.append(par_ordenado)
        mont = mne.channels.make_dig_montage(chs)
        figura = mont.plot()

        adj_matriz_maiores_sem_peso = np.zeros((len(chs), len(chs)))
        for indice in indices:
            adj_matriz_maiores_sem_peso[int(indice[0]), int(indice[1])] = adj_matriz_maiores_sem_peso[
                int(indice[1]), int(indice[0])] = 1

        G = nx.Graph()
        for key, value in chs.items():
            G.add_node(key, pos=(value[0], value[1]))
        for i in indices:
            node1 = list(chs.keys())[int(i[0])]
            node2 = list(chs.keys())[int(i[1])]
            G.add_edge(node1, node2)
        pos = nx.get_node_attributes(G, 'pos')
        ax = figura.axes[0]
        for edge in G.edges():
            x_values = [pos[edge[0]][0], pos[edge[1]][0]]
            y_values = [pos[edge[0]][1], pos[edge[1]][1]]
            ax.plot(x_values, y_values, color='grey', linewidth=1)

        plt.savefig(f"Redes_Cabeças\Rede_paciente_controle_{paciente}_onda_{ondas[idx]}.png", dpi=300)
        plt.close()



