import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import hilbert
import networkx as nx
from scipy.stats import pearsonr
from itertools import combinations

np.set_printoptions(threshold=np.inf)

'''
x_a: autism group
x_c: control group

Data format x_a and x_c:
[Patient ID, Sensor ID, Wave ID, Time Series]

Wave ID:
[0:original, 1:trash, 2:delta, 3:theta, 4:alpha, 5:beta, 6:gamma]
'''

xxx = np.load('formatted_data.npy')
x_a = xxx[0]
x_c = xxx[1]

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

for key in chs.keys():
    chs[key] += [0]


a_hilbert = np.zeros((x_a.shape[0], len(ondas), x_a.shape[1], x_a.shape[1]))

for paciente in range(x_a.shape[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):

        sinais_hilbert = hilbert(x_a[paciente, :, k], axis=-1)
        fases = np.angle(sinais_hilbert)

        for i in range(x_a.shape[1]):
            for j in range(i + 1, x_a.shape[1]):
                fase_dif = fases[i] - fases[j]
                r = np.abs(np.mean(np.exp(1j * fase_dif)))
                a_hilbert[paciente, idx, i, j] = r
                a_hilbert[paciente, idx, j, i] = r

        matriz_a = a_hilbert[paciente, idx]

        n = matriz_a.shape[0]
        i_upper = np.triu_indices(n, k=1)
        valores = matriz_a[i_upper]
        k_edges = int(0.10 * len(valores))
        idx_maiores = np.argsort(valores)[-k_edges:]
        indices = list(zip(i_upper[0][idx_maiores],
                           i_upper[1][idx_maiores]))


        mont = mne.channels.make_dig_montage(chs)
        figura = mont.plot()

        G = nx.Graph()
        for key, value in chs.items():
            G.add_node(key, pos=(value[0], value[1]))

        for i, j in indices:
            node1 = list(chs.keys())[i]
            node2 = list(chs.keys())[j]
            G.add_edge(node1, node2)

        pos = nx.get_node_attributes(G, 'pos')
        ax = figura.axes[0]

        for edge in G.edges():
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]],
                    [pos[edge[0]][1], pos[edge[1]][1]],
                    color='grey', linewidth=1)

        plt.savefig(
            f"Redes_Cabeças/Rede_paciente_autista_{paciente}_onda_{ondas[idx]}.png",
            dpi=300
        )
        plt.close()


c_hilbert = np.zeros((x_c.shape[0], len(ondas), x_c.shape[1], x_c.shape[1]))

for paciente in range(x_c.shape[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):

        sinais_hilbert = hilbert(x_c[paciente, :, k], axis=-1)
        fases = np.angle(sinais_hilbert)

        for i in range(x_c.shape[1]):
            for j in range(i + 1, x_c.shape[1]):
                fase_dif = fases[i] - fases[j]
                r = np.abs(np.mean(np.exp(1j * fase_dif)))
                c_hilbert[paciente, idx, i, j] = r
                c_hilbert[paciente, idx, j, i] = r

        matriz_c = c_hilbert[paciente, idx]

        n = matriz_c.shape[0]
        i_upper = np.triu_indices(n, k=1)
        valores = matriz_c[i_upper]
        k_edges = int(0.10 * len(valores))
        idx_maiores = np.argsort(valores)[-k_edges:]
        indices = list(zip(i_upper[0][idx_maiores],
                           i_upper[1][idx_maiores]))

        mont = mne.channels.make_dig_montage(chs)
        figura = mont.plot()

        G = nx.Graph()
        for key, value in chs.items():
            G.add_node(key, pos=(value[0], value[1]))

        for i, j in indices:
            node1 = list(chs.keys())[i]
            node2 = list(chs.keys())[j]
            G.add_edge(node1, node2)

        pos = nx.get_node_attributes(G, 'pos')
        ax = figura.axes[0]

        for edge in G.edges():
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]],
                    [pos[edge[0]][1], pos[edge[1]][1]],
                    color='grey', linewidth=1)

        plt.savefig(
            f"Redes_Cabeças/Rede_paciente_controle_{paciente}_onda_{ondas[idx]}.png",
            dpi=300
        )
        plt.close()




a_pearson = np.zeros((x_a.shape[0], len(ondas), x_a.shape[1], x_a.shape[1]))

for paciente in range(x_a.shape[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):

        matriz_a = np.zeros((x_a.shape[1], x_a.shape[1]))

        for i, j in combinations(range(x_a.shape[1]), 2):
            r, _ = pearsonr(x_a[paciente, i, k, :],
                            x_a[paciente, j, k, :])
            d = np.sqrt(2 * (1 - r))
            matriz_a[i, j] = d
            matriz_a[j, i] = d

        a_pearson[paciente, idx] = matriz_a

        n = matriz_a.shape[0]
        i_upper = np.triu_indices(n, k=1)
        valores = matriz_a[i_upper]
        k_edges = int(0.10 * len(valores))
        idx_menores = np.argsort(valores)[:k_edges]
        indices = list(zip(i_upper[0][idx_menores],
                           i_upper[1][idx_menores]))

        mont = mne.channels.make_dig_montage(chs)
        figura = mont.plot()

        adj_matriz_maiores_sem_peso = np.zeros((len(chs), len(chs)))
        for i, j in indices:
            adj_matriz_maiores_sem_peso[i, j] = 1
            adj_matriz_maiores_sem_peso[j, i] = 1

        G = nx.Graph()
        for key, value in chs.items():
            G.add_node(key, pos=(value[0], value[1]))

        for i, j in indices:
            node1 = list(chs.keys())[i]
            node2 = list(chs.keys())[j]
            G.add_edge(node1, node2)

        pos = nx.get_node_attributes(G, 'pos')
        ax = figura.axes[0]

        for edge in G.edges():
            x_values = [pos[edge[0]][0], pos[edge[1]][0]]
            y_values = [pos[edge[0]][1], pos[edge[1]][1]]
            ax.plot(x_values, y_values, color='grey', linewidth=1)

        plt.savefig(
            f"Redes_Cabeças_Pearson/Rede_paciente_autista_{paciente}_onda_{ondas[idx]}.png",
            dpi=300
        )
        plt.close()

c_pearson = np.zeros((x_c.shape[0], len(ondas), x_c.shape[1], x_c.shape[1]))

for paciente in range(x_c.shape[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):

        matriz_c = np.zeros((x_c.shape[1], x_c.shape[1]))

        for i, j in combinations(range(x_c.shape[1]), 2):
            r, _ = pearsonr(x_c[paciente, i, k, :],
                            x_c[paciente, j, k, :])
            d = np.sqrt(2 * (1 - r))
            matriz_c[i, j] = d
            matriz_c[j, i] = d

        c_pearson[paciente, idx] = matriz_c

        n = matriz_c.shape[0]
        i_upper = np.triu_indices(n, k=1)
        valores = matriz_c[i_upper]
        k_edges = int(0.10 * len(valores))
        idx_menores = np.argsort(valores)[:k_edges]
        indices = list(zip(i_upper[0][idx_menores],
                           i_upper[1][idx_menores]))

        mont = mne.channels.make_dig_montage(chs)
        figura = mont.plot()

        adj_matriz_maiores_sem_peso = np.zeros((len(chs), len(chs)))
        for i, j in indices:
            adj_matriz_maiores_sem_peso[i, j] = 1
            adj_matriz_maiores_sem_peso[j, i] = 1

        G = nx.Graph()
        for key, value in chs.items():
            G.add_node(key, pos=(value[0], value[1]))

        for i, j in indices:
            node1 = list(chs.keys())[i]
            node2 = list(chs.keys())[j]
            G.add_edge(node1, node2)

        pos = nx.get_node_attributes(G, 'pos')
        ax = figura.axes[0]

        for edge in G.edges():
            x_values = [pos[edge[0]][0], pos[edge[1]][0]]
            y_values = [pos[edge[0]][1], pos[edge[1]][1]]
            ax.plot(x_values, y_values, color='grey', linewidth=1)

        plt.savefig(
            f"Redes_Cabeças_Pearson/Rede_paciente_controle_{paciente}_onda_{ondas[idx]}.png",
            dpi=300
        )
        plt.close()