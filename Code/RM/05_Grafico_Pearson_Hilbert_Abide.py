import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
import matplotlib.pyplot as plt
import time
np.set_printoptions(threshold=np.inf)


n_pacientes_geral = 500
tipo_analise = 'ccs'
qualidade = True
regressao_global = True



# Carregar dados
x_a = np.load(f'Matrizes_Adjacencia_Abide/x_a_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}_tensor.npy')
x_c = np.load(f'Matrizes_Adjacencia_Abide/x_c_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}_tensor.npy')
matrizes_a_hilbert = np.load(f"Matrizes_Adjacencia_Abide/Matrizes_adjacencia_a_hilbert_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy")
matrizes_c_hilbert = np.load(f"Matrizes_Adjacencia_Abide/Matrizes_adjacencia_c_hilbert_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy")


# Pearson - grupo autista
inicio = time.time()
matrizes_a_pearson_abide = np.zeros((np.shape(x_a)[0], len(x_a[0]), len(x_a[0])))

for paciente in range(np.shape(x_a)[0]):
    matriz_a = np.zeros((np.shape(x_a)[1], np.shape(x_a)[1]))
    for i, j in combinations(range(np.shape(x_a)[1]), 2):
        ts_i = x_a[paciente, i, :]
        ts_j = x_a[paciente, j, :]
        r, _ = pearsonr(ts_i, ts_j)
        matriz_a[i, j] = r
        matriz_a[j, i] = r
    matrizes_a_pearson_abide[paciente] = matriz_a
    print(f'{time.time() - inicio:.2f} segundos (autista - paciente {paciente})')
# Pearson - grupo controle


matrizes_c_pearson_abide = np.zeros((np.shape(x_c)[0], np.shape(x_c)[1], np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    matriz_c = np.zeros((np.shape(x_c)[1], np.shape(x_c)[1]))
    for i, j in combinations(range(np.shape(x_c)[1]), 2):
        ts_i = x_c[paciente, i, :]
        ts_j = x_c[paciente, j, :]
        r, _ = pearsonr(ts_i, ts_j)
        matriz_c[i, j] = r
        matriz_c[j, i] = r
    matrizes_c_pearson_abide[paciente] = matriz_c
    print(f'{time.time() - inicio:.2f} segundos (controle - paciente {paciente})')



'''
a_pearson/c_pearson: [ID paciente, matriz de adjacência (região i x região j)]
'''


# Gráfico do grupo autista
plt.figure(figsize=(8, 6))
x_a_flat = matrizes_a_pearson_abide[:, :].flatten()
y_a_flat = matrizes_a_hilbert[:, :].flatten()
plt.scatter(x_a_flat, y_a_flat, alpha=0.2, s=10)
plt.title(f'Grupo Autista_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}')
plt.xlabel('Correlação de Pearson')
plt.ylabel('Sincronização de Fase (Hilbert)')
plt.xticks(np.arange(-1, 1.1, 0.2))
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Graficos/Autistas_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}')

# Gráfico do grupo controle
plt.figure(figsize=(8, 6))
x_c_flat = matrizes_c_pearson_abide[:, :].flatten()
y_c_flat = matrizes_c_hilbert[:, :].flatten()
plt.scatter(x_c_flat, y_c_flat, alpha=0.2, s=10)
plt.title(f'Grupo Controle_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}')
plt.xlabel('Correlação de Pearson')
plt.ylabel('Sincronização de Fase (Hilbert)')
plt.xticks(np.arange(-1, 1.1, 0.2))
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Graficos/Controle_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}')
