import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
import time

np.set_printoptions(threshold=np.inf)

n_pacientes_geral = 50
tipo_analise = 'cpac'
qualidade = True
regressao_global = True


'''
x_a: autism group

x_c: control group

Data format x_a and x_c: [Patient ID, Região Cérebro, Time Series]

'''

x_a_lista = np.load(f'Matrizes_Adjacencia_Abide/x_a_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy', allow_pickle=True)
x_c_lista = np.load(f'Matrizes_Adjacencia_Abide/x_c_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy', allow_pickle=True)
x_a = np.load(f'Matrizes_Adjacencia_Abide/x_a_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}_tensor.npy')
x_c = np.load(f'Matrizes_Adjacencia_Abide/x_c_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}_tensor.npy')


'''
Pearson para autismo
'''

inicio = time.time()


a_pearson_abide = np.zeros((len(x_a[:,0,0]), len(x_a[0]), len(x_a[0])))
for paciente in range(len(x_a[:, 0, 0])):
    matriz_a = np.zeros((len(x_a[0]), len(x_a[0])))
    for i, j in combinations(range(len(x_a[0])), 2):
        ts_i = x_a[paciente, i, :]
        ts_j = x_a[paciente, j, :]
        if np.std(ts_i) == 0 or np.std(ts_j) == 0:
            d = 0
        else:
            r, _ = pearsonr(ts_i, ts_j)
            d = np.sqrt(2 * (1 - r))
        matriz_a[i, j] = d
        matriz_a[j, i] = d
    a_pearson_abide[paciente] = matriz_a
    fim = time.time()
    print(f'{fim - inicio} segundos')

np.save(f"Matrizes_Adjacencia_Abide/Matrizes_adjacencia_a_pearson_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy", a_pearson_abide)

'''
Pearson para controle
'''

c_pearson_abide = np.zeros((len(x_c[:,0,0]), len(x_c[0]), len(x_c[0])))
for paciente in range(len(x_c[:,0,0])):
    matriz_c = np.zeros((len(x_c[0]), len(x_c[0])))
    for i, j in combinations(range(len(x_c[0])), 2):
        ts_i = x_c[paciente, i, :]
        ts_j = x_c[paciente, j, :]
        if np.std(ts_i) == 0 or np.std(ts_j) == 0:
            d = 0
        else:
            r, _ = pearsonr(ts_i, ts_j)
            d = np.sqrt(2 * (1 - r))
        matriz_c[i, j] = d
        matriz_c[j, i] = d
    c_pearson_abide[paciente] = matriz_c
    fim = time.time()
    print(f'{fim - inicio} segundos')

np.save(f"Matrizes_Adjacencia_Abide/Matrizes_adjacencia_c_pearson_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy", c_pearson_abide)

'''
a_pearson/c_pearson: [Patient ID, Matriz de adjacência (Regioes i e j)]
'''
