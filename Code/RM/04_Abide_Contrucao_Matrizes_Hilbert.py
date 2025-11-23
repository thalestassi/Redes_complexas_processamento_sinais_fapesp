import numpy as np
from scipy.signal import hilbert
import time
np.set_printoptions(threshold=np.inf)


n_pacientes_geral = 500
tipo_analise = 'ccs'
qualidade = True
regressao_global = True





'''
x_a: autism group

x_c: control group

'''

x_a_lista = np.load(f'Matrizes_Adjacencia_Abide/x_a_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy', allow_pickle=True)
x_c_lista = np.load(f'Matrizes_Adjacencia_Abide/x_c_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy', allow_pickle=True )
x_a = np.load(f'Matrizes_Adjacencia_Abide/x_a_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}_tensor.npy')
x_c = np.load(f'Matrizes_Adjacencia_Abide/x_c_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}_tensor.npy')


''''' Hilbert para autismo
'''''

inicio = time.time()



a_hilbert_abide = np.zeros((np.shape(x_a)[0], np.shape(x_a)[1], np.shape(x_a)[1]))
for paciente in range(np.shape(x_a)[0]):
    sinais_hilbert = hilbert(x_a[paciente, :], axis=-1)
    fases = np.angle(sinais_hilbert)
    # Calcula a matriz de conectividade
    for i in range(np.shape(x_a)[1]):
        for j in range(i + 1, np.shape(x_a)[1]):
            fase_dif = fases[i] - fases[j]
            r = np.abs(np.mean(np.exp(1j * fase_dif)))
            a_hilbert_abide[paciente, i, j] = r
            a_hilbert_abide[paciente, j, i] = r  # simetria
    fim = time.time()
    print(f'{fim - inicio} segundos')

np.save(f"Matrizes_Adjacencia_Abide/Matrizes_adjacencia_a_hilbert_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy", a_hilbert_abide)



''''' Hilbert para controle
'''''

c_hilbert_abide = np.zeros((np.shape(x_c)[0], np.shape(x_c)[1], np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    sinais_hilbert = hilbert(x_c[paciente, :], axis=-1)
    fases = np.angle(sinais_hilbert)
    # Calcula a matriz de conectividade
    for i in range(np.shape(x_c)[1]):
        for j in range(i + 1, np.shape(x_c)[1]):
            fase_dif = fases[i] - fases[j]
            r = np.abs(np.mean(np.exp(1j * fase_dif)))
            c_hilbert_abide[paciente, i, j] = r
            c_hilbert_abide[paciente, j, i] = r  # simetria
    fim = time.time()
    print(f'{fim - inicio} segundos')

np.save(f"Matrizes_Adjacencia_Abide/Matrizes_adjacencia_c_hilbert_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy", c_hilbert_abide)


'''
a_hilbert/c_hilbert: [Patient ID,  Matriz de adjacência (Regiões i e j)]
'''




