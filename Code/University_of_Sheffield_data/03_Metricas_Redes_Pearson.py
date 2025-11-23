import numpy as np
import networkx as nx
from scipy.stats import kurtosis, entropy
from sklearn.metrics import mutual_info_score


'''
files.npy: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''
ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']

matrizes_a_pearson = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_autista_pearson_i.npy')
matrizes_c_pearson = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_controle_pearson_i.npy')
matrizes_a_hilbert = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_autistas_hilbert_i.npy')
matrizes_c_hilbert = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_controle_hilbert_i.npy')

x_a = np.load('x_autism_new_data.npy') # autism data
x_c = np.load('x_control_new_data.npy')  # control data

'''
Grau Médio
'''
grau_medio_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        grau_medio = np.sum(matrizes_a_pearson[paciente, onda].flatten()) / np.shape(x_c)[1]
        grau_medio_a_pearson[paciente, onda] = grau_medio

grau_medio_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        grau_medio = np.sum(matrizes_c_pearson[paciente, onda].flatten()) / np.shape(x_c)[1]
        grau_medio_c_pearson[paciente, onda] = grau_medio

grau_medio_a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        grau_medio = np.sum(matrizes_a_hilbert[paciente, onda].flatten()) / np.shape(x_c)[1]
        grau_medio_a_hilbert[paciente, onda] = grau_medio

grau_medio_c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        grau_medio = np.sum(matrizes_c_hilbert[paciente, onda].flatten()) / np.shape(x_c)[1]
        grau_medio_c_hilbert[paciente, onda] = grau_medio

'''
Grau por linha da matriz
'''

grau_cada_linha_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas), np.shape(x_c)[1]))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        for j in range(np.shape(x_a)[1]):
            grau_cada_linha_a_pearson[paciente, onda, j] = sum(matrizes_a_pearson[paciente, onda, j])

grau_cada_linha_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas), np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        for j in range(np.shape(x_c)[1]):
            grau_cada_linha_c_pearson[paciente, onda, j] = sum(matrizes_c_pearson[paciente, onda, j])

grau_cada_linha_a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas), np.shape(x_c)[1]))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        for j in range(np.shape(x_a)[1]):
            grau_cada_linha_a_hilbert[paciente, onda, j] = sum(matrizes_a_hilbert[paciente, onda, j])

grau_cada_linha_c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas), np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        for j in range(np.shape(x_c)[1]):
            grau_cada_linha_c_hilbert[paciente, onda, j] = sum(matrizes_c_hilbert[paciente, onda, j])

'''
Variância da distribuição de grau
'''

variancia_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        variancia = 0
        for i in range(np.shape(x_a)[1]):
            variancia = variancia + (
                    (grau_cada_linha_a_pearson[paciente, onda, i] - grau_medio_a_pearson[paciente, onda]) ** 2) / \
                        np.shape(x_c)[1]
        variancia_a_pearson[paciente, onda] = variancia

variancia_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        variancia = 0
        for i in range(np.shape(x_c)[1]):
            variancia = variancia + (
                    (grau_cada_linha_c_pearson[paciente, onda, i] - grau_medio_c_pearson[paciente, onda]) ** 2) / \
                        np.shape(x_c)[1]
        variancia_c_pearson[paciente, onda] = variancia

variancia_a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        variancia = 0
        for i in range(np.shape(x_a)[1]):
            variancia = variancia + (
                    (grau_cada_linha_a_hilbert[paciente, onda, i] - grau_medio_a_hilbert[paciente, onda]) ** 2) / \
                        np.shape(x_c)[1]
        variancia_a_hilbert[paciente, onda] = variancia

variancia_c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        variancia = 0
        for i in range(np.shape(x_c)[1]):
            variancia = variancia + (
                    (grau_cada_linha_c_hilbert[paciente, onda, i] - grau_medio_c_hilbert[paciente, onda]) ** 2) / \
                        np.shape(x_c)[1]
        variancia_c_hilbert[paciente, onda] = variancia


def flatten_matrices(arr):
    """
    Recebe array no formato [paciente, onda, matriz_adj]
    Retorna array no formato [paciente, onda, features_flatten]
    """

    n_pac, n_ondas, _, _ = arr.shape
    return arr.reshape(n_pac, n_ondas, -1)


'''
Mutual Information
'''


def mutual_information(grupo1, grupo2):
    n_pac, n_ondas, n_feat = grupo1.shape
    mi = np.zeros((n_pac, n_ondas))
    for i in range(n_pac):
        for j in range(n_ondas):
            x = grupo1[i, j, :]
            y = grupo2[i, j, :]
            x_discr = np.digitize(x, np.histogram(x, bins=10)[1])
            y_discr = np.digitize(y, np.histogram(y, bins=10)[1])
            mi[i, j] = mutual_info_score(x_discr, y_discr)
    return mi


mutual_information_a = mutual_information(flatten_matrices(matrizes_a_pearson), flatten_matrices(matrizes_a_hilbert))
mutual_information_c = mutual_information(flatten_matrices(matrizes_c_pearson), flatten_matrices(matrizes_c_hilbert))


'''
Kurtosis
'''

kurtosis_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        kurtosis_a_pearson[paciente, onda] = kurtosis(grau_cada_linha_a_pearson[paciente, onda])

kurtosis_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        kurtosis_c_pearson[paciente, onda] = kurtosis(grau_cada_linha_c_pearson[paciente, onda])

kurtosis_a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        kurtosis_a_hilbert[paciente, onda] = kurtosis(grau_cada_linha_a_hilbert[paciente, onda])

kurtosis_c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        kurtosis_c_hilbert[paciente, onda] = kurtosis(grau_cada_linha_c_hilbert[paciente, onda])

'''
Entropia do grau
'''

entropia_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        entropia_a_pearson[paciente, onda] = entropy(grau_cada_linha_a_pearson[paciente, onda])

entropia_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        entropia_c_pearson[paciente, onda] = entropy(grau_cada_linha_c_pearson[paciente, onda])

entropia_a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        entropia_a_hilbert[paciente, onda] = entropy(grau_cada_linha_a_hilbert[paciente, onda])

entropia_c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        entropia_c_hilbert[paciente, onda] = entropy(grau_cada_linha_c_hilbert[paciente, onda])

'''
Coeficiente de aglomeração médio
'''

agrupamento_medio_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        agrupamento_medio_a_pearson[paciente, onda] = nx.average_clustering(
            nx.from_numpy_array(matrizes_a_pearson[paciente, onda]), weight="weight")

agrupamento_medio_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        agrupamento_medio_c_pearson[paciente, onda] = nx.average_clustering(
            nx.from_numpy_array(matrizes_c_pearson[paciente, onda]), weight="weight")

agrupamento_medio_a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        agrupamento_medio_a_hilbert[paciente, onda] = nx.average_clustering(
            nx.from_numpy_array(matrizes_a_hilbert[paciente, onda]), weight="weight")

agrupamento_medio_c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        agrupamento_medio_c_hilbert[paciente, onda] = nx.average_clustering(
            nx.from_numpy_array(matrizes_c_hilbert[paciente, onda]), weight="weight")

'''
Menor caminho médio
'''

menor_caminho_medio_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        menor_caminho_medio_a_pearson[paciente, onda] = nx.average_shortest_path_length(
            nx.from_numpy_array(matrizes_a_pearson[paciente, onda]), weight="weight")

menor_caminho_medio_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        menor_caminho_medio_c_pearson[paciente, onda] = nx.average_shortest_path_length(
            nx.from_numpy_array(matrizes_c_pearson[paciente, onda]), weight="weight")

menor_caminho_medio_a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        menor_caminho_medio_a_hilbert[paciente, onda] = nx.average_shortest_path_length(
            nx.from_numpy_array(matrizes_a_hilbert[paciente, onda]), weight="weight")

menor_caminho_medio_c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        menor_caminho_medio_c_hilbert[paciente, onda] = nx.average_shortest_path_length(
            nx.from_numpy_array(matrizes_c_hilbert[paciente, onda]), weight="weight")

'''
Assortatividade de grau
'''

assortatividade_grau_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        G = nx.from_numpy_array(matrizes_a_pearson[paciente, onda])
        assortatividade_grau_a_pearson[paciente, onda] = nx.degree_assortativity_coefficient(G, weight="weight")

assortatividade_grau_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        G = nx.from_numpy_array(matrizes_c_pearson[paciente, onda])
        assortatividade_grau_c_pearson[paciente, onda] = nx.degree_assortativity_coefficient(G, weight="weight")

assortatividade_grau_a_hilbert = np.zeros((np.shape(x_a)[0], len(ondas)))
for paciente in range(np.shape(x_a)[0]):
    for onda in range(len(ondas)):
        G = nx.from_numpy_array(matrizes_a_hilbert[paciente, onda])
        assortatividade_grau_a_hilbert[paciente, onda] = nx.degree_assortativity_coefficient(G, weight="weight")

assortatividade_grau_c_hilbert = np.zeros((np.shape(x_c)[0], len(ondas)))
for paciente in range(np.shape(x_c)[0]):
    for onda in range(len(ondas)):
        G = nx.from_numpy_array(matrizes_c_hilbert[paciente, onda])
        assortatividade_grau_c_hilbert[paciente, onda] = nx.degree_assortativity_coefficient(G, weight="weight")

metricas = ['Grau Médio', 'Var dist grau', 'Mutual Information', 'Kurtosis', 'Entropia do grau',
            'Coeficiente de aglomeração médio', 'Menor caminho médio']

resultados_metricas_a_pearson = np.array(
    [grau_medio_a_pearson, variancia_a_pearson, mutual_information_a, kurtosis_a_pearson, entropia_a_pearson,
     agrupamento_medio_a_pearson, menor_caminho_medio_a_pearson])
resultados_metricas_c_pearson = np.array(
    [grau_medio_c_pearson, variancia_c_pearson, mutual_information_c, kurtosis_c_pearson, entropia_c_pearson,
     agrupamento_medio_c_pearson, menor_caminho_medio_c_pearson])
resultados_metricas_a_hilbert = np.array(
    [grau_medio_a_hilbert, variancia_a_hilbert, mutual_information_a, kurtosis_a_hilbert, entropia_a_hilbert,
     agrupamento_medio_a_hilbert, menor_caminho_medio_a_hilbert])
resultados_metricas_c_hilbert = np.array(
    [grau_medio_c_hilbert, variancia_c_hilbert, mutual_information_c, kurtosis_c_hilbert, entropia_c_hilbert,
     agrupamento_medio_c_hilbert, menor_caminho_medio_c_hilbert])

np.save("Metricas/Metricas_autistas_pearson_i.npy", resultados_metricas_a_pearson)
np.save("Metricas/Metricas_controle_pearson_i.npy", resultados_metricas_c_pearson)
np.save("Metricas/Metricas_autistas_hilbert_i.npy", resultados_metricas_a_hilbert)
np.save("Metricas/Metricas_controle_hilbert_i.npy", resultados_metricas_c_hilbert)

# # Preparação dos dados
# a_data = resultados_metricas_a_pear.transpose(1, 0, 2).reshape(np.shape(x_a)[0], 35)
# c_data = resultados_metricas_c_pear.transpose(1, 0, 2).reshape(np.shape(x_c)[0], 35)
#
# X = np.vstack((a_data, c_data))
# y = np.array([0]*np.shape(x_a)[0] + [1]*np.shape(x_c)[0])
#
# # Aplicação do t-SNE
# tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, max_iter=1000)
# X_tsne = tsne.fit_transform(X)
#
# tsne1 = X_tsne[:, 0]
# tsne2 = X_tsne[:, 1]
#
# # Visualização
# plt.figure(figsize=(8, 6))
# plt.scatter(
#     tsne1[y == 0],
#     tsne2[y == 0],
#     marker='s',
#     color='tab:red',
#     label='Autism'
# )
# plt.scatter(
#     tsne1[y == 1],
#     tsne2[y == 1],
#     marker='o',
#     color='tab:blue',
#     label='Control'
# )
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.legend()
# plt.title('t-SNE Projection')
# plt.tight_layout()
# plt.savefig(f"TSNE.png", dpi=300)


# indices_tsne2_acima_50 = np.where(tsne2 > 50)[0]
# autistas_acima_50 = indices_tsne2_acima_50[indices_tsne2_acima_50 < np.shape(x_a)[0]]
# controles_acima_50 = indices_tsne2_acima_50[indices_tsne2_acima_50 >= np.shape(x_c)[0]]
#
# print("Pacientes autistas com t-SNE 2 > 50:", autistas_acima_50)
# print("Pacientes controle com t-SNE 2 > 50:", controles_acima_50 - np.shape(x_c)[0])
#
