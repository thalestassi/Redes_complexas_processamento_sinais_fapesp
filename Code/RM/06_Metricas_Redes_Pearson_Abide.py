import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import skew, kurtosis, entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

n_pacientes_geral = 50
tipo_analise = 'cpac'
qualidade = True
regressao_global = True

matrizes_a_pear = np.load(f"Matrizes_Adjacencia_Abide/Matrizes_adjacencia_a_pearson_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy")
matrizes_c_pear = np.load(f"Matrizes_Adjacencia_Abide/Matrizes_adjacencia_c_pearson_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy")


'''
Grau Médio
'''

grau_medio_a_pear = np.zeros(np.shape(matrizes_a_pear)[0])
for paciente in range(np.shape(matrizes_a_pear)[0]):
    grau_medio = np.sum(matrizes_a_pear[paciente].flatten()) / np.shape(matrizes_a_pear)[1]
    grau_medio_a_pear[paciente] = grau_medio

grau_medio_c_pear = np.zeros(np.shape(matrizes_c_pear)[0])
for paciente in range(np.shape(matrizes_c_pear)[0]):
    grau_medio = np.sum(matrizes_c_pear[paciente].flatten()) / np.shape(matrizes_c_pear)[1]
    grau_medio_c_pear[paciente] = grau_medio

grau_cada_linha_a_pear = np.zeros((matrizes_a_pear.shape[0], matrizes_a_pear.shape[1]))

for paciente in range(matrizes_a_pear.shape[0]):
    grau_cada_linha_a_pear[paciente, :] = np.sum(matrizes_a_pear[paciente], axis=1)


grau_cada_linha_c_pear = np.zeros((matrizes_c_pear.shape[0], matrizes_c_pear.shape[1]))

for paciente in range(matrizes_c_pear.shape[0]):
    grau_cada_linha_c_pear[paciente, :] = np.sum(matrizes_c_pear[paciente], axis=1)


'''
Variância da distribuição de grau
'''

variancia_a_pear = np.zeros(np.shape(matrizes_a_pear)[0])
for paciente in range(np.shape(matrizes_a_pear)[0]):
    variancia = 0
    for i in range(np.shape(matrizes_a_pear)[1]):
        variancia = variancia + (
                    (grau_cada_linha_a_pear[paciente, i] - grau_medio_a_pear[paciente]) ** 2) / np.shape(matrizes_a_pear)[1]
    variancia_a_pear[paciente] = variancia

variancia_c_pear = np.zeros(np.shape(matrizes_c_pear)[0])
for paciente in range(np.shape(matrizes_c_pear)[0]):
    variancia = 0
    for i in range(np.shape(matrizes_c_pear)[1]):
        variancia = variancia + (
                    (grau_cada_linha_c_pear[paciente, i] - grau_medio_c_pear[paciente]) ** 2) / np.shape(matrizes_c_pear)[1]
    variancia_c_pear[paciente] = variancia

'''
Skewness
'''

skewness_a_pear = np.zeros(np.shape(matrizes_a_pear)[0])
for paciente in range(np.shape(matrizes_a_pear)[0]):
    skewness_a_pear[paciente] = skew(grau_cada_linha_a_pear[paciente])

skewness_c_pear = np.zeros(np.shape(matrizes_c_pear)[0])
for paciente in range(np.shape(matrizes_c_pear)[0]):
    skewness_c_pear[paciente] = skew(grau_cada_linha_c_pear[paciente])

'''
Kurtosis
'''

kurtosis_a_pear = np.zeros(np.shape(matrizes_a_pear)[0])
for paciente in range(np.shape(matrizes_a_pear)[0]):
    kurtosis_a_pear[paciente] = kurtosis(grau_cada_linha_a_pear[paciente])

kurtosis_c_pear = np.zeros(np.shape(matrizes_c_pear)[0])
for paciente in range(np.shape(matrizes_c_pear)[0]):
    kurtosis_c_pear[paciente] = kurtosis(grau_cada_linha_c_pear[paciente])

'''
Entropia do grau
'''

entropia_a_pear = np.zeros(np.shape(matrizes_a_pear)[0])
for paciente in range(np.shape(matrizes_a_pear)[0]):
    entropia_a_pear[paciente] = entropy(grau_cada_linha_a_pear[paciente])

entropia_c_pear = np.zeros(np.shape(matrizes_c_pear)[0])
for paciente in range(np.shape(matrizes_c_pear)[0]):
    entropia_c_pear[paciente] = entropy(grau_cada_linha_c_pear[paciente])
'''
Coeficiente de aglomeração médio
'''

agrupamento_medio_a_pear = np.zeros(np.shape(matrizes_a_pear)[0])
for paciente in range(np.shape(matrizes_a_pear)[0]):
    agrupamento_medio_a_pear[paciente] = nx.average_clustering(
        nx.from_numpy_array(matrizes_a_pear[paciente]), weight="weight")

agrupamento_medio_c_pear = np.zeros(np.shape(matrizes_c_pear)[0])
for paciente in range(np.shape(matrizes_c_pear)[0]):
    agrupamento_medio_c_pear[paciente] = nx.average_clustering(
        nx.from_numpy_array(matrizes_c_pear[paciente]), weight="weight")

'''
Menor caminho médio
'''

menor_caminho_medio_a_pear = np.zeros(np.shape(matrizes_a_pear)[0])
for paciente in range(np.shape(matrizes_a_pear)[0]):
    G = nx.from_numpy_array(matrizes_a_pear[paciente])
    if nx.is_connected(G):
        menor_caminho_medio_a_pear[paciente] = nx.average_shortest_path_length(G, weight="weight")
    else:
        componentes = sorted(nx.connected_components(G), key=len, reverse=True)
        G_maior = G.subgraph(componentes[0])
        menor_caminho_medio_a_pear[paciente] = nx.average_shortest_path_length(G_maior, weight="weight")


menor_caminho_medio_c_pear = np.zeros(np.shape(matrizes_c_pear)[0])
for paciente in range(np.shape(matrizes_c_pear)[0]):
    G = nx.from_numpy_array(matrizes_c_pear[paciente])
    if nx.is_connected(G):
        menor_caminho_medio_c_pear[paciente] = nx.average_shortest_path_length(G, weight="weight")
    else:
        componentes = sorted(nx.connected_components(G), key=len, reverse=True)
        G_maior = G.subgraph(componentes[0])
        menor_caminho_medio_c_pear[paciente] = nx.average_shortest_path_length(G_maior, weight="weight")

'''
Assortatividade de grau
'''

assortatividade_grau_a_pear = np.zeros(np.shape(matrizes_a_pear)[0])
for paciente in range(np.shape(matrizes_a_pear)[0]):
    G = nx.from_numpy_array(matrizes_a_pear[paciente])
    assortatividade_grau_a_pear[paciente] = nx.degree_assortativity_coefficient(G,  weight="weight")

assortatividade_grau_c_pear = np.zeros(np.shape(matrizes_c_pear)[0])
for paciente in range(np.shape(matrizes_c_pear)[0]):
    G = nx.from_numpy_array(matrizes_c_pear[paciente])
    assortatividade_grau_c_pear[paciente] = nx.degree_assortativity_coefficient(G,  weight="weight")



metricas = ['Grau Médio', 'Var dist grau', 'Skewness', 'Kurtosis', 'Entropia do grau', 'Coeficiente de aglomeração médio', 'Menor caminho médio']
resultados_metricas_a_pear = np.array([grau_medio_a_pear, variancia_a_pear, skewness_a_pear, kurtosis_a_pear, entropia_a_pear, agrupamento_medio_a_pear, menor_caminho_medio_a_pear])
resultados_metricas_c_pear = np.array([grau_medio_c_pear, variancia_c_pear, skewness_c_pear, kurtosis_c_pear, entropia_c_pear, agrupamento_medio_c_pear, menor_caminho_medio_c_pear])




# Preparação dos dados
a_data = resultados_metricas_a_pear.T
c_data = resultados_metricas_c_pear.T

X = np.vstack((a_data, c_data))

nan_por_coluna = np.isnan(X).sum(axis=0)

for i, total_nans in enumerate(nan_por_coluna):
    if total_nans > 0:
        print(f"Métrica na coluna {i} contém {total_nans} NaN(s)")


y = np.array([0]*np.shape(matrizes_a_pear)[0] + [1]*(np.shape(matrizes_c_pear)[0]))

# Aplicação do t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=100, max_iter=1000)
X_tsne = tsne.fit_transform(X)

tsne1 = X_tsne[:, 0]
tsne2 = X_tsne[:, 1]

# Visualização
plt.figure(figsize=(8, 6))
plt.scatter(
    tsne1[y == 0],
    tsne2[y == 0],
    marker='s',
    color='tab:red',
    label='Autism'
)
plt.scatter(
    tsne1[y == 1],
    tsne2[y == 1],
    marker='o',
    color='tab:blue',
    label='Control'
)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.title('t-SNE Projection')
plt.tight_layout()
plt.show()

# indices_tsne1_acima_100 = np.where(tsne1 > 100)[0]
# autistas_acima_100 = indices_tsne1_acima_100[indices_tsne1_acima_100 < 24]
# controles_acima_100 = indices_tsne1_acima_100[indices_tsne1_acima_100 >= 24]
#
# print("Pacientes autistas com t-SNE 1 > 100:", autistas_acima_100)
# print("Pacientes controle com t-SNE 1 > 100:", controles_acima_100 - 24)



#
# for j in range(5):
#     for i in range(7):
#         plt.subplot(3, 3, i+1)
#         plt.boxplot([resultados_metricas_a_pear[i,:,j], resultados_metricas_c_pear[i,:,j]],
#                  positions=[1, 2],
#                  widths=0.2,
#                  )
#         plt.xticks([1, 2], ['Autismo', 'Controle'])  # Legenda embaixo de cada boxplot
#         plt.ylabel(f'{metricas[i]}', fontsize=8)
#
#     plt.suptitle(f'Métricas Pearson - Ondas {ondas[j]}', fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f"Box_Plot_Metricas_Pearson/Métricas_Pearson_{ondas[j]}.png", dpi=300)
#     plt.close()
#
# fig, axs = plt.subplots(5, 7, figsize=(20, 12))  # 5 linhas (ondas), 7 colunas (métricas)
#
# for j in range(5):  # para cada onda
#     for i in range(7):  # para cada métrica
#         ax = axs[j, i]
#         ax.boxplot([resultados_metricas_a_pear[i, :, j], resultados_metricas_c_pear[i, :, j]],
#                    positions=[1, 2],
#                    widths=0.2)
#         ax.set_xticks([1, 2])
#         ax.set_xticklabels(['Autismo', 'Controle'], fontsize=6)
#         ax.set_title(metricas[i], fontsize=8)
#         if i == 0:
#             ax.set_ylabel(f'{ondas[j]}', fontsize=10)  # título da linha com o nome da onda
#
# plt.suptitle('Boxplots das Métricas - Pearson - Autismo vs Controle (5 Ondas)', fontsize=14)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # espaço para o título
# plt.savefig(f"Box_Plot_Metricas_Pearson/Métricas_Pearson_combinadas.png", dpi=300)
# plt.show()

# a_data = resultados_metricas_a_pear.transpose(1, 0, 2).reshape(24, 35)
# c_data = resultados_metricas_c_pear.transpose(1, 0, 2).reshape(24, 35)
#
# X = np.vstack((a_data, c_data))
# y = np.array([0]*24 + [1]*24)
#
#
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
#
# pca1 = X_pca[:, 0]
# pca2 = X_pca[:, 1]
#
#
# plt.figure(figsize=(8, 6))
# plt.scatter(
#     pca1[y == 0],
#     pca2[y == 0],
#     marker='s',
#     color='tab:red',
#     label='Autism'
# )
# plt.scatter(
#     pca1[y == 1],
#     pca2[y == 1],
#     marker='o',
#     color='tab:blue',
#     label='Control'
# )
# plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
# plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
#
# plt.legend()
# plt.title('PCA')
# plt.show()

# print(pca.components_[0])