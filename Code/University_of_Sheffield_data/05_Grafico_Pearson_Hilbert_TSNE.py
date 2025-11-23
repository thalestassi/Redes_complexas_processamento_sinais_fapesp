import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# x_a = np.load('x_autistas_inglaterra_50.npy') # autism data
# x_c = np.load('x_controle_inglaterra_50.npy')  # control data

x_a = np.load('x_autism_new_data.npy')  # autism data
x_c = np.load('x_control_new_data.npy')  # control data

ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ondas_latex = [r'Ondas $\delta$', r'Ondas $\theta$', r'Ondas $\alpha$', r'Ondas $\beta$', r'Ondas $\gamma$']

matrizes_a_hilbert = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_autistas_hilbert_i.npy')
matrizes_c_hilbert = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_controle_hilbert_i.npy')
matrizes_a_pearson = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_autista_pearson_i_1.npy')
matrizes_c_pearson = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_controle_pearson_i_1.npy')

resultados_metricas_a_pearson = np.load("Metricas/Metricas_autistas_pearson_i.npy")
resultados_metricas_c_pearson = np.load("Metricas/Metricas_controle_pearson_i.npy")
resultados_metricas_a_hilbert = np.load("Metricas/Metricas_autistas_hilbert_i.npy")
resultados_metricas_c_hilbert = np.load("Metricas/Metricas_controle_hilbert_i.npy")

X = np.vstack((resultados_metricas_a_pearson.transpose(1, 0, 2).reshape(np.shape(x_a)[0], 35),
               resultados_metricas_c_pearson.transpose(1, 0, 2).reshape(np.shape(x_c)[0], 35)
               ))
y = np.array([0] * np.shape(x_a)[0] + [1] * np.shape(x_c)[0])

# '''
# PCA
# '''
#
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
#
# pca1 = X_pca[:, 0]
# pca2 = X_pca[:, 1]
#
# plt.figure(figsize=(10, 5))
# plt.scatter(
#     pca1[y == 0],
#     pca2[y == 0],
#     marker='s',
#     color='tab:orange',
#     label='Autism'
# )
# plt.scatter(
#     pca1[y == 1],
#     pca2[y == 1],
#     marker='o',
#     color='tab:blue',
#     label='Control'
# )
# plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
# plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
#
# plt.legend()
# plt.title('PCA')
# plt.tight_layout()
# plt.show()
# # plt.savefig(f"PCA.png", dpi=300)
#
'''
t-SNE
'''

tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, max_iter=1000)
X_tsne = tsne.fit_transform(X)
tsne1 = X_tsne[:, 0]
tsne2 = X_tsne[:, 1]
#
# plt.figure(figsize=(10, 5))
# plt.scatter(
#     tsne1[y == 0],
#     tsne2[y == 0],
#     marker='s',
#     color='tab:orange',
#     label='Autism'
# )
# plt.scatter(
#     tsne1[y == 1],
#     tsne2[y == 1],
#     marker='o',
#     color='tab:blue',
#     label='Control'
# )
# ax = plt.gca()
# ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.legend()
# plt.title('t-SNE Projection')
# plt.tight_layout()
# plt.show()
# # plt.savefig(f"t_SNE.png", dpi=300)
#
# indices_tsne2_cluster = np.where(tsne2 < -50)[0]
# a_indices_cluster = indices_tsne2_cluster[indices_tsne2_cluster < np.shape(x_a)[0]]
# c_indices_cluster = indices_tsne2_cluster[indices_tsne2_cluster >= np.shape(x_c)[0]] - np.shape(x_c)[0]
#
# print("Pacientes autistas com t-SNE 2 < -50:", a_indices_cluster)
# print("Pacientes controle com t-SNE 2 < -50:", c_indices_cluster)
#
# matrizes_a_hilbert_tsne = np.zeros((len(a_indices_cluster), len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
# contador = 0
# for i in range(np.shape(matrizes_a_hilbert)[0]):
#     if i in a_indices_cluster:
#         matrizes_a_hilbert_tsne[contador] = matrizes_a_hilbert[i]
#         contador = contador + 1
#
#
# matrizes_c_hilbert_tsne = np.zeros((len(c_indices_cluster), len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
# contador = 0
#
# for i in range(np.shape(matrizes_c_hilbert)[0]):
#     if i in c_indices_cluster:
#         matrizes_c_hilbert_tsne[contador] = matrizes_c_hilbert[i]
#         contador = contador + 1
#
# matrizes_a_pearson_tsne = np.zeros((len(a_indices_cluster), len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
# contador = 0
#
# for i in range(np.shape(matrizes_a_pearson)[0]):
#     if i in a_indices_cluster:
#         matrizes_a_pearson_tsne[contador] = matrizes_a_pearson[i]
#         contador = contador + 1
#
# matrizes_c_pearson_tsne = np.zeros((len(c_indices_cluster), len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
# contador = 0
#
# for i in range(np.shape(matrizes_c_pearson)[0]):
#     if i in c_indices_cluster:
#         matrizes_c_pearson_tsne[contador] = matrizes_c_pearson[i]
#         contador = contador + 1
#
#
#
# '''
# a_pearson/c_pearson: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]
#
# Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
# '''
#
# fig, axs = plt.subplots(2, 5, figsize=(15, 5), sharex=True, sharey=True)
# for i in range(5):
#     x_a_flat = matrizes_a_pearson_tsne[:, i].flatten()
#     y_a_flat = matrizes_a_hilbert_tsne[:, i].flatten()
#     axs[0, i].scatter(x_a_flat, y_a_flat, alpha=0.05, s=10, color='orange')
#     axs[0, i].set_title(f'{ondas_latex[i]}')
#     x_c_flat = matrizes_c_pearson_tsne[:, i].flatten()
#     y_c_flat = matrizes_c_hilbert_tsne[:, i].flatten()
#     axs[1, i].scatter(x_c_flat, y_c_flat, alpha=0.05, s=10)
#
# for ax in axs[1, :]:
#     ax.set_xlabel(r'Correlação de Pearson ($\rho_{pq}$)')
# for ax in axs[:, 0]:
#     ax.set_ylabel(r'Sincronização de Fase (Hilbert, $r_{pq}$)')
# for ax in axs.flat:
#     ax.tick_params(axis='y', labelleft=True)
#
# plt.tight_layout()
# # plt.savefig(f"tSNE_Pearson_Hilbert.png", dpi=300)
# plt.show()

'''
Pacientes fora do cluster
'''

indices_tsne2_cluster_fora = np.where(tsne2 > -50)[0]
a_indices_cluster_fora = indices_tsne2_cluster_fora[indices_tsne2_cluster_fora < np.shape(x_a)[0]]
c_indices_cluster_fora = indices_tsne2_cluster_fora[indices_tsne2_cluster_fora >= np.shape(x_c)[0]] - np.shape(x_c)[0]

print("Pacientes autistas com t-SNE 2 > -50:", a_indices_cluster_fora)
print("Pacientes controle com t-SNE 2 > -50:", c_indices_cluster_fora)

matrizes_a_hilbert_tsne_fora = np.zeros((len(a_indices_cluster_fora), len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
contador = 0
for i in range(np.shape(matrizes_a_hilbert)[0]):
    if i in a_indices_cluster_fora:
        matrizes_a_hilbert_tsne_fora[contador] = matrizes_a_hilbert[i]
        contador = contador + 1


matrizes_c_hilbert_tsne_fora = np.zeros((len(c_indices_cluster_fora), len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
contador = 0

for i in range(np.shape(matrizes_c_hilbert)[0]):
    if i in c_indices_cluster_fora:
        matrizes_c_hilbert_tsne_fora[contador] = matrizes_c_hilbert[i]
        contador = contador + 1

matrizes_a_pearson_tsne_fora = np.zeros((len(a_indices_cluster_fora), len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
contador = 0

for i in range(np.shape(matrizes_a_pearson)[0]):
    if i in a_indices_cluster_fora:
        matrizes_a_pearson_tsne_fora[contador] = matrizes_a_pearson[i]
        contador = contador + 1

matrizes_c_pearson_tsne_fora = np.zeros((len(c_indices_cluster_fora), len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
contador = 0

for i in range(np.shape(matrizes_c_pearson)[0]):
    if i in c_indices_cluster_fora:
        matrizes_c_pearson_tsne_fora[contador] = matrizes_c_pearson[i]
        contador = contador + 1



'''
a_pearson/c_pearson: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''

fig, axs = plt.subplots(2, 5, figsize=(15, 5), sharex=True, sharey=True)
for i in range(5):
    x_a_flat = matrizes_a_pearson_tsne_fora[:, i].flatten()
    y_a_flat = matrizes_a_hilbert_tsne_fora[:, i].flatten()
    axs[0, i].scatter(x_a_flat, y_a_flat, alpha=0.05, s=10, color='orange')
    axs[0, i].set_title(f'{ondas_latex[i]}')
    x_c_flat = matrizes_c_pearson_tsne_fora[:, i].flatten()
    y_c_flat = matrizes_c_hilbert_tsne_fora[:, i].flatten()
    axs[1, i].scatter(x_c_flat, y_c_flat, alpha=0.05, s=10)
for ax in axs[1, :]:
    ax.set_xlabel(r'Correlação de Pearson ($\rho_{pq}$)')
for ax in axs[:, 0]:
    ax.set_ylabel(r'Sincronização de Fase (Hilbert, $r_{pq}$)')
for ax in axs.flat:
    ax.tick_params(axis='y', labelleft=True)

plt.tight_layout()
# plt.savefig(f"tSNE_Pearson_Hilbert.png", dpi=300)
plt.show()