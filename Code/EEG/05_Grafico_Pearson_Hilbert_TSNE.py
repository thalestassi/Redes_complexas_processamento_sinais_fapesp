import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

xxx = np.load('formatted_data.npy')
x_a = xxx[0]  # autism data
x_c = xxx[1]  # control data

ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ondas_latex = [r'Ondas $\delta$', r'Ondas $\theta$', r'Ondas $\alpha$', r'Ondas $\beta$', r'Ondas $\gamma$']

matrizes_a_hilbert = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_autistas_hilbert.npy')
matrizes_c_hilbert = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_controle_hilbert.npy')

resultados_metricas_a_pearson = np.load("Metricas/Metricas_autistas_pearson.npy")
resultados_metricas_c_pearson = np.load("Metricas/Metricas_controle_pearson.npy")
resultados_metricas_a_hilbert = np.load("Metricas/Metricas_autistas_hilbert.npy")
resultados_metricas_c_hilbert = np.load("Metricas/Metricas_controle_hilbert.npy")

X = np.vstack((resultados_metricas_a_pearson.transpose(1, 0, 2).reshape(np.shape(x_a)[0], 35),
               resultados_metricas_c_pearson.transpose(1, 0, 2).reshape(np.shape(x_c)[0], 35)
               ))
y = np.array([0] * np.shape(x_a)[0] + [1] * np.shape(x_c)[0])

'''
PCA
'''

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca1 = X_pca[:, 0]
pca2 = X_pca[:, 1]

plt.figure(figsize=(10, 5))
plt.scatter(
    pca1[y == 0],
    pca2[y == 0],
    marker='s',
    color='tab:orange',
    label='Autism'
)
plt.scatter(
    pca1[y == 1],
    pca2[y == 1],
    marker='o',
    color='tab:blue',
    label='Control'
)
plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')

plt.legend()
plt.title('PCA')
plt.tight_layout()
# plt.savefig(f"PCA.png", dpi=300)

'''
t-SNE
'''

tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, max_iter=1000, random_state=0)
X_tsne = tsne.fit_transform(X)
tsne1 = X_tsne[:, 0]
tsne2 = X_tsne[:, 1]

plt.figure(figsize=(10, 5))
plt.scatter(
    tsne1[y == 0],
    tsne2[y == 0],
    marker='s',
    color='tab:orange',
    label='Autism'
)
plt.scatter(
    tsne1[y == 1],
    tsne2[y == 1],
    marker='o',
    color='tab:blue',
    label='Control'
)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.title('t-SNE Projection')
plt.tight_layout()
# plt.savefig(f"t_SNE.png", dpi=300)

indices_tsne1_cluster = np.where(tsne1 > 150)[0]
a_indices_cluster = indices_tsne1_cluster[indices_tsne1_cluster < np.shape(x_a)[0]]
c_indices_cluster = indices_tsne1_cluster[indices_tsne1_cluster >= np.shape(x_c)[0]] - np.shape(x_c)[0]


matrizes_a_hilbert_tsne = np.zeros((len(a_indices_cluster), len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
contador = 0
for i in range(np.shape(matrizes_a_hilbert)[0]):
    if i in a_indices_cluster:
        matrizes_a_hilbert_tsne[contador] = matrizes_a_hilbert[i]
        contador = contador + 1


matrizes_c_hilbert_tsne = np.zeros((len(c_indices_cluster), len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
contador = 0

for i in range(np.shape(matrizes_c_hilbert)[0]):
    if i in c_indices_cluster:
        matrizes_c_hilbert_tsne[contador] = matrizes_c_hilbert[i]
        contador = contador + 1


matrizes_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
for paciente in range(np.shape(x_a)[0]):
    for idx, k in enumerate(range(2, 7)):
        matriz_a = np.zeros((np.shape(x_a)[1], np.shape(x_a)[1]))
        for i, j in combinations(range(np.shape(x_a)[1]), 2):
            r, _ = pearsonr(x_a[paciente, i, k, :], x_a[paciente, j, k, :])
            matriz_a[i, j] = r
            matriz_a[j, i] = r

        matrizes_a_pearson[paciente, idx] = matriz_a


matrizes_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    for idx, k in enumerate(range(2, 7)):
        matriz_c = np.zeros((np.shape(x_c)[1], np.shape(x_c)[1]))
        for i, j in combinations(range(np.shape(x_c)[1]), 2):
            r, _ = pearsonr(x_c[paciente, i, k, :], x_c[paciente, j, k, :])
            matriz_c[i, j] = r
            matriz_c[j, i] = r

        matrizes_c_pearson[paciente, idx] = matriz_c

matrizes_a_pearson_tsne = np.zeros((len(a_indices_cluster), len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
contador = 0

for i in range(np.shape(matrizes_a_pearson)[0]):
    if i in a_indices_cluster:
        matrizes_a_pearson_tsne[contador] = matrizes_a_pearson[i]
        contador = contador + 1

matrizes_c_pearson_tsne = np.zeros((len(c_indices_cluster), len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
contador = 0

for i in range(np.shape(matrizes_c_pearson)[0]):
    if i in c_indices_cluster:
        matrizes_c_pearson_tsne[contador] = matrizes_c_pearson[i]
        contador = contador + 1



'''
a_pearson/c_pearson: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''

fig, axs = plt.subplots(2, 5, figsize=(15, 5), sharex=True, sharey=True)
for i in range(5):
    x_a_flat = matrizes_a_pearson_tsne[:, i].flatten()
    y_a_flat = matrizes_a_hilbert_tsne[:, i].flatten()
    axs[0, i].scatter(x_a_flat, y_a_flat, alpha=0.05, s=10, color='orange')
    axs[0, i].set_title(f'{ondas_latex[i]}')
    x_c_flat = matrizes_c_pearson_tsne[:, i].flatten()
    y_c_flat = matrizes_c_hilbert_tsne[:, i].flatten()
    axs[1, i].scatter(x_c_flat, y_c_flat, alpha=0.05, s=10)

for ax in axs[1, :]:
    ax.set_xlabel(r'Correlação de Pearson ($\rho_{pq}$)')
for ax in axs[:, 0]:
    ax.set_ylabel(r'Sincronização de Fase (Hilbert, $r_{pq}$)')
for ax in axs.flat:
    ax.tick_params(axis='y', labelleft=True)

plt.tight_layout()
# plt.savefig(f"Gráficos_Pearson_Fase_Hilbert/tSNE_Pearson_Hilbert.png", dpi=300)
plt.show()


