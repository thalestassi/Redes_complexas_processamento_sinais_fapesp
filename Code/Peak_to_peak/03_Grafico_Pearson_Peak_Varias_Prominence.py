import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import combinations

xxx = np.load('formatted_data.npy')
x_a = xxx[0]  # autism data
x_c = xxx[1]  # control data


ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ondas_latex = [r'Ondas $\delta$', r'Ondas $\theta$', r'Ondas $\alpha$', r'Ondas $\beta$', r'Ondas $\gamma$']

prominencias = np.round(np.arange(0.00, 0.7, 0.1), 2)

matrizes_a_pearson = np.zeros((np.shape(x_a)[0], len(ondas), np.shape(x_a)[1], np.shape(x_a)[1]))
for paciente in range(np.shape(x_a)[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):
        matriz_a = np.zeros((np.shape(x_a)[1], np.shape(x_a)[1]))
        for i, j in combinations(range(np.shape(x_a)[1]), 2):
            r, _ = pearsonr(x_a[paciente, i, k, :], x_a[paciente, j, k, :])
            matriz_a[i, j] = r
            matriz_a[j, i] = r

        matrizes_a_pearson[paciente, idx] = matriz_a

matrizes_c_pearson = np.zeros((np.shape(x_c)[0], len(ondas), np.shape(x_c)[1], np.shape(x_c)[1]))
for paciente in range(np.shape(x_c)[0]):
    for idx, k in enumerate(range(2, len(ondas) + 2)):
        matriz_c = np.zeros((np.shape(x_c)[1], np.shape(x_c)[1]))
        for i, j in combinations(range(np.shape(x_c)[1]), 2):
            r, _ = pearsonr(x_c[paciente, i, k, :], x_c[paciente, j, k, :])
            matriz_c[i, j] = r
            matriz_c[j, i] = r

        matrizes_c_pearson[paciente, idx] = matriz_c

'''
a_pearson/c_pearson: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''


for p_idx, p in enumerate(prominencias):
    print(f"Analisando prominência: {p}")
    prom_str = f"{p:.2f}".replace('.', '_')
    matrizes_a_peak = np.load(f"Matrizes_Adjacencia_Prominence/Matriz_autista_peak_prom_{prom_str}.npy")
    matrizes_c_peak = np.load(f"Matrizes_Adjacencia_Prominence/Matriz_controle_peak_prom_{prom_str}.npy")
    fig, axs = plt.subplots(2, len(ondas), figsize=(15, 5), sharex=True, sharey=True)
    for i in range(len(ondas)):
        x_a_flat = matrizes_a_pearson[:, i].flatten()
        y_a_flat = matrizes_a_peak[:, i].flatten()
        axs[0, i].scatter(x_a_flat, y_a_flat, alpha=0.05, s=10, color='orange')
        axs[0, i].set_title(f'{ondas_latex[i]}')
        x_c_flat = matrizes_c_pearson[:, i].flatten()
        y_c_flat = matrizes_c_peak[:, i].flatten()
        axs[1, i].scatter(x_c_flat, y_c_flat, alpha=0.05, s=10)

    for ax in axs[1, :]:
        ax.set_xlabel(r'Correlação de Pearson ($\rho_{pq}$)')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'Sincronização de Fase (Peak-to-Peak, $phi_{pq}$)')
    for ax in axs.flat:
        ax.tick_params(axis='y', labelleft=True)

    plt.tight_layout()
    plt.savefig(f'Graficos_Pearson_Peak/Pearson_Peak_{p:.2f}.png', dpi=300)
