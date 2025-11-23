import numpy as np
import matplotlib.pyplot as plt


ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ondas_latex = [r'Ondas $\delta$', r'Ondas $\theta$', r'Ondas $\alpha$', r'Ondas $\beta$', r'Ondas $\gamma$']

matrizes_a_hilbert = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_autistas_hilbert_i.npy')
matrizes_c_hilbert = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_controle_hilbert_i.npy')
matrizes_a_pearson = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_autista_pearson_i_1.npy')
matrizes_c_pearson = np.load('Matrizes_Adjacencia/Matrizes_adjacencia_controle_pearson_i_1.npy')

fig, axs = plt.subplots(2, len(ondas), figsize=(15, 5), sharex=True, sharey=True)
for i in range(len(ondas)):
    x_a_flat = matrizes_a_pearson[:, i].flatten()
    y_a_flat = matrizes_a_hilbert[:, i].flatten()
    axs[0, i].scatter(x_a_flat, y_a_flat, alpha=0.05, s=10, color='orange')
    axs[0, i].set_title(f'{ondas_latex[i]}')
    x_c_flat = matrizes_c_pearson[:, i].flatten()
    y_c_flat = matrizes_c_hilbert[:, i].flatten()
    axs[1, i].scatter(x_c_flat, y_c_flat, alpha=0.05, s=10)

for ax in axs[1, :]:
    ax.set_xlabel(r'Correlação de Pearson ($\rho_{pq}$)')
for ax in axs[:, 0]:
    ax.set_ylabel(r'Sincronização de Fase (Hilbert, $r_{pq}$)')
for ax in axs.flat:
    ax.tick_params(axis='y', labelleft=True)

plt.tight_layout()
plt.savefig(f"Todos_Pearson_Hilbert.png", dpi=300)
plt.show()


