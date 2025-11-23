import numpy as np
import matplotlib.pyplot as plt


'''
files.npy: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''
ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ondas_latex = [r'Ondas $\delta$', r'Ondas $\theta$', r'Ondas $\alpha$', r'Ondas $\beta$', r'Ondas $\gamma$']

resultados_metricas_a_pearson = np.load("Metricas/Metricas_autistas_pearson_i.npy")
resultados_metricas_c_pearson = np.load("Metricas/Metricas_controle_pearson_i.npy")
resultados_metricas_a_hilbert = np.load("Metricas/Metricas_autistas_hilbert_i.npy")
resultados_metricas_c_hilbert = np.load("Metricas/Metricas_controle_hilbert_i.npy")
metricas = ['Grau Médio ', 'Var dist grau', 'Mutual Information', 'Kurtosis', 'Entropia do grau',
            'Coeficiente de aglomeração médio', 'Menor caminho médio']

metricas = [
    r'Grau médio $\langle s \rangle$',
    r'Variância da distribuição de grau ($\text{Var}[s]$)',
    r'Mutual Information ($MI$)',
    r'Kurtosis ($\text{Kurt}[s]$)',
    r'Entropia do grau ($H$)',
    r'Coeficiente de aglomeração médio $\langle C^{(w)} \rangle$',
    r'Menor caminho médio ($L$)'
]
fig, axs = plt.subplots(len(ondas), len(metricas), figsize=(22, 12))#
for j in range(len(ondas)):
    for i in range(len(metricas)):
        ax = axs[j, i]
        bp = ax.boxplot([resultados_metricas_a_pearson[i, :, j], resultados_metricas_c_pearson[i, :, j]],
                   positions=[1, 2],
                   widths=0.2)
        bp['medians'][0].set_color('orange')
        bp['medians'][1].set_color('blue')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Autism', 'Control'], fontsize=8)
        ax.set_title(f'{metricas[i]}', fontsize=8)
        if i == 0:
            ax.set_ylabel(f'{ondas_latex[j]}', fontsize=10)

fig.tight_layout()
# plt.savefig(f"Box_Plot_Metricas_Pearson/Métricas_Pearson_Box.png", dpi=300)
plt.show()
