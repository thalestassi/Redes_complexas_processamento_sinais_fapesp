import numpy as np
import matplotlib.pyplot as plt


'''
files.npy: [Patient ID, Wave ID, Matriz de adjacência (eletrodos i e j)]

Wave ID: [0:delta, 1:theta, 2:alpha, 3:beta, 4:gamma]
'''
ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ondas_latex = [r'Ondas $\delta$', r'Ondas $\theta$', r'Ondas $\alpha$', r'Ondas $\beta$', r'Ondas $\gamma$']

resultados_metricas_a_pearson = np.load("Metricas/Metricas_autistas_pearson.npy")
resultados_metricas_c_pearson = np.load("Metricas/Metricas_controle_pearson.npy")
resultados_metricas_a_hilbert = np.load("Metricas/Metricas_autistas_hilbert.npy")
resultados_metricas_c_hilbert = np.load("Metricas/Metricas_controle_hilbert.npy")



metricas = [
    r'Grau médio $\langle s \rangle$',
    r'Variância da distribuição de grau ($\text{Var}[s]$)',
    r'Mutual Information ($MI$)',
    r'Kurtosis ($\text{Kurt}[s]$)',
    r'Entropia de distribuição de grau ($H$)',
    r'Coeficiente de aglomeração médio $\langle C^{(w)} \rangle$',
    r'Menor caminho médio ($L$)'
]

metricas_h = [
    r'Grau médio $\langle s \rangle$',
    r'Variância da distribuição de grau ($\text{Var}[s]$)',
    r'Kurtosis ($\text{Kurt}[s]$)',
    r'Entropia de distribuição de grau ($H$)',
    r'Coeficiente de aglomeração médio $\langle C^{(w)} \rangle$',
    r'Menor caminho médio ($L$)'
]

indices_para_plotar = [0, 1, 3, 4, 5, 6]

metricas_labels = [metricas[i] for i in indices_para_plotar]

fig, axs = plt.subplots(len(ondas), len(indices_para_plotar), figsize=(22, 12))

for j in range(len(ondas)):
    for k, idx_dado in enumerate(indices_para_plotar):
        ax = axs[j, k]
        dados_autismo = resultados_metricas_a_pearson[idx_dado, :, j]
        dados_controle = resultados_metricas_c_pearson[idx_dado, :, j]

        bp = ax.boxplot([dados_autismo, dados_controle],
                        positions=[1, 2],
                        widths=0.2)
        bp['medians'][0].set_color('orange')
        bp['medians'][1].set_color('blue')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Autismo', 'Controle'], fontsize=8)
        ax.set_title(f'{metricas_labels[k]}', fontsize=8)
        if k == 0:
            ax.set_ylabel(f'{ondas_latex[j]}', fontsize=10)

fig.tight_layout()
plt.savefig(f"Metricas_Pearson_Box.png", dpi=300)

fig, axs = plt.subplots(len(ondas), len(metricas_h), figsize=(22, 12))
for j in range(len(ondas)):
    for i in range(len(metricas_h)):
        ax = axs[j, i]
        bp = ax.boxplot([resultados_metricas_a_hilbert[i, :, j], resultados_metricas_c_hilbert[i, :, j]],
                   positions=[1, 2],
                   widths=0.2)
        bp['medians'][0].set_color('orange')
        bp['medians'][1].set_color('blue')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Autismo', 'Controle'], fontsize=8)
        ax.set_title(f'{metricas_h[i]}', fontsize=8)
        if i == 0:
            ax.set_ylabel(f'{ondas_latex[j]}', fontsize=10)

fig.tight_layout()
plt.savefig(f"Métricas_Hilbert_Box.png", dpi=300)

fig, axs = plt.subplots(1, 5, figsize=(22, 4))

for j in range(len(ondas_latex)):
    ax = axs[j]
    bp = ax.boxplot([resultados_metricas_a_pearson[2, :, j], resultados_metricas_c_pearson[2, :, j]],
                    positions=[1, 2],
                    widths=0.4)

    bp['medians'][0].set_color('orange')
    bp['medians'][0].set_linewidth(2)
    bp['medians'][1].set_color('blue')
    bp['medians'][1].set_linewidth(2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Autismo', 'Controle'], fontsize=10)
    ax.set_title(ondas_latex[j], fontsize=12)
    if j == 0:
        ax.set_ylabel(f'Informação Mútua (MI)', fontsize=12)


plt.savefig("Mutual_Information_Box.png", dpi=300)

plt.show()
