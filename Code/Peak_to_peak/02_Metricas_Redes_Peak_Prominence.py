import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import skew, kurtosis, entropy


'''
x_a: autism group

x_c: control group

Data format x_a and x_c: [Patient ID, Sensor ID, Wave ID, Time Series]

Wave ID: [0:original, 1:trash, 2:delta, 3:theta, 4:alpha, 5:beta, 6:gamma]
'''

xxx = np.load('formatted_data.npy')
x_a = xxx[0]  # autism data
x_c = xxx[1]  # control data

ondas = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ondas_latex = [r'Ondas $\delta$', r'Ondas $\theta$', r'Ondas $\alpha$', r'Ondas $\beta$', r'Ondas $\gamma$']

metricas = ['Grau Médio', 'Var dist grau', 'Skewness', 'Kurtosis',
            'Entropia do grau', 'Coeficiente de aglomeração médio', 'Menor caminho médio']

prominencias = np.round(np.arange(0.00, 0.7, 0.1), 2)

for p_idx, p in enumerate(prominencias):
    print(f"Analisando prominência: {p}")
    prom_str = f"{p:.2f}".replace('.', '_')
    matrizes_a_peak = np.load(f"Matrizes_Adjacencia_Prominence/Matriz_autista_peak_prom_{prom_str}.npy")
    matrizes_c_peak = np.load(f"Matrizes_Adjacencia_Prominence/Matriz_controle_peak_prom_{prom_str}.npy")
    grau_medio_a_peak = np.zeros((np.shape(x_a)[0], len(ondas)))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            grau_medio = np.sum(matrizes_a_peak[paciente, onda].flatten()) / 22
            grau_medio_a_peak[paciente, onda] = grau_medio

    grau_medio_c_peak = np.zeros((np.shape(x_c)[0], len(ondas)))
    for paciente in range(np.shape(x_c)[0]):
        for onda in range(len(ondas)):
            grau_medio = np.sum(matrizes_c_peak[paciente, onda].flatten()) / 22
            grau_medio_c_peak[paciente, onda] = grau_medio

    grau_cada_linha_a_peak = np.zeros((np.shape(x_c)[0], len(ondas), 22))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            for j in range(22):
                grau_cada_linha_a_peak[paciente, onda, j] = sum(matrizes_a_peak[paciente, onda, j])

    grau_cada_linha_c_peak = np.zeros((np.shape(x_c)[0], len(ondas), 22))
    for paciente in range(np.shape(x_c)[0]):
        for onda in range(len(ondas)):
            for j in range(22):
                grau_cada_linha_c_peak[paciente, onda, j] = sum(matrizes_c_peak[paciente, onda, j])

    '''
    Variância da distribuição de grau
    '''

    variancia_a_peak = np.zeros((np.shape(x_a)[0], len(ondas)))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            variancia = 0
            for i in range(22):
                variancia = variancia + (
                        (grau_cada_linha_a_peak[paciente, onda, i] - grau_medio_a_peak[paciente, onda]) ** 2) / 22
            variancia_a_peak[paciente, onda] = variancia

    variancia_c_peak = np.zeros((np.shape(x_c)[0], len(ondas)))
    for paciente in range(np.shape(x_c)[0]):
        for onda in range(len(ondas)):
            variancia = 0
            for i in range(22):
                variancia = variancia + (
                        (grau_cada_linha_c_peak[paciente, onda, i] - grau_medio_c_peak[paciente, onda]) ** 2) / 22
            variancia_c_peak[paciente, onda] = variancia

    '''
    Skewness
    '''

    skewness_a_peak = np.zeros((np.shape(x_a)[0], len(ondas)))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            skewness_a_peak[paciente, onda] = skew(grau_cada_linha_a_peak[paciente, onda])

    skewness_c_peak = np.zeros((np.shape(x_a)[0], len(ondas)))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            skewness_c_peak[paciente, onda] = skew(grau_cada_linha_c_peak[paciente, onda])

    '''
    Kurtosis
    '''

    kurtosis_a_peak = np.zeros((np.shape(x_a)[0], len(ondas)))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            kurtosis_a_peak[paciente, onda] = kurtosis(grau_cada_linha_a_peak[paciente, onda])

    kurtosis_c_peak = np.zeros((np.shape(x_c)[0], len(ondas)))
    for paciente in range(np.shape(x_c)[0]):
        for onda in range(len(ondas)):
            kurtosis_c_peak[paciente, onda] = kurtosis(grau_cada_linha_c_peak[paciente, onda])

    '''
    Entropia do grau
    '''

    entropia_a_peak = np.zeros((np.shape(x_a)[0], len(ondas)))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            entropia_a_peak[paciente, onda] = entropy(grau_cada_linha_a_peak[paciente, onda])

    entropia_c_peak = np.zeros((np.shape(x_c)[0], len(ondas)))
    for paciente in range(np.shape(x_c)[0]):
        for onda in range(len(ondas)):
            entropia_c_peak[paciente, onda] = entropy(grau_cada_linha_c_peak[paciente, onda])
    '''
    Coeficiente de aglomeração médio
    '''

    agrupamento_medio_a_peak = np.zeros((np.shape(x_a)[0], len(ondas)))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            agrupamento_medio_a_peak[paciente, onda] = nx.average_clustering(
                nx.from_numpy_array(matrizes_a_peak[paciente, onda]), weight="weight")

    agrupamento_medio_c_peak = np.zeros((np.shape(x_c)[0], len(ondas)))
    for paciente in range(np.shape(x_c)[0]):
        for onda in range(len(ondas)):
            agrupamento_medio_c_peak[paciente, onda] = nx.average_clustering(
                nx.from_numpy_array(matrizes_c_peak[paciente, onda]), weight="weight")

    '''
    Menor caminho médio
    '''

    menor_caminho_medio_a_peak = np.zeros((np.shape(x_a)[0], len(ondas)))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            menor_caminho_medio_a_peak[paciente, onda] = nx.average_shortest_path_length(
                nx.from_numpy_array(matrizes_a_peak[paciente, onda]), weight="weight")

    menor_caminho_medio_c_peak = np.zeros((np.shape(x_c)[0], len(ondas)))
    for paciente in range(np.shape(x_c)[0]):
        for onda in range(len(ondas)):
            menor_caminho_medio_c_peak[paciente, onda] = nx.average_shortest_path_length(
                nx.from_numpy_array(matrizes_c_peak[paciente, onda]), weight="weight")

    '''
    Assortatividade de grau
    '''

    assortatividade_grau_a_peak = np.zeros((np.shape(x_a)[0], len(ondas)))
    for paciente in range(np.shape(x_a)[0]):
        for onda in range(len(ondas)):
            G = nx.from_numpy_array(matrizes_a_peak[paciente, onda])
            assortatividade_grau_a_peak[paciente, onda] = nx.degree_assortativity_coefficient(G, weight="weight")

    assortatividade_grau_c_peak = np.zeros((np.shape(x_c)[0], len(ondas)))
    for paciente in range(np.shape(x_c)[0]):
        for onda in range(len(ondas)):
            G = nx.from_numpy_array(matrizes_c_peak[paciente, onda])
            assortatividade_grau_c_peak[paciente, onda] = nx.degree_assortativity_coefficient(G, weight="weight")

    resultados_metricas_a_peak = np.array(
        [grau_medio_a_peak, variancia_a_peak, skewness_a_peak, kurtosis_a_peak, entropia_a_peak,
         agrupamento_medio_a_peak, menor_caminho_medio_a_peak])
    resultados_metricas_c_peak = np.array(
        [grau_medio_c_peak, variancia_c_peak, skewness_c_peak, kurtosis_c_peak, entropia_c_peak,
         agrupamento_medio_c_peak, menor_caminho_medio_c_peak])

    fig, axs = plt.subplots(len(ondas), len(metricas), figsize=(22, 12))  #
    for j in range(len(ondas)):
        for i in range(len(metricas)):
            ax = axs[j, i]
            bp = ax.boxplot([resultados_metricas_a_peak[i, :, j], resultados_metricas_c_peak[i, :, j]],
                            positions=[1, 2],
                            widths=0.2)
            bp['medians'][0].set_color('orange')
            bp['medians'][1].set_color('blue')
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Autism', 'Control'], fontsize=8)
            ax.set_title(f'{metricas[i]}', fontsize=8)
            if i == 0:
                ax.set_ylabel(f'{ondas_latex[j]}', fontsize=10)

    plt.suptitle(f'Boxplots das Métricas - PEAK Prominence = {p:.2f}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # espaço para o título
    plt.savefig(f"Box_Plot_Metricas_Peak/Métricas_Peak_Combinadas_{p:.2f}.png", dpi=300)


