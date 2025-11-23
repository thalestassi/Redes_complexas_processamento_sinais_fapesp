import numpy as np
from scipy.signal import find_peaks
import time

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

eletrodos = ['F7', 'T3', 'T5', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F8', 'T4', 'T6',
             'Fp2', 'F4', 'C4', 'P4', 'O2', 'Fz', 'Cz', 'Pz', 'Oz', 'A1', 'A2']

electrode_pairs = [(i, j) for i in range(len(eletrodos)) for j in range(len(eletrodos)) if i != j]


def calcular_phi_peak(matriz, prominence):
    peaks, _ = find_peaks(matriz, prominence=prominence)
    phi = np.zeros(26500)

    if len(peaks) == 0:
        return phi

    t_b = peaks[0]
    if t_b > 0:
        phi[:t_b + 1] = 2 * np.pi * np.arange(t_b + 1) / t_b

    for i in range(len(peaks) - 1):
        t_a = peaks[i]
        t_b = peaks[i + 1]
        if t_b > t_a:
            intervalo = np.arange(t_a + 1, t_b + 1)
            phi[intervalo] = 2 * np.pi * (intervalo - t_a) / (t_b - t_a)

    t_a = peaks[-1]
    t_b = 26499
    if t_b > t_a:
        intervalo = np.arange(t_a + 1, 26500)
        phi[intervalo] = 2 * np.pi * (intervalo - t_a) / (t_b - t_a)

    return phi


for prom in np.arange(0.00, 0.7, 0.1):
    a_peak = np.zeros((np.shape(x_a)[0], len(ondas), len(eletrodos), len(eletrodos)))
    c_peak = np.zeros((np.shape(x_c)[0], len(ondas), len(eletrodos), len(eletrodos)))
    inicio = time.time()
    print(f'Prominance: {prom}')
    for paciente in range(np.shape(x_a)[0]):
        for wave_idx, k in enumerate(range(2, len(ondas) + 2)):
            matriz_a = np.zeros((len(eletrodos), len(eletrodos)))
            phi_values_a = np.zeros((len(eletrodos), 26500))
            for eletrodo in range(len(eletrodos)):
                phi_values_a[eletrodo] = calcular_phi_peak(x_a[paciente, eletrodo, k], prominence=prom)
            for i, j in electrode_pairs:
                fase_dif = phi_values_a[i] - phi_values_a[j]
                r = np.abs(np.mean(np.exp(1j * fase_dif)))
                matriz_a[i, j] = r
            a_peak[paciente, wave_idx] = matriz_a

            matriz_c = np.zeros((len(eletrodos), len(eletrodos)))
            phi_values_c = np.zeros((len(eletrodos), 26500))
            for eletrodo in range(len(eletrodos)):
                phi_values_c[eletrodo] = calcular_phi_peak(x_c[paciente, eletrodo, k], prominence=prom)
            for i, j in electrode_pairs:
                fase_dif = phi_values_c[i] - phi_values_c[j]
                r = np.abs(np.mean(np.exp(1j * fase_dif)))
                matriz_c[i, j] = r
            c_peak[paciente, wave_idx] = matriz_c

            fim = time.time()
            print(f"Tempo total: {fim - inicio:.2f} segundos")

    prom_str = f"{prom:.2f}".replace('.', '_')
    np.save(f"Matrizes_Adjacencia_Prominence/Matriz_autista_peak_prom_{prom_str}.npy", a_peak)
    np.save(f"Matrizes_Adjacencia_Prominence/Matriz_controle_peak_prom_{prom_str}.npy", c_peak)
