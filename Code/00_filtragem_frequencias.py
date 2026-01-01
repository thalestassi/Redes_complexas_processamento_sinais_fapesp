import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

DIRAutism = 'Autists_ASDProject_TimeSeries/'
DIRControl = 'Controls_ASDProject_TimeSeries/'

final_data = np.zeros((2, 24, 2, 7, 26500), dtype=np.float32)

ondas_defs = {
    1: [0, 1],  # Trash
    2: [1, 3],  # Delta
    3: [3, 8],  # Theta
    4: [8, 15],  # Alpha
    5: [15, 30],  # Beta
    6: [30, 1000]  # Gamma
}

dt = 1 / 200

def process_group(directory, class_idx, file_prefix):
    for i in range(1, 25):
        filename = os.path.join(directory, f'{file_prefix}{i}.mat')
        pat_idx = i - 1

        if not os.path.exists(filename):
            continue

        try:
            mat = scipy.io.loadmat(filename)

            if 'RawData_Segmented' in mat:
                data_struct = mat['RawData_Segmented']
            else:
                keys = [k for k in mat.keys() if not k.startswith('__')]
                if not keys:
                    continue
                data_struct = mat[keys[0]]

            electrodes_data = data_struct[0, 0]
            for sensor_idx in range(22):
                original_signal = electrodes_data[sensor_idx].flatten()
                current_len = len(original_signal)
                limit = min(current_len, 26500)

                final_data[class_idx, pat_idx, sensor_idx, 0, :limit] = original_signal[:limit]

                if limit > 0:
                    sig_to_process = original_signal[:limit] if current_len > 26500 else original_signal
                    sig_fft = fft(sig_to_process)
                    sample_freq = fftfreq(len(sig_to_process), d=dt)
                    freq_abs = np.abs(sample_freq)

                    for wave_id, limites in ondas_defs.items():
                        band_fft = sig_fft.copy()
                        mask = (freq_abs < limites[0]) | (freq_abs >= limites[1])
                        band_fft[mask] = 0

                        filtered_signal = ifft(band_fft).real

                        final_data[class_idx, pat_idx, sensor_idx, wave_id, :len(filtered_signal)] = filtered_signal

        except Exception as e:
            print(f"  [Erro] Falha ao processar {filename}: {e}")

    print(f"Finalizado grupo {file_prefix}.")



process_group(DIRAutism, 0, 'Autismo')
process_group(DIRControl, 1, 'Controle')


np.save("EEG/formatted_data.npy", final_data)

paciente = 3
grupo = 1
eletrodo = 5

waves_to_plot = [
    (0, r'$S_{5}(t)$', 'black'),
    (2, r'$S_{5}^\delta(t)$', 'blue'),
    (3, r'$S_{5}^\theta(t)$', 'green'),
    (4, r'$S_{5}^\alpha(t)$', 'orange'),
    (5, r'$S_{5}^\beta(t)$', 'red'),
    (6, r'$S_{5}^\gamma(t)$', 'purple')
]

indices = [w[0] for w in waves_to_plot]
all_selected_signals = final_data[grupo, paciente, eletrodo, indices, :]

global_min = np.min(all_selected_signals)
global_max = np.max(all_selected_signals)

margin = (global_max - global_min) * 0.05
y_limits = (global_min - margin, global_max + margin)

time_axis = np.arange(26500) * dt

fig, axes = plt.subplots(
    len(waves_to_plot),
    1,
    figsize=(15, 12),
    sharex=True,
    gridspec_kw={'hspace': 0.3}
)

for ax, (wave_idx, label, color) in zip(axes, waves_to_plot):
    signal = final_data[grupo, paciente, eletrodo, wave_idx, :]
    lw = 1
    ax.plot(time_axis, signal, color=color, linewidth=lw)
    ax.set_ylabel(label, fontsize=12)
    ax.set_ylim(y_limits)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[-1].set_xlabel('Tempo (segundos)', fontsize=12)
axes[-1].set_xlim(0, 26500 * dt)

fig.align_ylabels(axes)
plt.tight_layout()
plt.show()