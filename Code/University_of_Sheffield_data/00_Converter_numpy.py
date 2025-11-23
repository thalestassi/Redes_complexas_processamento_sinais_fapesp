import numpy as np
import scipy.io as sio

def mat_files_to_numpy_array(mat_files, var_name='eeg_data'):
    grupos = {'P': [], 'ASD': []}
    max_sensors = 0
    nBands = None
    band_labels = None

    for mat_file in mat_files:
        data = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        eeg_data = data[var_name]

        for patient in eeg_data:
            bands = patient.bands
            if nBands is None:
                nBands = len(bands)
                band_labels = patient.band_labels

            nSensors, nSamples = bands[0].shape
            max_sensors = max(max_sensors, nSensors)

    for mat_file in mat_files:
        print(mat_file)
        data = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        eeg_data = data[var_name]
        for patient in eeg_data:
            name = patient.name
            bands = patient.bands
            nSensors, nSamples = bands[0].shape

            patient_array = np.zeros((max_sensors, nBands, nSamples))

            for b in range(nBands):
                patient_array[:nSensors, b, :] = bands[b]

            if name.startswith('P'):
                grupos['P'].append(patient_array)
            elif name.startswith('ASD'):
                grupos['ASD'].append(patient_array)

    eeg_array = {}
    if grupos['P']:
        eeg_array['P'] = np.stack(grupos['P'], axis=0)
    if grupos['ASD']:
        eeg_array['ASD'] = np.stack(grupos['ASD'], axis=0)
    np.save("x_control_new_data.npy", eeg_array['P'])
    np.save("x_autism_new_data.npy", eeg_array['ASD'])
    return eeg_array, band_labels


mat_files = [
    'eeg_pacientes_1_10.mat',
    'eeg_pacientes_11_20.mat',
    'eeg_pacientes_21_30.mat',
    'eeg_pacientes_31_40.mat',
    'eeg_pacientes_41_50.mat',
    'eeg_pacientes_51_55.mat']

eeg_array, band_labels = mat_files_to_numpy_array(mat_files)
print("Shape grupo P:", eeg_array['P'].shape)
print("Shape grupo ASD:", eeg_array['ASD'].shape)

