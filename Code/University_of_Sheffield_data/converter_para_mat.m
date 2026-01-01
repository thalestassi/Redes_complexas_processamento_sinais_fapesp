function exportEEGLAB2mat(ALLEEG, filename)
    bands = struct( ...
        'Delta', [1 3], ...
		'Teta', [3 8], ...
        'Alpha', [8 15], ...
        'Beta',  [15 30], ...
        'Gamma', [30 1000]);
    bandNames = fieldnames(bands);
    nBands = numel(bandNames);

    nPatients = numel(ALLEEG);
    eeg_data = struct;

    for p = 51:nPatients
        patient_name = ALLEEG(p).setname;
        data         = ALLEEG(p).data;   % [nChannels x nSamples]
        fs           = ALLEEG(p).srate;
        [nSensors, nSamples] = size(data);

        freqs = (0:nSamples-1)*(fs/nSamples);

        eeg_data(p).name = patient_name;
        eeg_data(p).bands = cell(nBands,1);
        eeg_data(p).band_labels = bandNames;

        for b = 1:nBands
            range = bands.(bandNames{b});
            band_signal = zeros(nSensors, nSamples);

            for s = 1:nSensors
                X = fft(double(data(s,:)));  
                mask = (freqs >= range(1) & freqs <= range(2)) | ...
                       (freqs >= fs-range(2) & freqs <= fs-range(1));
                X(~mask) = 0;
                band_signal(s,:) = real(ifft(X)); 
            end
            eeg_data(p).bands{b} = band_signal;
        end
    end

    save(filename, 'eeg_data');
    fprintf('Arquivo salvo: %s\n', filename);
end
