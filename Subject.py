import numpy as np
from biosppy.signals import ecg
import io

def read_signal_data(file_path, buffer_size=1024 * 1024):
    with open(file_path, 'rb') as dataecg:
        data = io.BufferedReader(dataecg, buffer_size=buffer_size)
        while True:
            chunk = data.read(buffer_size)
            if not chunk:
                break
            lines = chunk.decode().strip().split('\n')
            for line in lines:
                line_data = line.strip().split('\t')
                if len(line_data) >= 3:
                    yield float(line_data[2])

def ssf_segmenter_generator(signal_generator, sampling_rate):
    signal = []
    for value in signal_generator:
        signal.append(value)
        if len(signal) >= sampling_rate:  # Batches of 1 second data
            ecg_output = ecg.ssf_segmenter(signal=np.array(signal), sampling_rate=sampling_rate)
            rr_locations = ecg_output['rpeaks']
            rr_intervals = np.diff(rr_locations)
            yield rr_intervals
            signal = []
    if signal:
        ecg_output = ecg.ssf_segmenter(signal=np.array(signal), sampling_rate=sampling_rate)
        rr_locations = ecg_output['rpeaks']
        rr_intervals = np.diff(rr_locations)
        yield rr_intervals

signal_generator = read_signal_data('nama_file.txt')
rr_intervals_generator = ssf_segmenter_generator(signal_generator, sampling_rate=3000000)

all_rr_intervals = []
for rr_intervals in rr_intervals_generator:
    all_rr_intervals.extend(rr_intervals)

print(np.array(all_rr_intervals))