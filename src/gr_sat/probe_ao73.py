import numpy as np
from scipy.io import wavfile
from scipy import signal
import os

filepath = os.path.join("satellite-recordings", "ao73.wav")
fs, data = wavfile.read(filepath)
data = data.astype(np.float32)
data /= np.max(np.abs(data))

f, Pxx = signal.welch(data, fs, nperseg=4096)
peak_idx = np.argmax(Pxx)
print(f"Direct Peak: {f[peak_idx]:.2f} Hz")

# Squaring for suppressed carrier BPSK
data_sq = data**2
f_sq, Pxx_sq = signal.welch(data_sq, fs, nperseg=4096)
peak_sq_idx = np.argmax(Pxx_sq)
print(f"Squared Peak (2*fc): {f_sq[peak_sq_idx]:.2f} Hz -> fc: {f_sq[peak_sq_idx]/2:.2f} Hz")
