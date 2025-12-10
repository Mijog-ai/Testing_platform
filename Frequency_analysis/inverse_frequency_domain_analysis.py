import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

def load_frf(file_path):
    """Load and compute impulse response from FRF file."""
    with open(file_path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    data_lines = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
    data_text = ''.join(data_lines).replace(',', '.')
    df = pd.read_csv(StringIO(data_text), sep=r"\s+", header=None, names=["Freq_Hz", "Real", "Imag"])
    df = df.apply(pd.to_numeric, errors="coerce")

    # === Build complex FRF ===
    H_f = df["Real"].values + 1j * df["Imag"].values

    # === Mirror spectrum (one-sided to full) ===
    H_full = np.concatenate([H_f, np.conj(H_f[-2:0:-1])])

    # === Inverse FFT ===
    h_t = np.fft.ifft(H_full).real

    # === Time vector ===
    Fs = 2 * df["Freq_Hz"].iloc[-1]  # Sampling frequency
    T = 1 / Fs
    t = np.arange(0, len(h_t)) * T

    return t, h_t

# === File paths ===
files = {
    "X": "FRF_TU_Berlin/assembly/FRF_2_X.txt",
    "Y": "FRF_TU_Berlin/assembly/FRF_2_Y.txt",
    "Z": "FRF_TU_Berlin/assembly/FRF_2_Z.txt"
}

# === Plot all three impulse responses ===
plt.figure(figsize=(10, 6))

for i, (label, path) in enumerate(files.items()):
    t, h_t = load_frf(path)
    plt.plot(t * 1000, h_t, label=f"FRF_2_{label}")

plt.xlabel("Time [ms]")
plt.ylabel("Amplitude [g/N]")
plt.title("Impulse Response Function – X, Y, Z Directions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
