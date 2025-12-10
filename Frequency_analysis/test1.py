import numpy as np
import matplotlib.pyplot as plt

# Example FRF data (replace with your own)
frequency = np.linspace(0, 100, 500)  # Hz
fn = 50  # natural frequency (Hz)
zeta_true = 0.02  # true damping ratio
omega_n = 2 * np.pi * fn

# Simulated magnitude of FRF
H = 1 / np.sqrt((1 - (frequency / fn)**2)**2 + (2 * zeta_true * frequency / fn)**2)
H = H / np.max(H)  # normalize

# Find half-power points
peak_idx = np.argmax(H)
H_peak = H[peak_idx]
H_half = H_peak / np.sqrt(2)

# Find frequencies around the peak where H crosses H_half
indices = np.where(H >= H_half)[0]
f1 = frequency[indices[0]]
f2 = frequency[indices[-1]]

# Damping ratio using half-power bandwidth
zeta_est = (f2 - f1) / (2 * fn)

print(f"Estimated damping ratio: {zeta_est:.4f}")

# Plot FRF and half-power points
plt.plot(frequency, H)
plt.axhline(H_half, color='r', linestyle='--', label='Half Power')
plt.axvline(f1, color='g', linestyle='--', label=f'f1 = {f1:.2f} Hz')
plt.axvline(f2, color='g', linestyle='--', label=f'f2 = {f2:.2f} Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|H(f)| (normalized)')
plt.legend()
plt.show()
