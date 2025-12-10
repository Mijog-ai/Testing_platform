import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import os

folder = "FRF_TU_Berlin/assembly"
files = ["FRF_2_X.txt", "FRF_2_Y.txt", "FRF_2_Z.txt"]
# files =  ["Coherence_2_X.txt","Coherence_2_Y.txt","Coherence_2_Z.txt"]
def load_frf(file_path):
    with open(file_path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    # Extract numeric lines
    data_lines = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
    data_text = ''.join(data_lines).replace(',', '.')

    df = pd.read_csv(StringIO(data_text), sep=r"\s+", header=None, names=["Freq", "Real", "Imag"])
    df = df.apply(pd.to_numeric, errors="coerce")
    df["Mag"] = np.sqrt(df["Real"]**2 + df["Imag"]**2)
    df["Phase_deg"] = np.degrees(np.arctan2(df["Imag"], df["Real"]))
    return df

# Load all 3 axes
frf_data = {}
for f in files:
    axis = os.path.splitext(f)[0].split('_')[-1]  # X, Y, or Z
    frf_data[axis] = load_frf(os.path.join(folder, f))

# Merge on frequency
combined = frf_data['X'][['Freq']].copy()
for axis in ['X', 'Y', 'Z']:
    combined[f"Mag_{axis}"] = frf_data[axis]["Mag"]

# === Statistical summary ===
combined["Mean_Mag"] = combined[["Mag_X", "Mag_Y", "Mag_Z"]].mean(axis=1)
combined["Std_Mag"] = combined[["Mag_X", "Mag_Y", "Mag_Z"]].std(axis=1)

# === Visualization ===
plt.figure(figsize=(10, 6))
plt.plot(combined["Freq"], combined["Mag_X"], label="FRF_X")
plt.plot(combined["Freq"], combined["Mag_Y"], label="FRF_Y")
plt.plot(combined["Freq"], combined["Mag_Z"], label="FRF_Z")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [g/N]")
plt.title("FRF Magnitude (X, Y, Z)")
plt.legend()
plt.grid(True)
plt.show()

# === Mean ± Std band ===
plt.figure(figsize=(10, 5))
plt.plot(combined["Freq"], combined["Mean_Mag"], label="Mean Magnitude", color="black")
plt.fill_between(combined["Freq"],
                 combined["Mean_Mag"] - combined["Std_Mag"],
                 combined["Mean_Mag"] + combined["Std_Mag"],
                 color="gray", alpha=0.3, label="±1 Std Dev")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [g/N]")
plt.title("Mean ± Std of FRFs (X, Y, Z)")
plt.legend()
plt.grid(True)
plt.show()


