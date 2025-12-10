"""
FRF and Coherence Data Analysis
================================

Analysis of structural vibration data from LMS Test.Lab
- Frequency Response Functions (FRF)
- Coherence measurements
- Modal parameter extraction
- Quality assessment

Author: Generated for structural dynamics analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import find_peaks


# ============================================================================
# PARSER FOR LMS TEST.LAB FORMAT
# ============================================================================

def parse_lms_file(filepath):
    """
    Parse LMS Test.Lab format file (FRF or Coherence)

    The file has two sections:
    1. Metadata header (key-value pairs separated by tabs)
    2. Data section (numerical values)

    Parameters:
    -----------
    filepath : str
        Path to the LMS file

    Returns:
    --------
    metadata : dict
        Dictionary containing all metadata from header
    data : pd.DataFrame
        DataFrame with frequency and measurement data
    """

    metadata = {}
    data_lines = []
    data_started = False

    print(f"Parsing file: {filepath}")

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check if data section starts
            if not data_started:
                parts = line.split('\t')
                try:
                    # Try to parse first element as number
                    # Replace European decimal comma with period
                    test_val = parts[0].replace(',', '.')
                    float(test_val)
                    # If successful, data has started
                    data_started = True
                    data_lines.append(parts)
                except (ValueError, IndexError):
                    # Still in metadata section
                    if '\t' in line:
                        parts = line.split('\t')
                        key = parts[0].strip()
                        if len(parts) > 1:
                            value = parts[1].strip()
                            if key and value:
                                metadata[key] = value
            else:
                # In data section
                parts = line.split('\t')
                data_lines.append(parts)

    # Convert data to DataFrame
    if data_lines:
        cleaned_data = []
        for row in data_lines:
            cleaned_row = []
            for val in row:
                if val.strip():
                    # Replace comma with period for decimal
                    cleaned_val = val.replace(',', '.')
                    try:
                        cleaned_row.append(float(cleaned_val))
                    except ValueError:
                        cleaned_row.append(np.nan)

            # Only add rows with at least 2 values
            if len(cleaned_row) >= 2:
                cleaned_data.append(cleaned_row)

        # Determine columns based on file type
        function_class = metadata.get('Function class', '')

        if 'Coherence' in function_class:
            # Coherence files: Frequency, Coherence, (Phase optional)
            if len(cleaned_data[0]) >= 3:
                df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'Coherence', 'Phase'])
            else:
                df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'Coherence'])

        elif 'FRF' in function_class:
            # FRF files: Frequency, Real, Imaginary (complex data)
            if len(cleaned_data[0]) >= 3:
                df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'FRF_Real', 'FRF_Imag'])
                # Calculate magnitude and phase
                df['FRF_Magnitude'] = np.sqrt(df['FRF_Real'] ** 2 + df['FRF_Imag'] ** 2)
                df['FRF_Phase_rad'] = np.arctan2(df['FRF_Imag'], df['FRF_Real'])
                df['FRF_Phase_deg'] = np.degrees(df['FRF_Phase_rad'])
            else:
                df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'Value'])
        else:
            # Generic format
            num_cols = max(len(row) for row in cleaned_data)
            col_names = ['Frequency_Hz'] + [f'Column_{i}' for i in range(1, num_cols)]
            df = pd.DataFrame(cleaned_data, columns=col_names[:num_cols])

        # Remove rows with all NaN
        df = df.dropna(how='all')

        print(f"  ✓ Loaded {len(df)} frequency points")
        print(f"  ✓ Frequency range: {df['Frequency_Hz'].min():.2f} - {df['Frequency_Hz'].max():.2f} Hz")
    else:
        df = pd.DataFrame()
        print("  ✗ No data found in file")

    return metadata, df


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def find_resonance_peaks(df_frf, prominence=0.1, distance=10):
    """
    Find resonance peaks (natural frequencies) in FRF data

    Parameters:
    -----------
    df_frf : pd.DataFrame
        FRF data with FRF_Magnitude column
    prominence : float
        Minimum prominence for peak detection
    distance : int
        Minimum distance between peaks (in samples)

    Returns:
    --------
    peaks_df : pd.DataFrame
        DataFrame with peak information
    """

    if 'FRF_Magnitude' not in df_frf.columns:
        return pd.DataFrame()

    magnitude = df_frf['FRF_Magnitude'].values
    frequencies = df_frf['Frequency_Hz'].values

    # Find peaks
    peaks, properties = find_peaks(magnitude, prominence=prominence, distance=distance)

    if len(peaks) == 0:
        return pd.DataFrame()

    peaks_df = pd.DataFrame({
        'Peak_Number': range(1, len(peaks) + 1),
        'Frequency_Hz': frequencies[peaks],
        'Magnitude': magnitude[peaks],
        'Prominence': properties['prominences']
    })

    # Sort by magnitude (highest first)
    peaks_df = peaks_df.sort_values('Magnitude', ascending=False).reset_index(drop=True)

    return peaks_df


def calculate_frf_statistics(df_frf):
    """
    Calculate statistical features from FRF data
    """

    if df_frf.empty or 'FRF_Magnitude' not in df_frf.columns:
        return {}

    mag = df_frf['FRF_Magnitude'].values
    freq = df_frf['Frequency_Hz'].values

    features = {
        'max_magnitude': np.max(mag),
        'mean_magnitude': np.mean(mag),
        'std_magnitude': np.std(mag),
        'median_magnitude': np.median(mag),
        'peak_frequency': freq[np.argmax(mag)],
        'energy_total': np.sum(mag ** 2),
        'rms_magnitude': np.sqrt(np.mean(mag ** 2))
    }

    # Frequency bands
    bands = {
        'low': (0, 100),
        'mid': (100, 500),
        'high': (500, 1000)
    }

    for band_name, (f_low, f_high) in bands.items():
        mask = (freq >= f_low) & (freq <= f_high)
        if np.any(mask):
            features[f'energy_{band_name}'] = np.sum(mag[mask] ** 2)
            features[f'mean_mag_{band_name}'] = np.mean(mag[mask])

    return features


def calculate_coherence_statistics(df_coh):
    """
    Calculate statistical features from Coherence data
    """

    if df_coh.empty or 'Coherence' not in df_coh.columns:
        return {}

    coh = df_coh['Coherence'].values

    features = {
        'coherence_mean': np.mean(coh),
        'coherence_std': np.std(coh),
        'coherence_min': np.min(coh),
        'coherence_max': np.max(coh),
        'coherence_median': np.median(coh),
        'coherence_q25': np.percentile(coh, 25),
        'coherence_q75': np.percentile(coh, 75),
        'pct_above_0.8': np.sum(coh > 0.8) / len(coh) * 100,
        'pct_above_0.9': np.sum(coh > 0.9) / len(coh) * 100,
        'pct_below_0.5': np.sum(coh < 0.5) / len(coh) * 100
    }

    return features


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_structural_data(data_directory, axes=['X', 'Y', 'Z']):
    """
    Complete analysis of FRF and Coherence data

    Parameters:
    -----------
    data_directory : str
        Path to directory containing FRF and Coherence files
    axes : list
        List of axes to analyze
    """

    results = {}

    for axis in axes:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING AXIS: {axis}")
        print(f"{'=' * 80}")

        results[axis] = {}

        # Load FRF data
        frf_file = f"{data_directory}/FRF_128_{axis}.txt"
        try:
            meta_frf, df_frf = parse_lms_file(frf_file)
            results[axis]['frf_data'] = df_frf
            results[axis]['frf_metadata'] = meta_frf

            # Extract FRF features
            frf_stats = calculate_frf_statistics(df_frf)
            results[axis]['frf_statistics'] = frf_stats

            # Find resonance peaks
            peaks = find_resonance_peaks(df_frf, prominence=0.05)
            results[axis]['resonance_peaks'] = peaks

            print(f"\nFRF Statistics for {axis}:")
            for key, value in list(frf_stats.items())[:8]:
                print(f"  {key:20s}: {value:.4f}")

            if not peaks.empty:
                print(f"\nTop 5 Resonance Peaks for {axis}:")
                print(peaks.head(5).to_string(index=False))

        except Exception as e:
            print(f"✗ Error loading FRF_{axis}: {e}")

        # Load Coherence data
        coh_file = f"{data_directory}/Coherence_128_{axis}.txt"
        try:
            meta_coh, df_coh = parse_lms_file(coh_file)
            results[axis]['coherence_data'] = df_coh
            results[axis]['coherence_metadata'] = meta_coh

            # Extract coherence features
            coh_stats = calculate_coherence_statistics(df_coh)
            results[axis]['coherence_statistics'] = coh_stats

            print(f"\nCoherence Statistics for {axis}:")
            for key, value in coh_stats.items():
                print(f"  {key:20s}: {value:.4f}")

        except Exception as e:
            print(f"✗ Error loading Coherence_{axis}: {e}")

    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_frf_analysis(results, save_path='frf_analysis.png'):
    """
    Create comprehensive FRF visualization
    """

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('FRF and Coherence Analysis - All Axes', fontsize=16, fontweight='bold')

    colors = {'X': 'red', 'Y': 'blue', 'Z': 'green'}

    for idx, axis in enumerate(['X', 'Y', 'Z']):
        if axis not in results or 'frf_data' not in results[axis]:
            continue

        df_frf = results[axis]['frf_data']
        df_coh = results[axis].get('coherence_data', pd.DataFrame())
        color = colors[axis]

        # Row 1: FRF Magnitude
        ax1 = axes[idx, 0]
        ax1.plot(df_frf['Frequency_Hz'], df_frf['FRF_Magnitude'],
                 color=color, linewidth=1.5, label=f'Axis {axis}')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (g/N)')
        ax1.set_title(f'FRF Magnitude - Axis {axis}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')

        # Mark peaks
        if 'resonance_peaks' in results[axis] and not results[axis]['resonance_peaks'].empty:
            peaks = results[axis]['resonance_peaks'].head(5)
            ax1.scatter(peaks['Frequency_Hz'], peaks['Magnitude'],
                        color='red', s=100, marker='x', linewidths=3,
                        label='Resonances', zorder=5)

        # Row 2: FRF Phase
        ax2 = axes[idx, 1]
        if 'FRF_Phase_deg' in df_frf.columns:
            ax2.plot(df_frf['Frequency_Hz'], df_frf['FRF_Phase_deg'],
                     color=color, linewidth=1.5)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Phase (degrees)')
            ax2.set_title(f'FRF Phase - Axis {axis}')
            ax2.grid(True, alpha=0.3)

        # Row 3: Coherence
        ax3 = axes[idx, 2]
        if not df_coh.empty and 'Coherence' in df_coh.columns:
            ax3.plot(df_coh['Frequency_Hz'], df_coh['Coherence'],
                     color=color, linewidth=1.5)
            ax3.axhline(y=0.8, color='orange', linestyle='--',
                        label='Quality threshold (0.8)', linewidth=2)
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Coherence')
            ax3.set_title(f'Coherence - Axis {axis}')
            ax3.set_ylim([0, 1.05])
            ax3.grid(True, alpha=0.3)
            ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {save_path}")

    return fig


def create_comparison_report(results, save_path='comparison_report.txt'):
    """
    Create a text report comparing all axes
    """

    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FRF AND COHERENCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary table
        f.write("SUMMARY COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<30} {'X-Axis':>15} {'Y-Axis':>15} {'Z-Axis':>15}\n")
        f.write("-" * 80 + "\n")

        metrics = [
            ('Peak Frequency (Hz)', 'peak_frequency'),
            ('Max Magnitude', 'max_magnitude'),
            ('Mean Magnitude', 'mean_magnitude'),
            ('RMS Magnitude', 'rms_magnitude'),
            ('Total Energy', 'energy_total')
        ]

        for metric_name, metric_key in metrics:
            values = []
            for axis in ['X', 'Y', 'Z']:
                if axis in results and 'frf_statistics' in results[axis]:
                    val = results[axis]['frf_statistics'].get(metric_key, 0)
                    values.append(f"{val:>15.4f}")
                else:
                    values.append(f"{'N/A':>15}")

            f.write(f"{metric_name:<30} {values[0]} {values[1]} {values[2]}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("COHERENCE QUALITY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<30} {'X-Axis':>15} {'Y-Axis':>15} {'Z-Axis':>15}\n")
        f.write("-" * 80 + "\n")

        coh_metrics = [
            ('Mean Coherence', 'coherence_mean'),
            ('Min Coherence', 'coherence_min'),
            ('% Above 0.8', 'pct_above_0.8'),
            ('% Above 0.9', 'pct_above_0.9')
        ]

        for metric_name, metric_key in coh_metrics:
            values = []
            for axis in ['X', 'Y', 'Z']:
                if axis in results and 'coherence_statistics' in results[axis]:
                    val = results[axis]['coherence_statistics'].get(metric_key, 0)
                    values.append(f"{val:>15.2f}")
                else:
                    values.append(f"{'N/A':>15}")

            f.write(f"{metric_name:<30} {values[0]} {values[1]} {values[2]}\n")

        # Resonance peaks
        f.write("\n" + "=" * 80 + "\n")
        f.write("RESONANCE PEAKS (NATURAL FREQUENCIES)\n")
        f.write("=" * 80 + "\n")

        for axis in ['X', 'Y', 'Z']:
            if axis in results and 'resonance_peaks' in results[axis]:
                peaks = results[axis]['resonance_peaks']
                if not peaks.empty:
                    f.write(f"\nAxis {axis}:\n")
                    f.write(peaks.head(10).to_string(index=False))
                    f.write("\n")

    print(f"✓ Saved report: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FRF AND COHERENCE DATA ANALYSIS")
    print("=" * 80)

    # UPDATE THIS PATH to your data directory
    DATA_DIR = "../Datasets"  # or "H:/Your/Path/To/Data"

    # Run analysis
    results = analyze_structural_data(DATA_DIR, axes=['X', 'Y', 'Z'])

    plot_frf_analysis(results, save_path='frf_analysis_128.png')

    create_comparison_report(results, save_path='comparison_report_128.txt')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. frf_analysis.png - Visualization of all FRF and coherence data")
    print("  2. comparison_report.txt - Detailed comparison report")