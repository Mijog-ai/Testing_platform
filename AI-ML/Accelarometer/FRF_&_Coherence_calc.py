"""
Complete FRF and Coherence Analysis with Automatic File Reading
Reads FRF_1_X.txt and Coherence_1_X.txt files
Performs step-by-step calculations and plots results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# FUNCTION TO PARSE LMS FILES
# ============================================================================

def parse_lms_file(filepath):
    """Parse LMS Test.Lab format file"""

    print(f"\nReading file: {filepath}")

    metadata = {}
    data_lines = []
    data_started = False

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if not data_started:
                parts = line.split('\t')
                try:
                    # Try to parse as number (data section starts)
                    test_val = parts[0].replace(',', '.')
                    float(test_val)
                    data_started = True
                    data_lines.append(parts)
                except (ValueError, IndexError):
                    # Still in metadata
                    if '\t' in line:
                        parts = line.split('\t')
                        key = parts[0].strip()
                        if len(parts) > 1:
                            value = parts[1].strip()
                            if key and value:
                                metadata[key] = value
            else:
                parts = line.split('\t')
                data_lines.append(parts)

    # Convert to numbers
    cleaned_data = []
    for row in data_lines:
        cleaned_row = []
        for val in row:
            if val.strip():
                cleaned_val = val.replace(',', '.')
                try:
                    cleaned_row.append(float(cleaned_val))
                except ValueError:
                    cleaned_row.append(np.nan)
        if len(cleaned_row) >= 2:
            cleaned_data.append(cleaned_row)

    # Create DataFrame
    function_class = metadata.get('Function class', '')

    if 'FRF' in function_class:
        if len(cleaned_data[0]) >= 3:
            df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'Real', 'Imag'])
        else:
            df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'Value'])
    elif 'Coherence' in function_class:
        if len(cleaned_data[0]) >= 3:
            df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'Coherence', 'Phase'])
        else:
            df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'Coherence'])
    else:
        num_cols = max(len(row) for row in cleaned_data)
        col_names = ['Frequency_Hz'] + [f'Col_{i}' for i in range(1, num_cols)]
        df = pd.DataFrame(cleaned_data, columns=col_names[:num_cols])

    df = df.dropna(how='all')

    print(f"✓ Loaded {len(df)} data points")
    print(f"  Frequency range: {df['Frequency_Hz'].min():.2f} - {df['Frequency_Hz'].max():.2f} Hz")

    return metadata, df


# ============================================================================
# STEP-BY-STEP CALCULATIONS
# ============================================================================

def calculate_frf_magnitude_step_by_step(df_frf, num_points=5):
    """Calculate FRF magnitude with detailed steps"""

    print("\n" + "=" * 80)
    print("STEP 2: CALCULATE FRF MAGNITUDE")
    print("Formula: |H(ω)| = √(Real² + Imag²)")
    print("=" * 80)

    if 'Real' not in df_frf.columns or 'Imag' not in df_frf.columns:
        print("Error: FRF file must have Real and Imag columns")
        return df_frf

    magnitudes = []
    phases_rad = []
    phases_deg = []

    # Show detailed calculation for first few points
    for i in range(min(num_points, len(df_frf))):
        freq = df_frf['Frequency_Hz'].iloc[i]
        real = df_frf['Real'].iloc[i]
        imag = df_frf['Imag'].iloc[i]

        print(f"\n--- Point {i + 1}: f = {freq:.4f} Hz ---")
        print(f"Real = {real:.9f} g/N")
        print(f"Imag = {imag:.9f} g/N")

        # Step 1: Square real part
        real_squared = real ** 2
        print(f"\nStep 1: Real² = ({real:.9f})²")
        print(f"              = {real_squared:.12e}")

        # Step 2: Square imaginary part
        imag_squared = imag ** 2
        print(f"\nStep 2: Imag² = ({imag:.9f})²")
        print(f"              = {imag_squared:.12e}")

        # Step 3: Sum
        sum_squares = real_squared + imag_squared
        print(f"\nStep 3: Real² + Imag² = {real_squared:.12e} + {imag_squared:.12e}")
        print(f"                       = {sum_squares:.12e}")

        # Step 4: Square root
        magnitude = np.sqrt(sum_squares)
        print(f"\nStep 4: |H| = √({sum_squares:.12e})")
        print(f"            = {magnitude:.9f} g/N")

        # Calculate phase
        phase_rad = np.arctan2(imag, real)
        phase_deg = np.degrees(phase_rad)

        print(f"\nPhase calculation:")
        print(f"  Phase = arctan(Imag/Real)")
        print(f"        = arctan({imag:.9f}/{real:.9f})")
        print(f"        = {phase_rad:.6f} rad")
        print(f"        = {phase_deg:.2f}°")

        magnitudes.append(magnitude)
        phases_rad.append(phase_rad)
        phases_deg.append(phase_deg)

    # Calculate for all remaining points without printing
    for i in range(num_points, len(df_frf)):
        real = df_frf['Real'].iloc[i]
        imag = df_frf['Imag'].iloc[i]
        magnitude = np.sqrt(real ** 2 + imag ** 2)
        phase_rad = np.arctan2(imag, real)
        phase_deg = np.degrees(phase_rad)

        magnitudes.append(magnitude)
        phases_rad.append(phase_rad)
        phases_deg.append(phase_deg)

    if num_points < len(df_frf):
        print(f"\n... (calculated remaining {len(df_frf) - num_points} points)")

    # Add to dataframe
    df_frf['Magnitude'] = magnitudes
    df_frf['Phase_rad'] = phases_rad
    df_frf['Phase_deg'] = phases_deg

    print(f"\n✓ Calculated magnitude and phase for all {len(df_frf)} points")

    return df_frf


def analyze_coherence_step_by_step(df_coh, num_points=5):
    """Analyze coherence with quality assessment"""

    print("\n" + "=" * 80)
    print("STEP 3: ANALYZE COHERENCE")
    print("Formula: γ²(ω) - Quality measure from 0 to 1")
    print("=" * 80)

    if 'Coherence' not in df_coh.columns:
        print("Error: Coherence file must have Coherence column")
        return df_coh

    quality = []

    for i in range(min(num_points, len(df_coh))):
        freq = df_coh['Frequency_Hz'].iloc[i]
        coh = df_coh['Coherence'].iloc[i]

        print(f"\n--- Point {i + 1}: f = {freq:.4f} Hz ---")
        print(f"Coherence γ² = {coh:.6f}")
        print(f"Percentage = {coh * 100:.2f}%")

        # Quality assessment
        if coh > 0.95:
            qual = "EXCELLENT"
            symbol = "✓✓✓"
        elif coh > 0.80:
            qual = "GOOD"
            symbol = "✓✓"
        elif coh > 0.50:
            qual = "MARGINAL"
            symbol = "⚠️"
        else:
            qual = "POOR"
            symbol = "✗"

        print(f"Quality: {qual} {symbol}")

        if coh < 0.80:
            print(f"⚠️ Warning: FRF data at this frequency may not be reliable")

        quality.append(qual)

    # Assess remaining points
    for i in range(num_points, len(df_coh)):
        coh = df_coh['Coherence'].iloc[i]
        if coh > 0.95:
            qual = "EXCELLENT"
        elif coh > 0.80:
            qual = "GOOD"
        elif coh > 0.50:
            qual = "MARGINAL"
        else:
            qual = "POOR"
        quality.append(qual)

    if num_points < len(df_coh):
        print(f"\n... (assessed remaining {len(df_coh) - num_points} points)")

    df_coh['Quality'] = quality

    # Overall statistics
    print(f"\n" + "-" * 80)
    print("OVERALL COHERENCE STATISTICS:")
    print(f"  Mean coherence: {df_coh['Coherence'].mean():.4f}")
    print(f"  Min coherence:  {df_coh['Coherence'].min():.4f}")
    print(f"  Max coherence:  {df_coh['Coherence'].max():.4f}")

    excellent_pct = (df_coh['Quality'] == 'EXCELLENT').sum() / len(df_coh) * 100
    good_pct = (df_coh['Quality'] == 'GOOD').sum() / len(df_coh) * 100
    marginal_pct = (df_coh['Quality'] == 'MARGINAL').sum() / len(df_coh) * 100
    poor_pct = (df_coh['Quality'] == 'POOR').sum() / len(df_coh) * 100

    print(f"\n  Quality distribution:")
    print(f"    EXCELLENT (>0.95): {excellent_pct:.1f}%")
    print(f"    GOOD (0.80-0.95):  {good_pct:.1f}%")
    print(f"    MARGINAL (0.50-0.80): {marginal_pct:.1f}%")
    print(f"    POOR (<0.50):      {poor_pct:.1f}%")

    return df_coh


def find_peaks_and_calculate_damping(df_frf):
    """Find peaks and calculate damping ratio"""

    print("\n" + "=" * 80)
    print("STEP 4: FIND PEAKS AND CALCULATE DAMPING")
    print("=" * 80)

    if 'Magnitude' not in df_frf.columns:
        print("Error: Must calculate magnitude first")
        return None

    magnitude = df_frf['Magnitude'].values
    frequencies = df_frf['Frequency_Hz'].values

    # Find peaks (simple method - look for local maxima)
    from scipy.signal import find_peaks

    baseline = np.median(magnitude)
    prominence = 0.05 * np.max(magnitude)

    peaks, properties = find_peaks(magnitude, prominence=prominence, distance=50)

    if len(peaks) == 0:
        print("No significant peaks found in FRF")
        return None

    print(f"\n✓ Found {len(peaks)} peaks")

    results = []

    for idx, peak_idx in enumerate(peaks[:5]):  # Show first 5 peaks
        fn = frequencies[peak_idx]
        Hmax = magnitude[peak_idx]

        print(f"\n" + "-" * 80)
        print(f"PEAK {idx + 1}")
        print("-" * 80)
        print(f"Natural frequency fₙ = {fn:.2f} Hz")
        print(f"Peak magnitude Hₘₐₓ = {Hmax:.6f} g/N")

        # Calculate 70.7% level
        target_level = Hmax / np.sqrt(2)
        print(f"\nHalf-power level = Hₘₐₓ / √2")
        print(f"                 = {Hmax:.6f} / 1.414214")
        print(f"                 = {target_level:.6f} g/N")

        # Find crossing points (simplified)
        # Search left
        left_idx = max(0, peak_idx - 100)
        mag_left = magnitude[left_idx:peak_idx]
        freq_left = frequencies[left_idx:peak_idx]

        # Find where it crosses target level
        crossings_left = np.where(np.diff(np.sign(mag_left - target_level)))[0]
        if len(crossings_left) > 0:
            f1 = freq_left[crossings_left[-1]]
        else:
            f1 = fn - 5  # default

        # Search right
        right_idx = min(len(magnitude), peak_idx + 100)
        mag_right = magnitude[peak_idx:right_idx]
        freq_right = frequencies[peak_idx:right_idx]

        crossings_right = np.where(np.diff(np.sign(mag_right - target_level)))[0]
        if len(crossings_right) > 0:
            f2 = freq_right[crossings_right[0]]
        else:
            f2 = fn + 5  # default

        print(f"\nBandwidth calculation:")
        print(f"  f₁ (left crossing)  = {f1:.2f} Hz")
        print(f"  f₂ (right crossing) = {f2:.2f} Hz")

        # Calculate bandwidth
        bandwidth = abs(f2 - f1)
        print(f"  Δf = f₂ - f₁ = {f2:.2f} - {f1:.2f}")
        print(f"     = {bandwidth:.2f} Hz")

        # Calculate damping ratio
        zeta = bandwidth / (2 * fn)
        zeta_percent = zeta * 100
        print(f"\nDamping ratio calculation:")
        print(f"  ζ = Δf / (2 × fₙ)")
        print(f"    = {bandwidth:.2f} / (2 × {fn:.2f})")
        print(f"    = {bandwidth:.2f} / {2 * fn:.2f}")
        print(f"    = {zeta:.6f}")
        print(f"    = {zeta_percent:.3f}%")

        # Calculate quality factor
        Q = 1 / (2 * zeta) if zeta > 0 else 999
        print(f"\nQuality factor:")
        print(f"  Q = 1 / (2 × ζ)")
        print(f"    = 1 / (2 × {zeta:.6f})")
        print(f"    = 1 / {2 * zeta:.6f}")
        print(f"    = {Q:.2f}")
        print(f"\n  ⚠️ Amplification at resonance = {Q:.1f}×")

        # Severity assessment
        if zeta < 0.01:
            severity = "CRITICAL"
            symbol = "⚠️⚠️⚠️"
        elif zeta < 0.02:
            severity = "SEVERE"
            symbol = "⚠️⚠️"
        elif zeta < 0.05:
            severity = "WARNING"
            symbol = "⚠️"
        elif zeta < 0.10:
            severity = "MODERATE"
            symbol = "✓"
        else:
            severity = "GOOD"
            symbol = "✓✓"

        print(f"\nSeverity: {severity} {symbol}")

        results.append({
            'Peak_Number': idx + 1,
            'Natural_Freq_Hz': fn,
            'Peak_Magnitude': Hmax,
            'f1': f1,
            'f2': f2,
            'Bandwidth_Hz': bandwidth,
            'Damping_Ratio': zeta,
            'Damping_Percent': zeta_percent,
            'Quality_Factor': Q,
            'Severity': severity
        })

    return pd.DataFrame(results)


def plot_results(df_frf, df_coh, peaks_df):
    """Create comprehensive plots"""

    print("\n" + "=" * 80)
    print("STEP 5: CREATING PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('FRF and Coherence Analysis - Axis X', fontsize=14, fontweight='bold')

    # Plot 1: FRF Magnitude
    ax1 = axes[0]
    ax1.plot(df_frf['Frequency_Hz'], df_frf['Magnitude'],
             linewidth=1.5, color='blue', label='FRF Magnitude')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (g/N)')
    ax1.set_title('FRF Magnitude')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Mark peaks
    if peaks_df is not None and not peaks_df.empty:
        for idx, row in peaks_df.iterrows():
            fn = row['Natural_Freq_Hz']
            Hmax = row['Peak_Magnitude']
            zeta = row['Damping_Percent']
            Q = row['Quality_Factor']

            ax1.plot(fn, Hmax, 'ro', markersize=10)
            ax1.annotate(f'fₙ={fn:.1f}Hz\nζ={zeta:.2f}%\nQ={Q:.1f}',
                         xy=(fn, Hmax), xytext=(10, 10),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                         fontsize=8)

    ax1.legend()

    # Plot 2: FRF Phase
    ax2 = axes[1]
    ax2.plot(df_frf['Frequency_Hz'], df_frf['Phase_deg'],
             linewidth=1.5, color='blue')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('FRF Phase')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Coherence
    ax3 = axes[2]
    ax3.plot(df_coh['Frequency_Hz'], df_coh['Coherence'],
             linewidth=1.5, color='green')
    ax3.axhline(y=0.8, color='orange', linestyle='--',
                label='Quality threshold (0.8)', linewidth=2)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Coherence')
    ax3.set_title('Measurement Coherence')
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    save_path = 'frf_coherence_analysis_X.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {save_path}")

    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("COMPLETE FRF AND COHERENCE ANALYSIS")
    print("Reading from FRF_1_X.txt and Coherence_1_X.txt")
    print("=" * 80)

    # File paths - UPDATE THESE
    frf_file = "../Datasets/FRF_1_X.txt"
    coh_file = "../Datasets/Coherence_1_X.txt"

    try:
        # Step 1: Read files
        print("\n" + "=" * 80)
        print("STEP 1: READING FILES")
        print("=" * 80)

        metadata_frf, df_frf = parse_lms_file(frf_file)
        metadata_coh, df_coh = parse_lms_file(coh_file)

        # Step 2: Calculate FRF magnitude and phase
        df_frf = calculate_frf_magnitude_step_by_step(df_frf, num_points=5)

        # Step 3: Analyze coherence
        df_coh = analyze_coherence_step_by_step(df_coh, num_points=5)

        # Step 4: Find peaks and calculate damping
        peaks_df = find_peaks_and_calculate_damping(df_frf)

        # Step 5: Plot results
        plot_results(df_frf, df_coh, peaks_df)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)

        # Save results to CSV
        df_frf.to_csv('frf_results_X.csv', index=False)
        df_coh.to_csv('coherence_results_X.csv', index=False)
        if peaks_df is not None:
            peaks_df.to_csv('peaks_analysis_X.csv', index=False)

        print("\n✓ Saved results to CSV files:")
        print("  - frf_results_X.csv")
        print("  - coherence_results_X.csv")
        if peaks_df is not None:
            print("  - peaks_analysis_X.csv")

    except FileNotFoundError as e:
        print(f"\n✗ Error: Could not find file")
        print(f"  Make sure FRF_1_X.txt and Coherence_1_X.txt are in the same directory")
        print(f"  Error details: {e}")
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback

        traceback.print_exc()