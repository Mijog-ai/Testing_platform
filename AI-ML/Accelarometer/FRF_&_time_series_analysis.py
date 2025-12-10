"""
Comprehensive Vibration Analysis System
Based on: "Vibration Analysis for Machine Monitoring and Diagnosis: A Systematic Review"
Authors: Mohamad Hazwan Mohd Ghazali and Wan Rahiman (2021)

This system implements the three main analysis domains discussed in the paper:
1. Time Domain Analysis (Peak, RMS, Crest Factor, Kurtosis) - Section 7
2. Frequency Domain Analysis (FFT, Cepstrum, Envelope, Spectrum) - Section 8
3. Time-Frequency Domain Analysis (Wavelet, STFT, PSD) - Section 9
4. AI-Based Fault Detection (Feature-based diagnosis) - Section 10

Key Features:
- Converts FRF data from frequency to time domain
- Extracts statistical features for fault detection
- Multiple frequency analysis techniques
- Time-frequency localization for transient analysis
- Automated fault detection with severity assessment
- Comprehensive visualization and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, irfft, fftfreq
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import kurtosis as scipy_kurtosis
import pywt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING - LMS TEST.LAB FORMAT PARSER
# ============================================================================

def parse_lms_file(filepath):
    """
    Parse LMS Test.Lab format FRF data file

    Returns:
    --------
    metadata : dict
        File metadata (sampling info, test conditions, etc.)
    df : pd.DataFrame
        FRF data with Frequency, Real, Imaginary, Magnitude, Phase columns
    """
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
                    test_val = parts[0].replace(',', '.')
                    float(test_val)
                    data_started = True
                    data_lines.append(parts)
                except (ValueError, IndexError):
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

    if data_lines:
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

        function_class = metadata.get('Function class', '')

        if 'FRF' in function_class:
            if len(cleaned_data[0]) >= 3:
                df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'FRF_Real', 'FRF_Imag'])
                df['FRF_Magnitude'] = np.sqrt(df['FRF_Real'] ** 2 + df['FRF_Imag'] ** 2)
                df['FRF_Phase_rad'] = np.arctan2(df['FRF_Imag'], df['FRF_Real'])
            else:
                df = pd.DataFrame(cleaned_data, columns=['Frequency_Hz', 'Value'])
        else:
            num_cols = max(len(row) for row in cleaned_data)
            col_names = ['Frequency_Hz'] + [f'Column_{i}' for i in range(1, num_cols)]
            df = pd.DataFrame(cleaned_data, columns=col_names[:num_cols])

        df = df.dropna(how='all')
    else:
        df = pd.DataFrame()

    return metadata, df


def frf_to_time_domain(df_frf, method='ifft'):
    """
    Convert FRF (Frequency Response Function) to time domain impulse response
    using Inverse Fourier Transform

    Parameters:
    -----------
    df_frf : pd.DataFrame
        FRF data with Frequency_Hz, FRF_Real, FRF_Imag columns
    method : str
        'ifft' for standard Inverse FFT
        'irfft' for real-valued output

    Returns:
    --------
    time_data : pd.DataFrame
        Time domain data with Time_s and Amplitude columns
    sampling_rate : float
        Calculated sampling rate (Hz)
    """
    if df_frf.empty:
        return pd.DataFrame(), 0

    frequencies = df_frf['Frequency_Hz'].values

    if 'FRF_Real' in df_frf.columns and 'FRF_Imag' in df_frf.columns:
        frf_complex = df_frf['FRF_Real'].values + 1j * df_frf['FRF_Imag'].values
    else:
        print("⚠ No complex FRF data. Using magnitude only.")
        if 'FRF_Magnitude' in df_frf.columns:
            frf_complex = df_frf['FRF_Magnitude'].values
        else:
            return pd.DataFrame(), 0

    df_freq = frequencies[1] - frequencies[0]
    sampling_rate = 2 * frequencies[-1]  # Nyquist theorem

    if method == 'irfft':
        time_response = irfft(frf_complex)
    else:
        time_response = ifft(frf_complex)
        time_response = np.real(time_response)

    n_points = len(time_response)
    time_duration = n_points / sampling_rate
    time_array = np.linspace(0, time_duration, n_points)

    time_data = pd.DataFrame({
        'Time_s': time_array,
        'Amplitude': time_response
    })

    return time_data, sampling_rate


# ============================================================================
# TIME DOMAIN ANALYSIS - Section 7 of Paper
# Statistical Features: Peak, RMS, Crest Factor, Kurtosis
# ============================================================================

class TimeDomainAnalyzer:
    """
    Time Domain Analysis (Paper Section 7)

    Simplest vibration analysis for machine diagnosis.
    Analyzes amplitude of vibration signal plotted against time.

    Features extracted:
    - Peak: Maximum signal amplitude (Eq. 1)
    - RMS: Root Mean Square - energy content (Eq. 2)
    - Crest Factor: Peak/RMS ratio (Eq. 3)
    - Kurtosis: Non-Gaussianity measure (impulsiveness indicator)
    - Shape Factor, Impulse Factor, Clearance Factor

    Advantages:
    + Simple and easy to implement
    + Real-time capable
    + Low computational cost
    + Effective for imbalance detection (RMS)

    Disadvantages:
    - Limited for early fault detection
    - Sensitive to noise (Peak)
    - Cannot separate frequency components
    """

    @staticmethod
    def calculate_peak(signal):
        """
        Peak Value - Maximum absolute amplitude
        Equation (1): peak = |v(t)|_max

        Use: Detects impacts, transients
        Note: Very sensitive to noise
        """
        return np.max(np.abs(signal))

    @staticmethod
    def calculate_rms(signal):
        """
        RMS (Root Mean Square) - Power content indicator
        Equation (2): RMS = sqrt(1/T * integral(v(t)^2 dt))

        Use: Detects imbalance, misalignment
        Advantage: Less sensitive to noise than Peak
        Limitation: Not significantly affected by low-intensity vibrations
        """
        return np.sqrt(np.mean(signal ** 2))

    @staticmethod
    def calculate_crest_factor(signal):
        """
        Crest Factor - Peak to RMS ratio
        Equation (3): CF = Peak / RMS

        Use: Independent of speed, reliable for impulsive signals

        Typical values:
        - Pure sine wave: 1.414 (sqrt(2))
        - Random noise: ~3.0
        - Low CF (<1.2): Distributed wear
        - High CF (>6): Impact, crack, looseness
        """
        peak = TimeDomainAnalyzer.calculate_peak(signal)
        rms = TimeDomainAnalyzer.calculate_rms(signal)
        return peak / (rms + 1e-10)

    @staticmethod
    def calculate_kurtosis(signal):
        """
        Kurtosis - Statistical measure of "tailedness"

        Use: Detects non-Gaussianity, impulsive behavior

        Physical meaning:
        - High kurtosis: Presence of transient peaks (wear indication)
        - Normal distribution: Kurtosis = 3 (Fisher: 0)
        - >3: More peaked, heavier tails (fault indication)

        Advantage: Not sensitive to speed or load
        Limitation: Only reliable with significant impulsiveness
        """
        return scipy_kurtosis(signal, fisher=True)

    @staticmethod
    def calculate_shape_factor(signal):
        """Shape Factor = RMS / Mean(|signal|)"""
        mean_abs = np.mean(np.abs(signal))
        rms = TimeDomainAnalyzer.calculate_rms(signal)
        return rms / (mean_abs + 1e-10)

    @staticmethod
    def calculate_impulse_factor(signal):
        """Impulse Factor = Peak / Mean(|signal|)"""
        peak = TimeDomainAnalyzer.calculate_peak(signal)
        mean_abs = np.mean(np.abs(signal))
        return peak / (mean_abs + 1e-10)

    @staticmethod
    def calculate_clearance_factor(signal):
        """Clearance Factor = Peak / (Mean(sqrt(|signal|)))^2"""
        peak = TimeDomainAnalyzer.calculate_peak(signal)
        mean_sqrt = np.mean(np.sqrt(np.abs(signal)))
        return peak / ((mean_sqrt ** 2) + 1e-10)

    @staticmethod
    def analyze_signal(signal):
        """
        Complete time domain analysis

        Returns dictionary with all time domain features
        """
        if isinstance(signal, pd.DataFrame):
            signal = signal['Amplitude'].values

        features = {
            # Basic statistical features (Paper Section 7.1-7.4)
            'peak': TimeDomainAnalyzer.calculate_peak(signal),
            'rms': TimeDomainAnalyzer.calculate_rms(signal),
            'crest_factor': TimeDomainAnalyzer.calculate_crest_factor(signal),
            'kurtosis': TimeDomainAnalyzer.calculate_kurtosis(signal),

            # Additional shape descriptors
            'mean': np.mean(signal),
            'std': np.std(signal),
            'variance': np.var(signal),
            'peak_to_peak': np.ptp(signal),
            'shape_factor': TimeDomainAnalyzer.calculate_shape_factor(signal),
            'impulse_factor': TimeDomainAnalyzer.calculate_impulse_factor(signal),
            'clearance_factor': TimeDomainAnalyzer.calculate_clearance_factor(signal),

            # Energy-based features
            'energy': np.sum(signal ** 2),
            'power': np.mean(signal ** 2),
        }

        return features


# ============================================================================
# FREQUENCY DOMAIN ANALYSIS - Section 8 of Paper
# FFT, Cepstrum, Envelope, Spectrum Analysis
# ============================================================================

class FrequencyDomainAnalyzer:
    """
    Frequency Domain Analysis (Paper Section 8)

    Analyzes amplitude vs frequency (not time).
    Each frequency component appears as vertical line.
    Easier to detect resonant frequencies than time domain.

    Techniques:
    1. FFT: Fast Fourier Transform (Section 8.1)
    2. Cepstrum: Detects periodic structures (Section 8.2)
    3. Envelope: For bearing fault detection (Section 8.3)
    4. Spectrum: High-resolution spectral analysis (Section 8.4)

    Advantages:
    + Easy to detect resonant frequencies
    + Separates frequency components
    + Well-established techniques

    Disadvantages:
    - Not suitable for time-varying frequencies
    - Loss of time information during conversion
    - Cannot detect transient features efficiently
    """

    @staticmethod
    def perform_fft(signal, sampling_rate):
        """
        FFT - Fast Fourier Transform (Paper Section 8.1)
        Equations (4) & (5): Fourier Transform and Inverse

        Converts time domain signal to frequency domain.

        Use: Quickest way to separate signal frequencies
        Advantage: Fast, efficient algorithm
        Limitation: Cannot investigate transient features
        """
        if isinstance(signal, pd.DataFrame):
            signal = signal['Amplitude'].values

        N = len(signal)
        fft_values = fft(signal)
        fft_freq = fftfreq(N, 1/sampling_rate)

        # Keep only positive frequencies
        positive_idx = fft_freq > 0

        result = pd.DataFrame({
            'Frequency_Hz': fft_freq[positive_idx],
            'Magnitude': np.abs(fft_values[positive_idx]),
            'Phase_rad': np.angle(fft_values[positive_idx]),
            'Real': np.real(fft_values[positive_idx]),
            'Imag': np.imag(fft_values[positive_idx])
        })

        return result

    @staticmethod
    def perform_cepstrum(signal, sampling_rate):
        """
        Cepstrum Analysis (Paper Section 8.2)

        Power spectrum of logarithm of power spectrum.
        Detects periodic structures (harmonics, sidebands, echoes).

        Use: Bearing and gear fault detection (harmonically related frequencies)
        Domain: Quefrency (time-like domain for cepstrum)

        Advantage: Detects patterns in frequency spectrum
        Limitation: Insensitive to gear cracks (Dalpiaz et al.)
        """
        if isinstance(signal, pd.DataFrame):
            signal = signal['Amplitude'].values

        # Power spectrum
        fft_vals = fft(signal)
        power_spectrum = np.abs(fft_vals) ** 2 + 1e-10

        # Cepstrum = IFFT(log(power spectrum))
        cepstrum = np.real(ifft(np.log(power_spectrum)))

        N = len(cepstrum)
        quefrency = np.arange(N) / sampling_rate

        result = pd.DataFrame({
            'Quefrency_s': quefrency[:N//2],
            'Cepstrum': cepstrum[:N//2]
        })

        return result

    @staticmethod
    def perform_envelope_analysis(signal, sampling_rate, lowcut=None, highcut=None):
        """
        Envelope Analysis (Paper Section 8.3)
        Also known as: Amplitude Demodulation, Demodulated Resonance Analysis

        Separates low-frequency signal from background noise.
        Extracts signal envelope using Hilbert transform.

        Use: Rolling element bearing and low-speed machine diagnosis
        Advantage: Early detection of bearing problems

        Challenge: Determining best frequency band for enveloping
        Requirement: Sharp filter + precise frequency band specification
        """
        if isinstance(signal, pd.DataFrame):
            sig = signal['Amplitude'].values
            time = signal['Time_s'].values
        else:
            sig = signal
            time = np.arange(len(sig)) / sampling_rate

        # Optional bandpass filter
        if lowcut and highcut:
            nyquist = sampling_rate / 2
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            sig = signal.filtfilt(b, a, sig)

        sig_array = sig.values if isinstance(sig, pd.DataFrame) else sig
        analytic_sig = hilbert(sig_array)
        envelope = np.abs(analytic_sig)

        # FFT of envelope
        env_fft = fft(envelope)
        env_freq = fftfreq(len(envelope), 1/sampling_rate)
        pos_idx = env_freq > 0

        envelope_data = pd.DataFrame({
            'Time_s': time,
            'Signal': sig,
            'Envelope': envelope
        })

        envelope_spectrum = pd.DataFrame({
            'Frequency_Hz': env_freq[pos_idx],
            'Magnitude': np.abs(env_fft[pos_idx])
        })

        return envelope_data, envelope_spectrum

    @staticmethod
    def calculate_psd(signal, sampling_rate, nperseg=256):
        """
        Power Spectral Density (Paper Equation 9 & 10)

        Measures amplitude of oscillatory signals.
        Determines energy strength at different frequencies.

        Use: Analyzes faulty frequency bands without slip variation
        Advantage: Very little processing power, direct FFT calculation
        Limitation: Frequency resolution affected by window size
        """
        if isinstance(signal, pd.DataFrame):
            signal = signal['Amplitude'].values

        from scipy.signal import welch

        sig_array = signal if isinstance(signal, np.ndarray) else signal['Amplitude'].values

        frequencies, psd = welch(
            sig_array,
            fs=sampling_rate,
            nperseg=min(len(sig_array) // 8, nperseg)
        )

        result = pd.DataFrame({
            'Frequency_Hz': frequencies,
            'PSD': psd
        })

        return result


# ============================================================================
# TIME-FREQUENCY ANALYSIS - Section 9 of Paper
# Wavelet Transform, STFT, Combined Time-Frequency Methods
# ============================================================================

class TimeFrequencyAnalyzer:
    """
    Time-Frequency Domain Analysis (Paper Section 9)

    Integrates time and frequency domains.
    Determines signal frequency components AND their time-variant features simultaneously.

    Critical for: Non-stationary signals (frequencies vary over time)

    Techniques:
    1. Wavelet Transform (WT) - Section 9.1
    2. STFT (Short-Time Fourier Transform) - Section 9.4
    3. HHT (Hilbert-Huang Transform) - Section 9.3
    4. PSD (Power Spectral Density) - Section 9.5

    Advantage: Detects local features in time AND frequency
    Use case: When frequencies change over time (transient analysis)
    """

    @staticmethod
    def perform_cwt(signal, sampling_rate, wavelet='morl', scales=None):
        """
        Continuous Wavelet Transform - CWT (Paper Section 9.1)

        Decomposes signal into wavelets (local time functions with frequency content).
        Uses wavelets instead of sinusoidal functions as basis.

        Advantage: Superior time localization at high frequencies vs STFT
        Use: Non-stationary signals, transient signal analysis

        Challenge: Selecting appropriate wavelet basis for signal structure

        Types:
        - CWT: Arbitrary scaling factor (finer scale than DWT)
        - DWT: Power-of-two scaling (more efficient)
        - WPT: Wavelet Packet Transform (finer frequency resolution)
        """
        if isinstance(signal, pd.DataFrame):
            signal = signal['Amplitude'].values

        if scales is None:
            scales = np.arange(1, min(128, len(signal)//4))

        # Perform CWT
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet,
                                            sampling_period=1/sampling_rate)

        return coefficients, frequencies, scales

    @staticmethod
    def perform_dwt(signal, wavelet='db4', level=None):
        """
        Discrete Wavelet Transform - DWT

        More efficient than CWT computationally.
        Power-of-two scaling factor.

        Use: Feature extraction, denoising
        Advantage: Computational efficiency
        """
        if isinstance(signal, pd.DataFrame):
            signal = signal['Amplitude'].values

        if level is None:
            level = min(5, pywt.dwt_max_level(len(signal), wavelet))

        # Multilevel DWT decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        return coeffs

    @staticmethod
    def perform_stft(signal, sampling_rate, nperseg=256, noverlap=None):
        """
        Short-Time Fourier Transform - STFT (Paper Section 9.4)
        Equation (8) from paper

        Breaks signal into time segments by windowing, then applies FT to each segment.

        Resolution trade-off:
        - Large window: Better frequency accuracy
        - Small window: Better time accuracy

        Advantage: Straightforward, low computational complexity
        Disadvantage: Cannot achieve high resolution in BOTH time and frequency

        Good for: Extracting narrowband frequency in non-stationary/noisy signals
        """
        from scipy.signal import stft

        if isinstance(signal, pd.DataFrame):
            sig_array = signal['Amplitude'].values
        else:
            sig_array = signal

        if noverlap is None:
            noverlap = nperseg // 2

        # Compute STFT
        f, t, Zxx = stft(sig_array, fs=sampling_rate,
                         nperseg=nperseg, noverlap=noverlap)

        return f, t, Zxx


# ============================================================================
# FAULT DETECTION SYSTEM - Section 10 of Paper (AI-Based Methods)
# ============================================================================

class FaultDetector:
    """
    Fault Detection and Diagnosis (Paper Section 10)

    AI-based methods discussed in paper:
    - SVM (Support Vector Machine) - Section 10.1 [Most widely used]
    - Neural Networks (NN) - Section 10.2
    - Fuzzy Logic - Section 10.3
    - Genetic Algorithm (GA) - Section 10.4

    This implementation uses threshold-based logic (simplified AI approach).
    For production systems, implement machine learning classifiers.

    Paper finding: "AI-based techniques contribute about 57% of overall
    vibration analysis methods in machine diagnosis and monitoring"
    """

    def __init__(self):
        # Threshold values based on typical industrial machinery
        # Should be calibrated for specific application
        self.thresholds = {
            'rms_multiplier': 2.0,        # RMS > 2x baseline → fault
            'peak_multiplier': 3.0,        # Peak > 3x baseline → fault
            'crest_factor_min': 1.2,       # CF < 1.2 → wear
            'crest_factor_max': 6.0,       # CF > 6.0 → impact/crack
            'kurtosis_threshold': 3.0,     # Kurt > 3 → impulsive fault
            'shape_factor_threshold': 1.3,
        }

        self.baseline = None
        self.fault_history = []

    def set_baseline(self, healthy_features):
        """
        Set baseline from healthy machine operation

        Baseline should be established during:
        - Commissioning of new equipment
        - After maintenance/repair
        - During known good operating conditions
        """
        self.baseline = healthy_features.copy()
        print("✓ Baseline features established")

    def diagnose(self, current_features):
        """
        Diagnose machine condition based on feature comparison

        Fault types detected:
        1. Imbalance (high RMS)
        2. Misalignment (high RMS, specific frequency)
        3. Bearing faults (high kurtosis, envelope analysis)
        4. Gear faults (cepstrum analysis, high crest factor)
        5. Looseness (high peak, impact)
        6. Wear (low crest factor)

        Returns:
        --------
        diagnosis_report : dict
            Status, detected faults, severity, feature ratios
        """
        if self.baseline is None:
            return {
                'status': 'NO_BASELINE',
                'severity': 'UNKNOWN',
                'message': 'Baseline not established. Run healthy machine first.',
                'faults': []
            }

        faults = []
        severity = 'HEALTHY'
        feature_ratios = {}

        # RMS Analysis - Imbalance/Misalignment Detection
        rms_ratio = current_features['rms'] / self.baseline['rms']
        feature_ratios['rms_ratio'] = rms_ratio

        if rms_ratio > self.thresholds['rms_multiplier']:
            faults.append({
                'type': 'IMBALANCE/MISALIGNMENT',
                'description': f'RMS is {rms_ratio:.2f}x baseline',
                'indicator': 'High RMS value',
                'severity': 'WARNING'
            })
            severity = 'WARNING'

        # Peak Analysis - Impact/Looseness Detection
        peak_ratio = current_features['peak'] / self.baseline['peak']
        feature_ratios['peak_ratio'] = peak_ratio

        if peak_ratio > self.thresholds['peak_multiplier']:
            faults.append({
                'type': 'IMPACT/LOOSENESS',
                'description': f'Peak is {peak_ratio:.2f}x baseline',
                'indicator': 'High peak amplitude',
                'severity': 'WARNING'
            })
            if severity == 'HEALTHY':
                severity = 'WARNING'

        # Crest Factor Analysis - Wear/Impact Detection
        cf = current_features['crest_factor']
        feature_ratios['crest_factor'] = cf

        if cf < self.thresholds['crest_factor_min']:
            faults.append({
                'type': 'DISTRIBUTED_WEAR',
                'description': f'Crest Factor = {cf:.2f} (< {self.thresholds["crest_factor_min"]})',
                'indicator': 'Low crest factor indicates surface degradation',
                'severity': 'CRITICAL'
            })
            severity = 'CRITICAL'

        elif cf > self.thresholds['crest_factor_max']:
            faults.append({
                'type': 'IMPACT/CRACK',
                'description': f'Crest Factor = {cf:.2f} (> {self.thresholds["crest_factor_max"]})',
                'indicator': 'High crest factor indicates sharp impacts',
                'severity': 'CRITICAL'
            })
            severity = 'CRITICAL'

        # Kurtosis Analysis - Bearing/Gear Fault Detection
        kurt = current_features['kurtosis']
        feature_ratios['kurtosis'] = kurt

        if kurt > self.thresholds['kurtosis_threshold']:
            faults.append({
                'type': 'BEARING/GEAR_FAULT',
                'description': f'Kurtosis = {kurt:.2f} (> {self.thresholds["kurtosis_threshold"]})',
                'indicator': 'High kurtosis indicates impulsive behavior (bearing spalls, gear defects)',
                'severity': 'WARNING' if severity == 'HEALTHY' else severity
            })
            if severity == 'HEALTHY':
                severity = 'WARNING'

        # Impulse Factor Analysis
        if 'impulse_factor' in current_features:
            impulse_ratio = current_features['impulse_factor'] / self.baseline.get('impulse_factor', 1.0)
            feature_ratios['impulse_factor_ratio'] = impulse_ratio

            if impulse_ratio > 2.0:
                faults.append({
                    'type': 'INTERMITTENT_IMPACT',
                    'description': f'Impulse Factor ratio = {impulse_ratio:.2f}',
                    'indicator': 'Intermittent impacts detected',
                    'severity': 'WARNING'
                })

        # Create diagnosis report
        diagnosis_report = {
            'status': severity,
            'timestamp': pd.Timestamp.now(),
            'faults': faults if faults else [{'type': 'NONE', 'description': 'No faults detected', 'severity': 'HEALTHY'}],
            'feature_ratios': feature_ratios,
            'recommendations': self._generate_recommendations(faults, severity)
        }

        # Store in history
        self.fault_history.append(diagnosis_report)

        return diagnosis_report

    def _generate_recommendations(self, faults, severity):
        """Generate maintenance recommendations based on detected faults"""
        if severity == 'HEALTHY':
            return ['Continue normal operation', 'Maintain monitoring schedule']

        recommendations = []

        fault_types = [f['type'] for f in faults]

        if 'IMBALANCE/MISALIGNMENT' in fault_types:
            recommendations.append('Schedule balancing or alignment correction')
            recommendations.append('Check mounting bolts and foundation')

        if 'BEARING/GEAR_FAULT' in fault_types:
            recommendations.append('Inspect bearings and gears for damage')
            recommendations.append('Consider lubrication check')
            recommendations.append('Plan for component replacement')

        if 'DISTRIBUTED_WEAR' in fault_types:
            recommendations.append('CRITICAL: Severe wear detected')
            recommendations.append('Consider immediate shutdown and inspection')
            recommendations.append('Replace worn components')

        if 'IMPACT/CRACK' in fault_types:
            recommendations.append('CRITICAL: Sharp impact detected - possible crack')
            recommendations.append('Immediate inspection recommended')
            recommendations.append('Check for structural damage')

        if severity == 'CRITICAL':
            recommendations.insert(0, '⚠ CRITICAL CONDITION - IMMEDIATE ACTION REQUIRED')

        return recommendations

    def print_diagnosis(self, diagnosis):
        """Pretty print diagnosis report"""
        print("\n" + "="*80)
        print("FAULT DIAGNOSIS REPORT")
        print("="*80)
        print(f"Timestamp: {diagnosis['timestamp']}")
        print(f"Overall Status: {diagnosis['status']}")
        print("\n" + "-"*80)
        print("DETECTED FAULTS:")
        print("-"*80)

        for fault in diagnosis['faults']:
            print(f"\nFault Type: {fault['type']}")
            print(f"Description: {fault['description']}")
            if 'indicator' in fault:
                print(f"Indicator: {fault['indicator']}")
            print(f"Severity: {fault.get('severity', 'N/A')}")

        print("\n" + "-"*80)
        print("FEATURE RATIOS (Current / Baseline):")
        print("-"*80)
        for key, value in diagnosis['feature_ratios'].items():
            print(f"{key:25s}: {value:.3f}")

        print("\n" + "-"*80)
        print("RECOMMENDATIONS:")
        print("-"*80)
        for i, rec in enumerate(diagnosis['recommendations'], 1):
            print(f"{i}. {rec}")

        print("="*80 + "\n")


# ============================================================================
# COMPREHENSIVE ANALYSIS ORCHESTRATOR
# ============================================================================

class VibrationAnalysisSystem:
    """
    Complete Vibration Analysis System

    Integrates all analysis techniques from the paper into unified workflow:
    1. Data Loading (FRF files)
    2. Time Domain Conversion
    3. Multi-Domain Analysis (Time, Frequency, Time-Frequency)
    4. Fault Detection
    5. Visualization
    6. Reporting
    """

    def __init__(self):
        self.time_analyzer = TimeDomainAnalyzer()
        self.freq_analyzer = FrequencyDomainAnalyzer()
        self.tf_analyzer = TimeFrequencyAnalyzer()
        self.fault_detector = FaultDetector()

        self.results = {}

    def analyze_complete(self, frf_file, axis_name='X', output_dir='./results'):
        """
        Complete analysis pipeline

        Parameters:
        -----------
        frf_file : str
            Path to FRF data file
        axis_name : str
            Axis identifier (X, Y, Z)
        output_dir : str
            Output directory for results

        Returns:
        --------
        results : dict
            Complete analysis results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print(f"COMPREHENSIVE VIBRATION ANALYSIS - AXIS {axis_name}")
        print("="*80)

        # Step 1: Load FRF data
        print("\n[Step 1/8] Loading FRF data...")
        metadata, df_frf = parse_lms_file(frf_file)

        if df_frf.empty:
            print("❌ Error: No data loaded")
            return None

        print(f"✓ Loaded {len(df_frf)} frequency points")
        print(f"  Frequency range: {df_frf['Frequency_Hz'].min():.1f} - {df_frf['Frequency_Hz'].max():.1f} Hz")

        # Step 2: Convert to time domain
        print("\n[Step 2/8] Converting to time domain (IFFT)...")
        time_data, sampling_rate = frf_to_time_domain(df_frf)

        if time_data.empty:
            print("❌ Error: Conversion failed")
            return None

        print(f"✓ Generated {len(time_data)} time points")
        print(f"  Sampling rate: {sampling_rate:.2f} Hz")
        print(f"  Duration: {time_data['Time_s'].max():.4f} s")

        # Step 3: Time domain analysis
        print("\n[Step 3/8] Time Domain Analysis (Statistical Features)...")
        time_features = self.time_analyzer.analyze_signal(time_data)

        print("  Time Domain Features:")
        for key in ['peak', 'rms', 'crest_factor', 'kurtosis']:
            print(f"    {key:15s}: {time_features[key]:.6f}")

        # Step 4: Frequency domain analysis
        print("\n[Step 4/8] Frequency Domain Analysis...")

        fft_result = self.freq_analyzer.perform_fft(time_data, sampling_rate)
        print(f"  ✓ FFT: {len(fft_result)} frequency points")

        cepstrum_result = self.freq_analyzer.perform_cepstrum(time_data, sampling_rate)
        print(f"  ✓ Cepstrum: {len(cepstrum_result)} quefrency points")

        envelope_data, envelope_spectrum = self.freq_analyzer.perform_envelope_analysis(time_data, sampling_rate)
        print(f"  ✓ Envelope Analysis: {len(envelope_spectrum)} spectrum points")

        psd_result = self.freq_analyzer.calculate_psd(time_data, sampling_rate)
        print(f"  ✓ Power Spectral Density: {len(psd_result)} points")

        # Step 5: Time-frequency analysis
        print("\n[Step 5/8] Time-Frequency Analysis...")

        stft_freq, stft_time, stft_zxx = self.tf_analyzer.perform_stft(time_data, sampling_rate)
        print(f"  ✓ STFT: {len(stft_freq)} freq × {len(stft_time)} time points")

        # Step 6: Fault detection
        print("\n[Step 6/8] Fault Detection & Diagnosis...")
        diagnosis = self.fault_detector.diagnose(time_features)

        if diagnosis['status'] == 'NO_BASELINE':
            print(f"  ⚠ {diagnosis['message']}")
        else:
            print(f"  Status: {diagnosis['status']}")
            print(f"  Detected {len([f for f in diagnosis['faults'] if f['type'] != 'NONE'])} fault(s)")

        # Step 7: Visualization
        print("\n[Step 7/8] Generating visualizations...")

        # Store results
        results = {
            'axis': axis_name,
            'frf_data': df_frf,
            'time_data': time_data,
            'sampling_rate': sampling_rate,
            'time_features': time_features,
            'fft': fft_result,
            'cepstrum': cepstrum_result,
            'envelope_data': envelope_data,
            'envelope_spectrum': envelope_spectrum,
            'psd': psd_result,
            'stft': (stft_freq, stft_time, stft_zxx),
            'diagnosis': diagnosis
        }

        self._create_visualizations(results, output_dir)

        # Step 8: Export results
        print("\n[Step 8/8] Exporting results...")
        self._export_results(results, output_dir)

        self.results[axis_name] = results

        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {output_dir}/")

        return results

    def _create_visualizations(self, results, output_dir):
        """Create comprehensive visualization plots"""

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        axis_name = results['axis']

        # Row 1: FRF Data
        # Plot 1: FRF Magnitude
        ax1 = fig.add_subplot(gs[0, 0])
        df_frf = results['frf_data']
        if 'FRF_Magnitude' in df_frf.columns:
            ax1.semilogy(df_frf['Frequency_Hz'], df_frf['FRF_Magnitude'], 'b-', lw=1)
            ax1.set_xlabel('Frequency (Hz)', fontsize=10)
            ax1.set_ylabel('Magnitude (g/N)', fontsize=10)
            ax1.set_title('FRF Magnitude', fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # Plot 2: FRF Phase
        ax2 = fig.add_subplot(gs[0, 1])
        if 'FRF_Phase_rad' in df_frf.columns:
            ax2.plot(df_frf['Frequency_Hz'], np.degrees(df_frf['FRF_Phase_rad']), 'b-', lw=1)
            ax2.set_xlabel('Frequency (Hz)', fontsize=10)
            ax2.set_ylabel('Phase (°)', fontsize=10)
            ax2.set_title('FRF Phase', fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Time Domain Signal
        ax3 = fig.add_subplot(gs[0, 2])
        time_data = results['time_data']
        ax3.plot(time_data['Time_s'], time_data['Amplitude'], 'r-', lw=1)
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Amplitude', fontsize=10)
        ax3.set_title('Impulse Response (Time Domain)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='k', lw=0.5, ls='--')

        # Row 2: Envelope & FFT
        # Plot 4: Envelope
        ax4 = fig.add_subplot(gs[1, 0])
        env_data = results['envelope_data']
        ax4.plot(env_data['Time_s'], env_data['Signal'], 'r-', lw=0.8, alpha=0.5, label='Signal')
        ax4.plot(env_data['Time_s'], env_data['Envelope'], 'darkred', lw=2, label='Envelope')
        ax4.plot(env_data['Time_s'], -env_data['Envelope'], 'darkred', lw=2)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Amplitude', fontsize=10)
        ax4.set_title('Signal with Envelope', fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Plot 5: FFT Magnitude
        ax5 = fig.add_subplot(gs[1, 1])
        fft_result = results['fft']
        ax5.semilogy(fft_result['Frequency_Hz'], fft_result['Magnitude'], 'b-', lw=1)
        ax5.set_xlabel('Frequency (Hz)', fontsize=10)
        ax5.set_ylabel('Magnitude', fontsize=10)
        ax5.set_title('FFT Spectrum', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Plot 6: PSD
        ax6 = fig.add_subplot(gs[1, 2])
        psd_result = results['psd']
        ax6.semilogy(psd_result['Frequency_Hz'], psd_result['PSD'], 'g-', lw=1)
        ax6.set_xlabel('Frequency (Hz)', fontsize=10)
        ax6.set_ylabel('PSD (Power/Hz)', fontsize=10)
        ax6.set_title('Power Spectral Density', fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # Row 3: Cepstrum, Envelope Spectrum, STFT
        # Plot 7: Cepstrum
        ax7 = fig.add_subplot(gs[2, 0])
        cepstrum = results['cepstrum']
        mask = cepstrum['Quefrency_s'] < 0.1
        ax7.plot(cepstrum[mask]['Quefrency_s'], np.abs(cepstrum[mask]['Cepstrum']), 'm-', lw=1)
        ax7.set_xlabel('Quefrency (s)', fontsize=10)
        ax7.set_ylabel('Cepstrum', fontsize=10)
        ax7.set_title('Cepstrum Analysis', fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # Plot 8: Envelope Spectrum
        ax8 = fig.add_subplot(gs[2, 1])
        env_spec = results['envelope_spectrum']
        max_f = results['sampling_rate'] / 4
        mask = env_spec['Frequency_Hz'] < max_f
        ax8.semilogy(env_spec[mask]['Frequency_Hz'], env_spec[mask]['Magnitude'], 'darkred', lw=1)
        ax8.set_xlabel('Frequency (Hz)', fontsize=10)
        ax8.set_ylabel('Magnitude', fontsize=10)
        ax8.set_title('Envelope Spectrum', fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # Plot 9: STFT Spectrogram
        ax9 = fig.add_subplot(gs[2, 2])
        stft_f, stft_t, stft_z = results['stft']
        pcm = ax9.pcolormesh(stft_t, stft_f, np.abs(stft_z), shading='gouraud', cmap='viridis')
        ax9.set_xlabel('Time (s)', fontsize=10)
        ax9.set_ylabel('Frequency (Hz)', fontsize=10)
        ax9.set_title('STFT Spectrogram', fontweight='bold')
        cbar = plt.colorbar(pcm, ax=ax9)
        cbar.set_label('Magnitude', fontsize=9)

        # Row 4: Feature Summary and Diagnosis
        # Plot 10-11: Feature Bars
        ax10 = fig.add_subplot(gs[3, :2])
        features = results['time_features']

        key_features = ['peak', 'rms', 'crest_factor', 'kurtosis',
                       'shape_factor', 'impulse_factor']
        values = [features[k] for k in key_features]

        bars = ax10.bar(range(len(key_features)), values, color='steelblue', edgecolor='navy', alpha=0.7)
        ax10.set_xticks(range(len(key_features)))
        ax10.set_xticklabels([k.replace('_', '\n') for k in key_features], fontsize=9)
        ax10.set_ylabel('Value', fontsize=10)
        ax10.set_title('Time Domain Features', fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2, height,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 12: Diagnosis Summary
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.axis('off')

        diagnosis = results['diagnosis']
        status_colors = {
            'HEALTHY': 'green',
            'WARNING': 'orange',
            'CRITICAL': 'red',
            'NO_BASELINE': 'blue'
        }

        summary = f"DIAGNOSIS\n{'─'*30}\n"
        summary += f"Status: {diagnosis['status']}\n\n"

        if diagnosis['status'] != 'NO_BASELINE':
            summary += "Detected Faults:\n"
            for fault in diagnosis['faults'][:3]:  # Show first 3
                summary += f"• {fault['type']}\n"
        else:
            summary += diagnosis['message']

        text_color = status_colors.get(diagnosis['status'], 'black')
        ax12.text(0.1, 0.5, summary, fontsize=9, family='monospace',
                 va='center', color=text_color,
                 bbox=dict(boxstyle='round', fc='wheat', alpha=0.3))

        fig.suptitle(f'Comprehensive Vibration Analysis - Axis {axis_name}',
                    fontsize=16, fontweight='bold')

        # Save
        save_path = f"{output_dir}/analysis_{axis_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualization: {save_path}")
        plt.close(fig)

    def _export_results(self, results, output_dir):
        """Export results to CSV files"""
        axis = results['axis']

        # Time features
        pd.DataFrame([results['time_features']]).to_csv(
            f"{output_dir}/features_time_{axis}.csv", index=False)

        # Time signal
        results['time_data'].to_csv(
            f"{output_dir}/signal_time_{axis}.csv", index=False)

        # FFT
        results['fft'].to_csv(
            f"{output_dir}/spectrum_fft_{axis}.csv", index=False)

        # PSD
        results['psd'].to_csv(
            f"{output_dir}/spectrum_psd_{axis}.csv", index=False)

        # Envelope spectrum
        results['envelope_spectrum'].to_csv(
            f"{output_dir}/spectrum_envelope_{axis}.csv", index=False)

        # Diagnosis
        if results['diagnosis']['status'] != 'NO_BASELINE':
            diag_data = {
                'axis': axis,
                'status': results['diagnosis']['status'],
                'timestamp': results['diagnosis']['timestamp'],
                **results['diagnosis']['feature_ratios']
            }
            pd.DataFrame([diag_data]).to_csv(
                f"{output_dir}/diagnosis_{axis}.csv", index=False)

        print(f"  ✓ Exported CSV files for axis {axis}")

    def set_baseline(self, frf_file, axis_name='Baseline'):
        """Set baseline from healthy machine data"""
        print(f"\n{'='*80}")
        print(f"SETTING BASELINE FROM: {frf_file}")
        print(f"{'='*80}")

        metadata, df_frf = parse_lms_file(frf_file)
        time_data, _ = frf_to_time_domain(df_frf)

        if time_data.empty:
            print("❌ Error: Could not load baseline data")
            return False

        baseline_features = self.time_analyzer.analyze_signal(time_data)
        self.fault_detector.set_baseline(baseline_features)

        print("\nBaseline Features:")
        for key in ['peak', 'rms', 'crest_factor', 'kurtosis']:
            print(f"  {key:15s}: {baseline_features[key]:.6f}")

        print(f"\n{'='*80}\n")
        return True


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*80)
    print("COMPREHENSIVE VIBRATION ANALYSIS SYSTEM")
    print("Implementation of: 'Vibration Analysis for Machine Monitoring")
    print("                   and Diagnosis: A Systematic Review' (2021)")
    print("="*80)

    # Initialize system
    system = VibrationAnalysisSystem()

    # Configuration
    DATA_DIR = "../Datasets"
    OUTPUT_DIR = "./vibration_analysis_results"

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Optional: Set baseline from healthy machine
    # baseline_file = f"{DATA_DIR}/FRF_Healthy_X.txt"
    # if os.path.exists(baseline_file):
    #     system.set_baseline(baseline_file)

    # Analyze each axis
    axes = ['X', 'Y', 'Z']

    for axis in axes:
        frf_file = f"{DATA_DIR}/FRF_128_{axis}.txt"

        if not os.path.exists(frf_file):
            print(f"\n⚠ Skipping {axis}: File not found")
            continue

        try:
            results = system.analyze_complete(frf_file, axis, OUTPUT_DIR)

            # Print diagnosis
            if results and results['diagnosis']['status'] != 'NO_BASELINE':
                system.fault_detector.print_diagnosis(results['diagnosis'])

        except Exception as e:
            print(f"\n❌ Error processing axis {axis}: {e}")
            import traceback
            traceback.print_exc()

    # Multi-axis comparison
    if len(system.results) > 0:
        print("\n" + "="*80)
        print("MULTI-AXIS COMPARISON")
        print("="*80)

        comparison = []
        for axis, res in system.results.items():
            feat = res['time_features']
            comparison.append({
                'Axis': axis,
                'RMS': feat['rms'],
                'Peak': feat['peak'],
                'Crest Factor': feat['crest_factor'],
                'Kurtosis': feat['kurtosis'],
                'Status': res['diagnosis']['status']
            })

        comp_df = pd.DataFrame(comparison)
        print("\n", comp_df.to_string(index=False))

        comp_df.to_csv(f"{OUTPUT_DIR}/comparison_summary.csv", index=False)
        print(f"\n✓ Comparison saved: {OUTPUT_DIR}/comparison_summary.csv")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • analysis_X/Y/Z.png          - Comprehensive visualizations")
    print("  • features_time_X/Y/Z.csv     - Time domain features")
    print("  • signal_time_X/Y/Z.csv       - Time signals")
    print("  • spectrum_fft_X/Y/Z.csv      - FFT spectra")
    print("  • spectrum_psd_X/Y/Z.csv      - Power spectral density")
    print("  • spectrum_envelope_X/Y/Z.csv - Envelope spectra")
    print("  • diagnosis_X/Y/Z.csv         - Fault diagnosis reports")
    print("  • comparison_summary.csv      - Multi-axis comparison")
    print("\n" + "="*80 + "\n")