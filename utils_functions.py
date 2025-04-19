from scipy.signal import freqz, spectrogram, filtfilt, sosfreqz
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import math

def plot_graph_FIR(fir_coeff, filtered_data, original_data, fs, filter_name: str):
    w, h = freqz(fir_coeff, fs=fs)
    
    f, t, Sxx = spectrogram(filtered_data, fs)
    
    freqs_raw,amp_raw = compute_fourier(original_data,fs)

    freqs_filt,amp_filt = compute_fourier(filtered_data,fs)    

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    
    
    ax[0].plot(w, 20 * np.log10(np.abs(h)), linewidth=1)
    ax[0].set_title("Magnitude Response - " + filter_name)
    ax[0].set_xlabel(r'$f\,[Hz]$', fontsize=12)
    ax[0].set_ylabel(r'$A\,(dB)$', fontsize=12)
    ax[0].grid(True)
    
    pcm = ax[1].pcolormesh(t, f, np.log10(np.abs(Sxx)), shading='gouraud', cmap='jet', rasterized=True)
    ax[1].set_title("Spektrogram filtrováného signálu - " + filter_name)
    ax[1].set_xlabel(r'$t\,[s]$', fontsize=12)
    ax[1].set_ylabel(r'$f\,[Hz]$', fontsize=12)
    fig.colorbar(pcm, ax=ax[1], label="Log Power")
    
    ax[2].plot(freqs_raw, amp_raw, label="Před filtrací", linewidth=1)
    ax[2].plot(freqs_filt, amp_filt, label="Po filtraci", linewidth=1)
    ax[2].set_title("Amplitudově-frekvenční charakteristika - " + filter_name)
    ax[2].set_xlabel(r'$f\,[Hz]$', fontsize=12)
    ax[2].set_ylabel(r'$|A|$', fontsize=12)
    ax[2].grid(True)
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()

def custom_FIR(order, band, f_parasite, data, fs, plot=False):
    h = list()
    f_h = f_parasite+band
    f_d = f_parasite-band

    omega_h = (2*math.pi*f_h)/fs
    omega_d = (2*math.pi*f_d)/fs

    for n in range(order):
        if n==0:
            h.append(1-1/math.pi*(omega_h-omega_d))
        else:
            h.append((math.sin(omega_d*n)-math.sin(omega_h*n))/(math.pi*n))

    h = h[::-1][:-1] + h # je nutné zajistit, aby byl filtr symetrický

    # pro názornost přidána podmínka, že h je symetrické
    h = np.array(h)
    if np.allclose(h, h[::-1], atol=1e-12):
        print(f"[OK] h je symetrické, délka {len(h)}")
    else:
        print(f"[CHYBA] nesymetrie")

    w_freq, h_freq = freqz(h, fs=fs)

    if plot:
        _, ax = plt.subplots(figsize=(8, 6))
        ax[0].plot(w_freq, 20 * np.log10(np.abs(h_freq)), linewidth=1)
        ax[0].set_title("Magnitude Response - custom filter")
        ax[0].set_xlabel(r'$f\,[Hz]$', fontsize=12)
        ax[0].set_ylabel(r'$A\,(dB)$', fontsize=12)
        ax[0].grid(True)
        
        plt.tight_layout()
        plt.show()
        

    filtered_data = filtfilt(h,[1.0],data)

    return (filtered_data,h)


def plot_graph_IIR(b,a, filtered_data, original_data, fs, filter_name: str):
    w, h = freqz(b,a, fs=fs)
    
    f, t, Sxx = spectrogram(filtered_data, fs)

    freqs_raw,amp_raw = compute_fourier(original_data,fs)

    freqs_filt,amp_filt = compute_fourier(filtered_data,fs)

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    
    ax[0].plot(w, 20 * np.log10(np.abs(h)), linewidth=1)
    ax[0].set_title("Magnitude Response - " + filter_name)
    ax[0].set_xlabel(r'$f\,[Hz]$', fontsize=12)
    ax[0].set_ylabel(r'$A\,(dB)$', fontsize=12)
    ax[0].grid(True)

    pcm = ax[1].pcolormesh(t, f, np.log10(np.abs(Sxx)), shading='gouraud', cmap='jet', rasterized=True)
    ax[1].set_title("Spektrogram filtrováného signálu - " + filter_name)
    ax[1].set_xlabel(r'$t\,[s]$', fontsize=12)
    ax[1].set_ylabel(r'$f\,[Hz]$', fontsize=12)
    fig.colorbar(pcm, ax=ax[1], label="Log Power")
    
    ax[2].plot(freqs_raw, amp_raw, label="Před filtrací", linewidth=1)
    ax[2].plot(freqs_filt, amp_filt, label="Po filtraci", linewidth=1)
    ax[2].set_title("Amplitudově-frekvenční charakteristika - " + filter_name)
    ax[2].set_xlabel(r'$f\,[Hz]$', fontsize=12)
    ax[2].set_ylabel(r'$|A|$', fontsize=12)
    ax[2].grid(True)
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()
    

def plot_graph_IIR_sos(sos, filtered_data, original_data, fs, filter_name: str):
    w, h = sosfreqz(sos, fs=fs)
    
    f, t, Sxx = spectrogram(filtered_data, fs)
    
    freqs_raw,amp_raw = compute_fourier(original_data,fs)

    freqs_filt,amp_filt = compute_fourier(filtered_data,fs)
    
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    
    ax[0].plot(w, 20 * np.log10(np.abs(h)), linewidth=1)
    ax[0].set_title("Magnitude Response - " + filter_name)
    ax[0].set_xlabel("Frekvence (Hz)")
    ax[0].set_ylabel("Amplituda (dB)")
    ax[0].grid(True)
    
    
    pcm = ax[1].pcolormesh(t, f, np.log10(np.abs(Sxx)), shading='gouraud', cmap='jet', rasterized=True)
    ax[1].set_title("Spektrogram filtrováného signálu - " + filter_name)
    ax[1].set_xlabel(r'$t\,[s]$', fontsize=12)
    ax[1].set_ylabel(r'$f\,[Hz]$', fontsize=12)
    fig.colorbar(pcm, ax=ax[1], label="Intenzita (dB)")
    
    ax[2].plot(freqs_raw, amp_raw, label="Před filtrací")
    ax[2].plot(freqs_filt, amp_filt, label="Po filtraci")
    ax[2].set_title("Amplitudově-frekvenční charakteristika - " + filter_name)
    ax[2].set_xlabel("Frekvence (Hz)")
    ax[2].set_ylabel("Amplituda")
    ax[2].grid(True)
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()

def noise_cancel(m, noise, resampled_data, a):
    """
    Funkce noise_cancel provádí odstranění šumu pomocí metody spektrálního odečtu.
    
    Vstupy:
      - m: délka segmentu
      - noise: šum (např. získaný z částí signálu, kde není mluvení)
      - resampled_data: záznam k filtraci (převzorkovaný signál)
      - a: parametr alfa – míra spektrálního odečítání
      
    Výstup:
      - y_clean: vyfiltrovaný signál s odstraněným aditivním šumem
    """
    num_segments_noise = math.floor(len(noise) / m)
    segments_noise = np.zeros((num_segments_noise, m))
    segments_FFT_noise = np.zeros((num_segments_noise, m), dtype=complex)
    for i in range(num_segments_noise):
        segments_noise[i, :] = noise[i*m : i*m + m]
        segments_FFT_noise[i, :] = custom_FFT(segments_noise[i, :])
    
    num_segments_signal = math.floor(len(resampled_data) / m)
    segments_y = np.zeros((num_segments_signal, m))
    segments_FFT_y = np.zeros((num_segments_signal, m), dtype=complex)
    for i in range(num_segments_signal):
        segments_y[i, :] = resampled_data[i*m : i*m + m]
        segments_FFT_y[i, :] = custom_FFT(segments_y[i, :])
    
    N_omega_mean = np.mean(np.abs(segments_FFT_noise), axis=0)
    
    segments_FFT_y_clean = np.zeros((num_segments_signal, m), dtype=complex)
    y_clean_segments = np.zeros((num_segments_signal, m))
    for i in range(num_segments_signal):
        Y_omega_abs = np.abs(segments_FFT_y[i, :])
        S_omega_hat = Y_omega_abs - a * N_omega_mean
        S_omega_hat[S_omega_hat < 0] = 0
        segments_FFT_y_clean[i, :] = S_omega_hat * np.exp(1j * np.angle(segments_FFT_y[i, :]))
        y_clean_segments[i, :] = np.real(np.fft.ifft(segments_FFT_y_clean[i, :]))
    
    y_clean = y_clean_segments.flatten()
    
    return y_clean



    
def custom_FFT(data):
    data = np.asarray(data, dtype=complex)
    N = len(data)
    if N == 0:
        return np.array([], dtype=complex)
    if N == 1:
        return data
    if N % 2 != 0:
        data = np.append(data, 0)
        N += 1

    data_even = custom_FFT(data[0::2])
    data_odd  = custom_FFT(data[1::2])
    m = N // 2
    W_N = np.exp(-2j * np.pi / N)
    W = 1
    output = np.zeros(N, dtype=complex)
    for n in range(m):
        output[n]     = data_even[n] + W * data_odd[n]
        output[n + m] = data_even[n] - W * data_odd[n]
        W *= W_N
    return output

def plot_spectrogram(data, fs, title=""):

    window_size = int(round(0.03 * fs))
    overlap = int(round(0.5 * window_size))
    fft_resolution = window_size


    freq, times, Sxx = spectrogram(data, fs, nperseg=window_size, noverlap=overlap, nfft=fft_resolution)


    fig = plt.figure(figsize=(8, 6))
    plt.pcolormesh(times, freq, np.log10(np.abs(Sxx)), shading='gouraud', cmap='jet', rasterized=True)
    if title =="":
        plt.title("Spectrogram")
    else:
        plt.title("Spectrogram, "+str(title))
    plt.xlabel(r'$t\,[s]$', fontsize=12)
    plt.ylabel(r'$f\,[Hz]$', fontsize=12)
    plt.colorbar(label="Log Power")
    plt.tight_layout()
    plt.show()

def plot_amplitude_char(data, fs):
    freqs, amp  = compute_fourier(data,fs)

    plt.figure(figsize=(8, 6))
    plt.plot(freqs, amp, linewidth=1)
    plt.title('Amplitudová frekvenční charakteristika')
    plt.xlabel(r'$f\,[Hz]$', fontsize=12)
    plt.ylabel(r'$|A|$', fontsize=12)
    plt.grid(True)
    plt.xlim(0,fs/2)
    plt.show()

def compute_fourier(data,fs):
    N = len(data)
    fft_vals = np.fft.rfft(data)
    amp = np.abs(fft_vals) / N
    amp[1:-1] *= 2
    freqs = np.fft.rfftfreq(N, d=1/fs)
    return freqs,amp
