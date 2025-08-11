import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("FFT")

# ----- Controls -----
A1 = st.sidebar.number_input("$A_1$", 0.0, 5.0, 1.0, 0.1)
f1 = st.sidebar.number_input("$f_1$ (Hz)", 0.0, 100.0, 1.0, 0.1, format="%.2f")
p1 = st.sidebar.number_input("$\phi_1$ (rad)", 0.0, float(2*np.pi), 0.0)

A2 = st.sidebar.number_input("$A_2$", 0.0, 5.0, 0.8, 0.1)
f2 = st.sidebar.number_input("$f_2$ (Hz)", 0.0, 100.0, 2.3, 0.1, format="%.2f")
p2 = st.sidebar.number_input("$\phi_2$ (rad)", 0.0, float(2*np.pi), 0.0)

A3 = st.sidebar.number_input("$A_3$", 0.0, 5.0, 0.6, 0.1)
f3 = st.sidebar.number_input("$f_3$ (Hz)", 0.0, 100.0, 3.7, 0.1, format="%.2f")
p3 = st.sidebar.number_input("$\phi_3$ (rad)", 0.0, float(2*np.pi), 0.0)

st.sidebar.header("Grids")
N_plot = st.sidebar.number_input("Plot grid (N_plot)", 0, 500, 100, 1)
N_samp = st.sidebar.number_input("Sampling grid (N_samp)", 0, 500, 100, 1)


amplitudes = np.array([A1, A2, A3])
freqs = np.array([f1, f2, f3])
phases = np.array([p1, p2, p3])

def original_signal(t, amplitudes, freqs, phases):
    signal = 0
    for (amplitude, freq, phase) in zip(amplitudes, freqs, phases):
        signal += amplitude*np.sin(2*np.pi*freq*t + phase)

    return signal


t_plot = np.linspace(0, 1, N_plot, endpoint=False)
true_signal_plot = original_signal(t_plot, amplitudes, freqs, phases)


t_samp = np.linspace(0, 1, N_samp, endpoint=False)
true_signal_sample = original_signal(t_samp, amplitudes, freqs, phases)


# Reconstruct from samples
Cr = np.fft.rfft(true_signal_sample)  # length = N_samp//2 + 1

# 2) Zero-pad the HIGH frequencies to the target length
C = np.zeros(N_plot//2 + 1, dtype=complex)
scale = N_plot / N_samp
C[:len(Cr)] = scale * Cr

# 3) Inverse real FFT to get N_plot points (band-limited interpolation)
x_recon_plot = np.fft.irfft(C, n=N_plot)
x_recon_plot = np.real_if_close(x_recon_plot, tol=1000)

# ----- Plot -----
fig, axes = plt.subplots(2, 1, figsize=(10, 7), tight_layout=True)

# Time domain
axes[0].plot(t_plot, true_signal_plot, label="Original (analytic)", linewidth=2)
axes[0].plot(t_plot, x_recon_plot, "--", label="Reconstructed (from N_samp samples)")
axes[0].plot(t_samp, true_signal_sample, "o", ms=3, label="Samples")
axes[0].set_title("Time domain (both on plot grid)")
axes[0].set_xlabel("t (s)")
axes[0].set_ylabel("Amplitude")
axes[0].legend(loc="upper right")

# Spectrum (sampling grid)
freq_samp = np.fft.fftfreq(len(C), d=1/len(C))
pos = freq_samp >= 0
axes[1].stem(freq_samp[pos], np.abs(C)[pos]/len(C), basefmt=" ", label="|DFT(x_samp)|/N_samp")
axes[1].set_title("Magnitude spectrum (of sampled signal)")
axes[1].set_xlabel("Frequency (bin Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].legend(loc="upper right")

st.pyplot(fig)

st.caption(
    "We sample the 3-sine signal on N_samp points, take its DFT, then zero-pad the spectrum "
    "to N_plot and IFFT to view the reconstructed signal on the plot grid. "
    "If the sine frequencies are not on the DFT grid (k Hz), the reconstruction is the "
    "best periodic, band-limited fit to the samples (spectral leakage appears). "
    "Increase N_samp to improve the match."
)
