import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Prefix-locked Fourier Deformation (keep t ≤ a, randomize t > a)")

# ----- Controls -----
st.sidebar.header("Previously Predicted Signal")

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
N_samp = st.sidebar.number_input("Sampling grid (N_samp)", 0, 500, 100, 1)

st.sidebar.header("Prefix Lock")
a = st.sidebar.slider("Lock prefix until a (seconds)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

st.sidebar.header("Transition Period")
transition = st.sidebar.slider("Smoothly unlock (seconds)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

st.sidebar.header("New Signal (Without Lock)")

A1_new = st.sidebar.number_input("$A^{new}_1$", 0.0, 5.0, 1.0, 0.1)
f1_new = st.sidebar.number_input("$f^{new}_1$ (Hz)", 0.0, 100.0, 1.0, 0.1, format="%.2f")
p1_new = st.sidebar.number_input("$\phi^{new}_1$ (rad)", 0.0, float(2*np.pi), 0.0)

A2_new = st.sidebar.number_input("$A^{new}_2$", 0.0, 5.0, 0.8, 0.1)
f2_new = st.sidebar.number_input("$f^{new}_2$ (Hz)", 0.0, 100.0, 2.3, 0.1, format="%.2f")
p2_new = st.sidebar.number_input("$\phi^{new}_2$ (rad)", 0.0, float(2*np.pi), 0.0)

A3_new = st.sidebar.number_input("$A^{new}_3$", 0.0, 5.0, 0.6, 0.1)
f3_new = st.sidebar.number_input("$f^{new}_3$ (Hz)", 0.0, 100.0, 3.7, 0.1, format="%.2f")
p3_new = st.sidebar.number_input("$\phi^{new}_3$ (rad)", 0.0, float(2*np.pi), 0.0)


amplitudes = np.array([A1, A2, A3])
freqs = np.array([f1, f2, f3])
phases = np.array([p1, p2, p3])


amplitudes_new = np.array([A1_new, A2_new, A3_new])
freqs_new = np.array([f1_new, f2_new, f3_new])
phases_new = np.array([p1_new, p2_new, p3_new])

def sample_signal(t, amplitudes, freqs, phases):
    signal = 0
    for (amplitude, freq, phase) in zip(amplitudes, freqs, phases):
        signal += amplitude*np.sin(2*np.pi*freq*t + phase)

    return signal

t_samp = np.linspace(0, 1, N_samp, endpoint=False)
previous_inference_signal = sample_signal(t_samp, amplitudes, freqs, phases)
new_inference_signal = sample_signal(t_samp, amplitudes_new, freqs_new, phases_new)


def rfft_to_fft(Cr, N):
    """Expand rFFT coeffs (length N//2+1) to full FFT (length N) with Hermitian symmetry."""
    X = np.zeros(N, dtype=complex)
    K = N // 2
    X[0] = Cr[0]
    if N % 2 == 0:
        X[K] = Cr[K]                     # Nyquist (real)
        X[1:K] = Cr[1:K]
        X[-(K-1):] = np.conj(Cr[1:K][::-1])
    else:
        X[1:K+1] = Cr[1:K+1]
        X[-K:] = np.conj(Cr[1:K+1][::-1])
    return X

def fft_to_rfft(X):
    """Compress full FFT (length N) back to rFFT (length N//2+1)."""
    N = len(X)
    K = N // 2
    Cr = np.empty(K + 1, dtype=complex)
    Cr[0] = X[0]
    if N % 2 == 0:
        Cr[1:K] = X[1:K]
        Cr[K] = X[K]
    else:
        Cr[1:K+1] = X[1:K+1]
    return Cr


# FFT from samples
cr0 = np.fft.rfft(previous_inference_signal)
c0 = rfft_to_fft(cr0, N_samp)

cr0_new = np.fft.rfft(new_inference_signal)
c0_new = rfft_to_fft(cr0_new, N_samp)


def idft_matrix(N: int) -> np.ndarray:
    # IDFT matrix so that x = A @ c equals np.fft.ifft(c)
    n = np.arange(N)[:, None]
    k = np.arange(N)[None, :]
    return np.exp(1j * 2*np.pi * n * k / N) / N


# Weight function: 1 in locked region, smoothly decays after a
def weight_fn(t, a, transition=0.1):
    w = np.zeros_like(t)
    mask = (t > a) & (t < a + transition)
    w[t >= a + transition] = 1.0
    # smooth decay in [a, a+transition]
    w[mask] = 0.5 * (1 - np.cos(np.pi * (t[mask]-a)/transition))
    return w

N_head = int(np.floor(a * N_samp)) 
if N_head == N_samp:
    c_new = c0.copy()
    x_new = previous_inference_signal.copy()
else:
    A = idft_matrix(N_samp)
    N_head = int(np.floor(a * N_samp))
    A_head, A_tail = A[:N_head], A[N_head:]

    U, S, Vh = np.linalg.svd(A_head, full_matrices=True)
    tol = max(A_head.shape) * (S.max() if S.size else 0.0) * np.finfo(float).eps
    r = int(np.sum(S > tol))
    Nmat = Vh.conj().T[:, r:]

    weights = np.sqrt(weight_fn(t_samp, a, transition)[N_head:])
    M = A_tail @ Nmat
    d   = A_tail @ (c0_new - c0) 
    dw   = weights * d

    z = np.linalg.lstsq(M, dw, rcond=None)[0]
    c_new = c0 + Nmat @ z

    new_inference_signal_prefix_locked = np.fft.ifft(c_new, n = N_samp)
    new_inference_signal_prefix_locked = np.real_if_close(new_inference_signal_prefix_locked, tol=1000)

# Plots
fft_freq = np.fft.fftfreq(N_samp, d=1/N_samp)
pos_mask = fft_freq >= 0
fft_freq_pos = fft_freq[pos_mask]

fft_mag_core = np.abs(c0)[pos_mask] / N_samp
fft_mag_new  = np.abs(c_new)[pos_mask] / N_samp

def plot():
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), tight_layout=True)

    # Time domain
    axes[0].plot(t_samp, previous_inference_signal, label='Previous inference', linewidth=2)
    axes[0].plot(t_samp, new_inference_signal, label='New inference', linestyle='--')
    axes[0].plot(t_samp, new_inference_signal_prefix_locked, label='New inference with prefix-locked', linestyle='--')
    axes[0].axvspan(0, a, alpha=0.12, label='Locked region')
    axes[0].set_title('Time-Domain Signal')
    axes[0].set_xlabel('t (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='lower left')

    # Spectrum
    axes[1].stem(fft_freq_pos, fft_mag_core, label='Original |c|/N', basefmt=" ")
    axes[1].stem(fft_freq_pos, fft_mag_new, label='New |c|/N', linefmt='C1-', markerfmt='C1o', basefmt=" ")
    axes[1].set_title('Magnitude Spectrum (positive freqs)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].legend(loc='upper right')

    return fig

fig = plot()
st.pyplot(fig)
