import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Prefix-locked Fourier Deformation (keep t ≤ a, randomize t > a)")

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
N_samp = st.sidebar.number_input("Sampling grid (N_samp)", 0, 500, 100, 1)

st.sidebar.header("Prefix Lock")
a = st.sidebar.slider("Lock prefix until a (seconds)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

st.sidebar.header("Random Tail")
seed = st.sidebar.number_input("Random seed", value=0, step=1)
tail_scale = st.sidebar.number_input("Tail noise scale (σ)", min_value=0.0, max_value=50.0, value=0.5, step=0.1)


amplitudes = np.array([A1, A2, A3])
freqs = np.array([f1, f2, f3])
phases = np.array([p1, p2, p3])

def original_signal(t, amplitudes, freqs, phases):
    signal = 0
    for (amplitude, freq, phase) in zip(amplitudes, freqs, phases):
        signal += amplitude*np.sin(2*np.pi*freq*t + phase)

    return signal

t_samp = np.linspace(0, 1, N_samp, endpoint=False)
target_signal = original_signal(t_samp, amplitudes, freqs, phases)


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
Cr0 = np.fft.rfft(target_signal)
c0  = rfft_to_fft(Cr0, N_samp)


def idft_matrix(N: int) -> np.ndarray:
    # IDFT matrix so that x = A @ c equals np.fft.ifft(c)
    n = np.arange(N)[:, None]
    k = np.arange(N)[None, :]
    return np.exp(1j * 2*np.pi * n * k / N) / N


N_head = int(np.floor(a * N_samp)) 
if N_head == N_samp:
    c_new = c0.copy()
    x_new = target_signal.copy()
else:
    A = idft_matrix(N_samp)
    N_head = int(np.floor(a * N_samp))
    A_head, A_tail = A[:N_head], A[N_head:]

    # --- 3) Nullspace projector for prefix lock ---
    U, S, Vh = np.linalg.svd(A_head, full_matrices=True)
    tol = max(A_head.shape) * (S.max() if S.size else 0.0) * np.finfo(float).eps
    r = int(np.sum(S > tol))
    Nmat = Vh.conj().T[:, r:]  # basis of Null(A_head)
    # Optional projector: P = I - A_head^† A_head
    # but using Nmat is fine.

    # --- 4) Propose a frequency-domain modification (random bins) ---
    rng = np.random.default_rng(seed)
    delta_f = tail_scale*rng.standard_normal(N_samp) + 1j*rng.standard_normal(N_samp)
    # (Optionally only touch “high” bins, e.g., indices k>=k0)
    # k0 = N_samp//8; delta_f[:k0] = 0; delta_f[-k0:] = 0

    # --- 5) Project the freq change onto the feasible set (preserve head) ---
    # Feasible updates are exactly of the form Nmat @ z. Solve least-squares:
    z = np.linalg.lstsq(A_tail @ Nmat, A_tail @ delta_f, rcond=None)[0]
    c_new = c0 + Nmat @ z

    x_new = np.fft.ifft(c_new, n = N_samp)
    # Numerical guard: make it real if it's real up to tiny imag error
    x_new = np.real_if_close(x_new, tol=1000)

# -------------------------------
# Plots
# -------------------------------
fft_freq = np.fft.fftfreq(N_samp, d=1/N_samp)
pos_mask = fft_freq >= 0
fft_freq_pos = fft_freq[pos_mask]

fft_mag_core = np.abs(c0)[pos_mask] / N_samp
fft_mag_new  = np.abs(c_new)[pos_mask] / N_samp

def plot():
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), tight_layout=True)

    # Time domain
    axes[0].plot(t_samp, target_signal, label='Original', linewidth=2)
    axes[0].plot(t_samp, x_new, label='New (prefix-locked)', linestyle='--')
    axes[0].axvspan(0, a, alpha=0.12, label='Locked region')
    axes[0].set_title('Time-Domain Signal')
    axes[0].set_xlabel('t (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')

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
