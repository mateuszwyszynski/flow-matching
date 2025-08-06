import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Interactive Discrete Fourier Series with Frequency Sampling")

# Sidebar: controls for each harmonic component
st.sidebar.header("Harmonic Components")
num_components = st.sidebar.number_input("Number of components", min_value=1, max_value=6, value=3, step=1)
components = []
for i in range(int(num_components)):
    st.sidebar.subheader(f"Component {i+1}")
    amp = st.sidebar.number_input(f"Amplitude A{i+1}", 0.0, 5.0, 1.0, key=f"amp{i}")
    freq = st.sidebar.number_input(
        f"Frequency k{i+1} (Hz)",
        min_value=0.0,
        max_value=100.0,
        value=float(i+1),
        step=0.1,
        format="%.2f",
        key=f"freq{i}"
    )
    phase = st.sidebar.number_input(f"Phase φ{i+1} (rad)", 0.0, 2*np.pi, 0.0, key=f"phase{i}")
    components.append((amp, freq, phase))

st.sidebar.header("Frequency Sampling")
sample_size = st.sidebar.number_input(
    "Number of samples", min_value=1, max_value=200, value=20, step=1
)
sigma = st.sidebar.number_input(
    "Frequency σ (Hz)", 0.0, float(max(c[1] for c in components) or 1), 1.0
)

N = 512
t = np.linspace(0, 1, N, endpoint=False)

amps = np.array([c[0] for c in components])
freqs = np.array([c[1] for c in components])
phases = np.array([c[2] for c in components])

core_signal = np.sum(
    amps[:, None] * np.sin(2*np.pi*freqs[:, None]*t + phases[:, None]),
    axis=0
)

cov = np.eye(len(freqs)) * sigma**2
sampled_freqs = np.random.default_rng().multivariate_normal(
    mean=freqs, cov=cov, size=sample_size
)

sampled_signals = [
    np.sum(amps[:, None] * np.sin(2*np.pi*sf[:, None]*t + phases[:, None]), axis=0)
    for sf in sampled_freqs
]

fft_vals = np.fft.fft(core_signal)
fft_freq = np.fft.fftfreq(N, d=1/N)
pos_mask = fft_freq >= 0
fft_freq = fft_freq[pos_mask]
fft_mag = np.abs(fft_vals)[pos_mask] / N

def plot():
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)
    for sig in sampled_signals:
        axes[0].plot(t, sig, linestyle='--', alpha=0.3)
    axes[0].plot(t, core_signal, color='black', linewidth=2)
    axes[0].set_title('Time-Domain Signal\n(core in black, samples dashed)')
    axes[0].set_xlabel('t (s)')
    axes[0].set_ylabel('Amplitude')

    axes[1].stem(fft_freq, fft_mag)
    axes[1].set_title('Magnitude Spectrum (Core Signal)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_xlim(0, freqs.max()*2 + 1)

    return fig

fig = plot()
st.pyplot(fig)
