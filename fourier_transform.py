import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Interactive Discrete Fourier Series")

# Sidebar: controls for each harmonic component
st.sidebar.header("Harmonic Components")
num_components = st.sidebar.number_input("Number of components", min_value=1, max_value=6, value=3, step=1)
components = []
for i in range(int(num_components)):
    st.sidebar.subheader(f"Component {i+1}")
    amp = st.sidebar.slider(f"Amplitude A{i+1}", 0.0, 5.0, 1.0, key=f"amp{i}")
    freq = st.sidebar.number_input(
        f"Frequency k{i+1} (Hz)",
        min_value=0.0,
        max_value=100.0,
        value=float(i+1),
        step=0.1,
        format="%.2f",
        key=f"freq{i}"
    )
    phase = st.sidebar.slider(f"Phase φ{i+1} (rad)", 0.0, 2*np.pi, 0.0, key=f"phase{i}")
    components.append((amp, freq, phase))

# Time domain
N = 512
t = np.linspace(0, 1, N, endpoint=False)

# Build signal
signal = np.zeros_like(t)
for amp, freq, phase in components:
    signal += amp * np.sin(2 * np.pi * freq * t + phase)

# Compute discrete Fourier transform
fft_vals = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(N, d=1/N)

# Take positive frequencies
pos_mask = fft_freq >= 0
fft_freq = fft_freq[pos_mask]
fft_mag = np.abs(fft_vals)[pos_mask] / N

# Plotting
def plot():
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)
    # Time-domain
    axes[0].plot(t, signal)
    axes[0].set_title('Time-Domain Signal')
    axes[0].set_xlabel('t (s)')
    axes[0].set_ylabel('Amplitude')
    
    # Frequency-domain
    axes[1].stem(fft_freq, fft_mag)
    axes[1].set_title('Magnitude Spectrum')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_xlim(0, max([freq for _, freq, _ in components]) * 2 + 1)
    return fig

fig = plot()
st.pyplot(fig)
