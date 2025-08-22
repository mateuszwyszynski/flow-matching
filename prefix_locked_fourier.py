import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Prefix-locked Fourier Deformation (keep t ≤ a, randomize t > a)")

# Controls
st.sidebar.header("Grids")
N_prev= st.sidebar.number_input("$N_{prev}$", 0, 1000, 500, 1)
N_next= st.sidebar.number_input("$N_{next}$", 0, 1000, 500, 1)

st.sidebar.header("Control Frequency")
control_freq = st.sidebar.slider("$f_c$ (Hz)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

st.sidebar.header("Prefix Lock")
a = st.sidebar.slider("Lock prefix until (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)/100

st.sidebar.header("Transition Period")
transition = st.sidebar.slider("Smoothly unlock (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)/100

K = 20
st.session_state.K = K

if "alpha_prev" not in st.session_state:
    st.session_state.alpha_prev = np.random.randn(K)
if "alpha_next" not in st.session_state:
    st.session_state.alpha_next = np.random.randn(K)

c1, c2 = st.sidebar.columns(2)
if c1.button("Resample prev $\alpha$"):
    st.session_state.alpha_prev = np.random.randn(K)
if c2.button("Resample next $\alpha$"):
    st.session_state.alpha_next = np.random.randn(K)

alpha_prev = st.session_state.alpha_prev
alpha_next = st.session_state.alpha_next


ks = np.arange(K)[None, :]


n_prev = np.arange(N_prev)[:, None]

Phi_prev = np.sqrt(2.0/N_prev) * np.cos(np.pi/N_prev * (n_prev + 0.5) * ks)
Phi_prev[:, 0] = 1.0/np.sqrt(N_prev)

t_grid_prev = np.linspace(0, N_prev / control_freq, N_prev, endpoint=False)
x_prev = Phi_prev @ alpha_prev


n_next = np.arange(N_next)[:, None]

Phi_next = np.sqrt(2.0/N_next) * np.cos(np.pi/N_next * (n_next + 0.5) * ks)
Phi_next[:, 0] = 1.0/np.sqrt(N_next)

t_grid_next = np.linspace(0, N_next / control_freq, N_next, endpoint=False)
x_next = Phi_next @ alpha_next


# Weight function: 1 in locked region, smoothly decays after a
def weight_fn(N, a, transition=0.1):
    w = np.ones(N)
    tnorm = np.linspace(0, 1, N, endpoint=False)
    mask = (tnorm > a) & (tnorm < a + transition)
    w[tnorm >= a + transition] = 0.0
    w[mask] = 0.5 * (1 + np.cos(np.pi * (tnorm[mask] - a) / transition))
    return w

common_N = min(N_next, N_prev)
weights = np.sqrt(weight_fn(common_N, a, transition))
x_next_locked = np.concat([
    weights*x_prev[:common_N] + (1-weights)*x_next[:common_N],
    x_next[common_N:]
])

alpha_next_locked = Phi_next.T @ x_next_locked

def plot():
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), tight_layout=True)

    # Time domain
    axes[0].plot(t_grid_prev, x_prev, label='Previous inference', linewidth=2)
    axes[0].plot(t_grid_next, x_next, label='Next inference', linewidth=2)
    axes[0].plot(t_grid_next, x_next_locked, label='Next prefix-locked', linestyle='--')
    axes[0].axvspan(0, a*max(np.max(t_grid_next), np.max(t_grid_next)), alpha=0.12, label='Locked region')
    axes[0].set_title('Time-Domain Signal')
    axes[0].set_xlabel('t (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()

    # Spectrum
    axes[1].stem(range(K), alpha_prev, label='Prev inference', basefmt=" ", linefmt="C0-")
    axes[1].stem(range(K), alpha_next, label='Next inference', basefmt=" ", linefmt="C1-")
    axes[1].stem(range(K), alpha_next_locked, label='Next inference locked', basefmt=" ", linefmt="C2-")
    axes[1].set_title('Inferred DCT coefficients')
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Magnitude')
    axes[1].legend(loc='upper right')

    return fig

fig = plot()
st.pyplot(fig)
