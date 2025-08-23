import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Prefix-locked Fourier Deformation (keep t ≤ a, randomize t > a)")

# Controls
st.sidebar.header("Shape grid")
N_shape = st.sidebar.number_input("$N_{shape}$", 0, 2000, 1000, 1)

st.sidebar.header("Inferred times")
T_prev = st.sidebar.number_input("$T_{prev}$", 0.0, 15.0, 10.0, 0.01)
T_next = st.sidebar.number_input("$T_{next}$", 0.0, 15.0, 10.0, 0.01)

st.sidebar.header("Control Frequency")
control_freq = st.sidebar.slider("$f_c$ (Hz)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

st.sidebar.header("Inference latency")
t_infer = st.sidebar.slider("Inference latency (s)", min_value=0.0, max_value=5.0, value=1.0, step=0.01)

tau_locked = t_infer / T_prev

N_prev = int(T_prev * control_freq)
N_next = int(T_next * control_freq)

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


def dct_matrix(K, N, N_scale: int = None):
    if N_scale is None:
        N_scale = N
    n = np.arange(N)[:, None]
    ks = np.arange(K)[None, :]

    Phi = np.sqrt(2.0/N_scale) * np.cos(np.pi/N * (n + 0.5) * ks)
    Phi[:, 0] = 1.0/np.sqrt(N_scale)

    return Phi


t_grid_prev = np.linspace(0, T_prev, N_prev, endpoint=False)
x_prev = dct_matrix(K, N_prev, N_scale=N_shape) @ alpha_prev


t_grid_next = np.linspace(0, T_next, N_next, endpoint=False)
Phi_next = dct_matrix(K, N_next, N_scale=N_shape)
x_next = Phi_next @ alpha_next


Phi_shape = dct_matrix(K, N_shape)

H = int(tau_locked * N_shape)
A = Phi_shape
Ah, At = A[:H], A[H:]

G = At.T @ At
C = Ah
rhs = np.concatenate([At.T @ (At @ alpha_next), C @ alpha_prev])
KKT = np.block([[G, C.T],
                [C, np.zeros((H, H))]])
sol = np.linalg.solve(KKT, rhs)
alpha_next_locked = sol[:K]


x_prev_normalised_time = Phi_shape @ alpha_prev
x_next_normalised_time = Phi_shape @ alpha_next
x_next_locked_normalised_time = Phi_shape @ alpha_next_locked


x_next_locked = Phi_next @ alpha_next_locked
t_locked = tau_locked * T_prev
t_exec_start = tau_locked * T_next

def plot():
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), tight_layout=True)

    # Time domain
    axes[0].plot(t_grid_prev, x_prev, label='Previous inference', linewidth=2)
    axes[0].plot(t_grid_next, x_next, label='Next inference', linewidth=2)
    axes[0].plot(t_grid_next, x_next_locked, label='Next prefix-locked', linestyle='--')
    axes[0].axvspan(0, t_locked, alpha=0.12, label='Locked region')
    axes[0].axvspan(t_locked, t_exec_start, alpha=0.2, label='Ignored for execution')
    axes[0].set_title('Time-Domain Signal')
    axes[0].set_xlabel('t (s)')
    axes[0].set_ylabel('Position x')
    axes[0].legend()

    # Normalised time domain
    tau_grid = np.linspace(0, 1, N_shape, endpoint=False)
    axes[1].plot(tau_grid, x_prev_normalised_time, label='Previous inference', linewidth=2)
    axes[1].plot(tau_grid, x_next_normalised_time, label='Next inference', linewidth=2)
    axes[1].plot(tau_grid, x_next_locked_normalised_time, label='Next prefix-locked', linestyle='--')
    axes[1].axvspan(0, tau_locked, alpha=0.12, label='Locked region')
    axes[1].set_title('Normalised Time-Domain Signal')
    axes[1].set_xlabel('tau')
    axes[1].set_ylabel('Position x')
    axes[1].legend()

    # Spectrum
    axes[2].stem(range(K), alpha_prev, label='Prev inference', basefmt=" ", linefmt="C0-")
    axes[2].stem(range(K), alpha_next, label='Next inference', basefmt=" ", linefmt="C1-")
    axes[2].stem(range(K), alpha_next_locked, label='Next inference locked', basefmt=" ", linefmt="C2-")
    axes[2].set_title('Inferred DCT coefficients')
    axes[2].set_xlabel('Component')
    axes[2].set_ylabel('Magnitude')
    axes[2].legend(loc='upper right')

    return fig

fig = plot()
st.pyplot(fig)
