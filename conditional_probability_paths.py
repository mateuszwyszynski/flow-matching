import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Conditional Probability Paths")

st.markdown(
    """
    This demo shows an optimal transport Gaussian conditional probability paths:

    $$p_t(x|x_1) = \mathcal{N}(\mu_t(x_1), \sigma_t(x_1))$$

    where:

    $$\mu_t(x_1) = tx_1$$

    $$\sigma_t(x_1) = 1 - (1 - \sigma_{min})t$$

    and $\sigma_{min} \in \mathbb{R}$
    is a minimum variance chosen when designing the conditional flow.
    """
)

# Sidebar controls for final means & std
st.sidebar.header("G at t=1")
mu_final = st.sidebar.slider("Mean at $t=1$", -5.0, 5.0, -3.0)
sigma_final = st.sidebar.slider("$\sigma_{min}$ at $t=1$", 0.1, 1.0, 0.4)

# Fixed initial stats at t=0
mu_init    = 0.0
sigma_init = 1.0

# Time‐slice baselines (t=1 top --> t=0 bottom)
time_slices = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

# Linear interpolation helper
def lerp(a, b, t):
    return a*t + b*(1 - t)

# Gaussian PDF
def gauss(x, mu, s):
    return np.exp(-(x - mu)**2 / (2 * s**2)) / (s * np.sqrt(2*np.pi))

# x‐axis
x = np.linspace(-10, 10, 500)

# Start figure
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-0.02, 1.3)
ax.set_xlabel("x")
ax.set_ylabel("time")

# Draw dashed baselines
for t in time_slices:
    ax.hlines(t, x.min(), x.max(), colors="gray", linestyles="dashed", linewidth=0.5)

means_t = [lerp(mu_final, mu_init, t) for t in time_slices]

# plot the trajectory
ax.plot(means_t, time_slices, ":", color="red", alpha=0.7)

# plot small pdf at each intermediate t
for t, m in zip(time_slices, means_t):
    # interpolate std between sigma_final and sigma_init
    sigma_t = lerp(sigma_final, sigma_init, t)
    y = gauss(x, m, sigma_t)*0.2
    ax.plot(x, y + t, color="gray", linewidth=1.5)

st.pyplot(fig)
