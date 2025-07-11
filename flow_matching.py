import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Conditional Flow Matching")

# Sidebar controls for final means & std
st.sidebar.header("Gᵢ at t=1")
mu1_final = st.sidebar.slider("G₁ mean (μ₁) at t=1", -5.0, 5.0, -2.0)
mu2_final = st.sidebar.slider("G₂ mean (μ₂) at t=1", -5.0, 5.0,  0.0)
mu3_final = st.sidebar.slider("G₃ mean (μ₃) at t=1", -5.0, 5.0,  3.0)
sigma_final = st.sidebar.slider("common σ at t=1", 0.1, 1.0, 1.0)

# Fixed initial stats at t=0
mu_init    = 0.0
sigma_init = 1.0

# Time‐slice baselines (t=1 top → t=0 bottom)
time_slices = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
colors      = ["#d62728", "#2ca02c", "#1f77b4"]  # red/green/blue

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
ax.set_ylim(-0.02, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("time")

# Draw dashed baselines
for t in time_slices:
    ax.hlines(t, x.min(), x.max(), colors="gray", linestyles="dashed", linewidth=0.5)

# Plot each component drifting & its small‐pdfs
for mu_f, c in zip([mu1_final, mu2_final, mu3_final], colors):
    # compute mean at each slice
    means_t = [lerp(mu_f, mu_init, t) for t in time_slices]
    # plot the trajectory
    ax.plot(means_t, time_slices, ":", color=c, alpha=0.7)
    # plot small pdf at each intermediate t
    for t, m in zip(time_slices[:-1], means_t[:-1]):
        # interpolate std between sigma_final and sigma_init
        sigma_t = lerp(sigma_final, sigma_init, t)
        y = gauss(x, m, sigma_t) * 0.1
        ax.plot(x, y + t, color=c, linewidth=1.5)

# Mixture at t=1 (using sigma_final)
pdfs_t1 = [gauss(x, mu1_final, sigma_final),
           gauss(x, mu2_final, sigma_final),
           gauss(x, mu3_final, sigma_final)]
mixture_t1 = sum(pdfs_t1) / 3
ax.plot(x, mixture_t1 * 0.4 + 1.0,
        color="black", linewidth=3,
        label="Mixture (t=1)")

# Standard Normal at t=0 (σ=1)
std_norm = gauss(x, 0, sigma_init)
ax.plot(x, std_norm * 0.4 + 0.0,
        color="gray", linewidth=3,
        label="Std Normal (t=0)")

ax.legend(loc="upper right")
st.pyplot(fig)
