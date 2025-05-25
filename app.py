import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Gaussian Mixture Explorer")

# Sliders for means and stds
m1 = st.sidebar.slider("μ₁", -10.0, 10.0, 0.0)
s1 = st.sidebar.slider("σ₁",  0.1,   5.0, 1.0)
m2 = st.sidebar.slider("μ₂", -10.0, 10.0, 0.0)
s2 = st.sidebar.slider("σ₂",  0.1,   5.0, 1.0)
m3 = st.sidebar.slider("μ₃", -10.0, 10.0, 0.0)
s3 = st.sidebar.slider("σ₃",  0.1,   5.0, 1.0)

# (optional) you can even put weights in the sidebar
w1 = st.sidebar.slider("w₁", 0.0, 1.0, 0.33)
w2 = st.sidebar.slider("w₂", 0.0, 1.0, 0.33)
w3 = st.sidebar.slider("w₃", 0.0, 1.0, 0.34)
weights = np.array([w1, w2, w3]) / (w1 + w2 + w3)

x = np.linspace(-10, 10, 1000)
def gauss(x, μ, σ):
    return 1/(σ*np.sqrt(2*np.pi))*np.exp(-(x-μ)**2/(2*σ**2))

pdfs = [gauss(x,m,s) for m,s in [(m1,s1),(m2,s2),(m3,s3)]]
mix = w1*pdfs[0] + w2*pdfs[1] + w3*pdfs[2]

fig, ax = plt.subplots()
for i,p in enumerate(pdfs):
    ax.plot(x, p, label=f"G{i+1} (μ={ [m1,m2,m3][i]:.1f}, σ={ [s1,s2,s3][i]:.1f})")
ax.plot(x, mix, "--", lw=4, label="Mixture")
ax.legend(); ax.set_xlabel("x"); ax.set_ylabel("Density")
st.pyplot(fig)
