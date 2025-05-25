import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Gaussian Mixture Explorer")

# Sliders for means and stds
m1 = st.slider("μ₁", -10.0, 10.0, 0.0)
s1 = st.slider("σ₁",  0.1,   5.0, 1.0)
m2 = st.slider("μ₂", -10.0, 10.0, 0.0)
s2 = st.slider("σ₂",  0.1,   5.0, 1.0)
m3 = st.slider("μ₃", -10.0, 10.0, 0.0)
s3 = st.slider("σ₃",  0.1,   5.0, 1.0)

# Fixed weights
w1, w2, w3 = 0.3, 0.4, 0.3

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
