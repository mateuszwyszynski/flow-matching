import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Unimodal to Multimodal Probability Deformation Demo", layout="centered")
st.title("Unimodal to Multimodal Probability Deformation")
st.markdown(
    """
    This demo shows how a simple diffeomorphism 
    $$f(x) = x + s\sin(tx)$$
    can warp a standard normal distribution into a bimodal one. 
    Adjust `s` and `t` below or press ▶️ to animate `t` across its allowed range.
    """
)

# Sidebar controls
st.sidebar.header("Transform Parameters")

# Initialize defaults on first run
if 's_slider' not in st.session_state:
    st.session_state['s_slider'] = 0.1
    st.session_state['s_input'] = 0.1
if 't_slider' not in st.session_state:
    st.session_state['t_slider'] = 5.0
    st.session_state['t_input'] = 5.0

# Callback functions
def sync_s_from_slider():
    st.session_state['s_input'] = st.session_state['s_slider']

def sync_s_from_input():
    st.session_state['s_slider'] = st.session_state['s_input']

def sync_t_from_slider():
    st.session_state['t_input'] = st.session_state['t_slider']

def sync_t_from_input():
    st.session_state['t_slider'] = st.session_state['t_input']

# Widgets with callbacks
s_slider = st.sidebar.slider(
    's (slider)',
    -2.0, 2.0,
    key='s_slider',
    step=0.01,
    on_change=sync_s_from_slider,
)
s_input = st.sidebar.number_input(
    's (input)',
    -2.0, 2.0,
    key='s_input',
    step=0.01,
    on_change=sync_s_from_slider,
)

t_slider = st.sidebar.slider(
    't (slider)',
    0.0, 20.0,
    key='t_slider',
    step=0.01,
    on_change=sync_t_from_slider,
)
t_input = st.sidebar.number_input(
    't (input)',
    0.0, 20.0,
    key='t_input',
    step = 0.01,
    on_change=sync_t_from_input,
    format="%.2f",
)

# Final parameters
s = st.session_state['s_slider']
t = st.session_state['t_slider']
play = st.sidebar.button("▶️ Play t")

# Warn if invertibility condition may fail
if abs(s * t) >= 1:
    st.warning("Warning: |s * t| >= 1 may violate strict monotonicity (invertibility) of f(x).")

# Prepare original density once
x = np.linspace(-5, 5, 1000)
orig_pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Placeholder for plot
plot_placeholder = st.empty()

# Drawing function
def draw(s_val, t_val):
    u = np.linspace(-5, 5, 1000)
    y = u + s_val * np.sin(t_val * u)
    pdf = (np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)) / (1 + s_val * t_val * np.cos(t_val * u))
    idx = np.argsort(y)
    y_sorted = y[idx]
    pdf_sorted = pdf[idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, orig_pdf, lw=2, label='Original N(0,1)')
    ax.plot(y_sorted, pdf_sorted, lw=2, linestyle='--', 
            label=f'Transformed (a={s_val:.2f}, t={t_val:.2f})')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)
    plot_placeholder.pyplot(fig)

# Main logic: animate or static
if play:
    for t_val in np.linspace(0.0, 1/s, 100):
        if t_val == 1/a:
            continue
        draw(s, t_val)
        time.sleep(0.05)
else:
    draw(s, t)

st.markdown("---")
st.caption("Transformation: f(x) = x + s sin(t x)")
