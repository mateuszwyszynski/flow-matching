import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Unimodal to Bimodal Transform Demo", layout="centered")
st.title("Unimodal → Bimodal Transformation")
st.markdown(
    """
    This demo shows how a simple diffeomorphism 
    $$f(x) = x + a\sin(tx)$$
    can warp a standard normal distribution into a bimodal one. 
    Adjust `a` and `t` below or press ▶️ to animate `t` across its allowed range.
    """
)

# Sidebar controls
st.sidebar.header("Transform Parameters")

# Initialize defaults on first run
if 'a_slider' not in st.session_state:
    st.session_state['a_slider'] = 0.1
    st.session_state['a_input'] = 0.1
if 't_slider' not in st.session_state:
    st.session_state['t_slider'] = 5.0
    st.session_state['t_input'] = 5.0

# Callback functions
def sync_a_from_slider():
    st.session_state['a_input'] = st.session_state['a_slider']

def sync_a_from_input():
    st.session_state['a_slider'] = st.session_state['a_input']

def sync_b_from_slider():
    st.session_state['t_input'] = st.session_state['t_slider']

def sync_b_from_input():
    st.session_state['t_slider'] = st.session_state['t_input']

# Widgets with callbacks
a_slider = st.sidebar.slider(
    'a (slider)', -2.0, 2.0, st.session_state['a_slider'], 0.01,
    key='a_slider', on_change=sync_a_from_slider
)
a_input = st.sidebar.number_input(
    'a (input)', -2.0, 2.0, st.session_state['a_input'], 0.01,
    format="%.2f", key='a_input', on_change=sync_a_from_input
)

t_slider = st.sidebar.slider(
    't (slider)', 0.0, 20.0, st.session_state['t_slider'], 0.01,
    key='t_slider', on_change=sync_b_from_slider
)
t_input = st.sidebar.number_input(
    't (input)', 0.0, 20.0, st.session_state['t_input'], 0.01,
    format="%.2f", key='t_input', on_change=sync_b_from_input
)

# Final parameters
a = st.session_state['a_slider']
t = st.session_state['t_slider']
play = st.sidebar.button("▶️ Play t")

# Warn if invertibility condition may fail
if abs(a * t) >= 1:
    st.warning("Warning: |a * t| >= 1 may violate strict monotonicity (invertibility) of f(x).")

# Prepare original density once
x = np.linspace(-5, 5, 1000)
orig_pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Placeholder for plot
plot_placeholder = st.empty()

# Drawing function
def draw(a_val, t_val):
    u = np.linspace(-5, 5, 1000)
    y = u + a_val * np.sin(t_val * u)
    pdf = (np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)) / (1 + a_val * t_val * np.cos(t_val * u))
    idx = np.argsort(y)
    y_sorted = y[idx]
    pdf_sorted = pdf[idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, orig_pdf, lw=2, label='Original N(0,1)')
    ax.plot(y_sorted, pdf_sorted, lw=2, linestyle='--', 
            label=f'Transformed (a={a_val:.2f}, t={t_val:.2f})')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)
    plot_placeholder.pyplot(fig)

# Main logic: animate or static
if play:
    for t_val in np.linspace(0.0, 1/a, 100):
        if t_val == 1/a:
            continue
        draw(a, t_val)
        time.sleep(0.05)
else:
    draw(a, t)

st.markdown("---")
st.caption("Transformation: f(x) = x + a sin(t x)")
