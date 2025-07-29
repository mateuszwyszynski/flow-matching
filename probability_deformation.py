import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Unimodal to Bimodal Transform Demo", layout="centered")
st.title("Unimodal → Bimodal Transformation")
st.markdown(
    """
    This demo shows how a simple diffeomorphism 
    $$f(x) = x + a\sin(bx)$$
    can warp a standard normal distribution into a bimodal one. 
    Adjust `a` and `b` below (including setting them to zero) to see the effect.
    """
)

# Sidebar controls
st.sidebar.header("Transform Parameters")

# Initialize defaults on first run
if 'a_slider' not in st.session_state:
    st.session_state['a_slider'] = 0.1
    st.session_state['a_input'] = 0.1
if 'b_slider' not in st.session_state:
    st.session_state['b_slider'] = 5.0
    st.session_state['b_input'] = 5.0

# Callback functions
def sync_a_from_slider():
    st.session_state['a_input'] = st.session_state['a_slider']

def sync_a_from_input():
    st.session_state['a_slider'] = st.session_state['a_input']

def sync_b_from_slider():
    st.session_state['b_input'] = st.session_state['b_slider']

def sync_b_from_input():
    st.session_state['b_slider'] = st.session_state['b_input']

# Widgets with callbacks
a_slider = st.sidebar.slider(
    'a (slider)', -2.0, 2.0, st.session_state['a_slider'], 0.01,
    key='a_slider', on_change=sync_a_from_slider
)
a_input = st.sidebar.number_input(
    'a (input)', -2.0, 2.0, st.session_state['a_input'], 0.01,
    format="%.2f", key='a_input', on_change=sync_a_from_input
)

b_slider = st.sidebar.slider(
    'b (slider)', 0.0, 20.0, st.session_state['b_slider'], 0.01,
    key='b_slider', on_change=sync_b_from_slider
)
b_input = st.sidebar.number_input(
    'b (input)', 0.0, 20.0, st.session_state['b_input'], 0.01,
    format="%.2f", key='b_input', on_change=sync_b_from_input
)

# Final parameters
a = st.session_state['a_slider']
b = st.session_state['b_slider']

# Warn if invertibility condition may fail
if abs(a * b) >= 1:
    st.warning("Warning: |a * b| >= 1 may violate strict monotonicity (invertibility) of f(x).")

# Prepare data
# Original density
x = np.linspace(-5, 5, 1000)
orig_pdf = 1/np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

# Compute analytic density via parametric plotting
u = np.linspace(-5, 5, 1000)
y_param = u + a * np.sin(b * u)
pdf_param = (1/np.sqrt(2 * np.pi) * np.exp(-0.5 * u**2)) / (1 + a * b * np.cos(b * u))
# Sort for a proper curve
idx = np.argsort(y_param)
y_param_sorted = y_param[idx]
pdf_param_sorted = pdf_param[idx]

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, orig_pdf, lw=2, label='Original N(0,1)')
ax.plot(y_param_sorted, pdf_param_sorted, lw=2, linestyle='--', label='Transformed (analytic)')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.markdown("---")
st.caption("Transformation: f(x) = x + a sin(b x)")
