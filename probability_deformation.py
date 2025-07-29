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

# Initialize session state for parameters
def init_param(name, default):
    if name not in st.session_state:
        st.session_state[name] = default

init_param('a', 0.5)
init_param('b', 5.0)

# Sidebar controls
st.sidebar.header("Transform Parameters")
a_slider = st.sidebar.slider('a (slider)', -2.0, 2.0, st.session_state['a'], 0.01)
a_input  = st.sidebar.number_input('a (input)', -2.0, 2.0, st.session_state['a'], 0.01)
if a_slider != st.session_state['a']:
    st.session_state['a'] = a_slider
elif a_input != st.session_state['a']:
    st.session_state['a'] = a_input

b_slider = st.sidebar.slider('b (slider)', 0.0, 20.0, st.session_state['b'], 0.1)
b_input  = st.sidebar.number_input('b (input)', 0.0, 20.0, st.session_state['b'], 0.1)
if b_slider != st.session_state['b']:
    st.session_state['b'] = b_slider
elif b_input != st.session_state['b']:
    st.session_state['b'] = b_input

# Retrieve parameters
a = st.session_state['a']
b = st.session_state['b']

# Warn if invertibility condition may fail
if abs(a * b) >= 1:
    st.warning("Warning: |a * b| >= 1 may violate strict monotonicity (invertibility) of f(x).")

# Prepare data
# Analytical original density
x = np.linspace(-5, 5, 1000)
orig_pdf = 1/np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

# Generate samples and transform
data = np.random.RandomState(0).normal(size=200000)
y = data + a * np.sin(b * data)

# Estimate transformed density via histogram
hist, bins = np.histogram(y, bins=200, density=True)
centers = 0.5 * (bins[:-1] + bins[1:])

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, orig_pdf, lw=2, label='Original N(0,1)')
ax.plot(centers, hist, lw=2, label='Transformed')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.markdown("---")
st.caption("Transformation: f(x) = x + a sin(b x)")
