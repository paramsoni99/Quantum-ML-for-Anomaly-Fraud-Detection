import streamlit as st
import numpy as np
import joblib
import pennylane as qml
from pennylane import numpy as pnp
import plotly.graph_objects as go

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Quantum-Quest | Fraud AI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, high-tech look
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00BABE;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #008f91;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Load the saved model components --------------------
@st.cache_resource
def load_models():
    pca = joblib.load('pca.pkl')               
    scaler = joblib.load('scaler.pkl')          
    params = joblib.load('best_params.pkl')     
    threshold = joblib.load('threshold.pkl')    
    return pca, scaler, pnp.array(params, requires_grad=False), threshold

pca, scaler, params, threshold = load_models()

# -------------------- Quantum circuit definition --------------------
n_qubits = 6  
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev, interface="autograd")
def circuit(params, x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i] * 0.5, wires=i)
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

def predict_from_pca(pca_components):
    X_norm = scaler.transform([pca_components])[0]
    exp = circuit(params, X_norm)
    prob = (exp + 1.0) / 2.0  
    pred = int(prob >= threshold)
    return prob.item(), pred

# -------------------- Session State Initialization --------------------
# We use session state to keep the sliders and number inputs perfectly synced
for i in range(n_qubits):
    if f"pc_{i}" not in st.session_state:
        st.session_state[f"pc_{i}"] = 0.0

def update_from_slider(i):
    st.session_state[f"pc_{i}"] = st.session_state[f"slider_{i}"]

def update_from_num(i):
    st.session_state[f"pc_{i}"] = st.session_state[f"num_{i}"]

# -------------------- Sidebar Information --------------------
with st.sidebar:
    st.title("Quantum-Quest")
    st.markdown("### Next-Gen Fraud Detection")
    st.write("This engine uses a 6-qubit Variational Quantum Classifier (VQC) to evaluate highly imbalanced financial transaction data.")
    st.divider()
    st.markdown("**Model Specs:**")
    st.write("• **Architecture:** PennyLane VQC")
    st.write("• **Entanglement:** Strongly Entangling Layers")
    st.write(f"• **Optimal Threshold:** `{threshold:.4f}`")
    st.divider()
    st.caption("Built by Twisha,Param,Pranshu")

# -------------------- Main UI Dashboard --------------------
st.title("Quantum Anomaly Detection Engine")
st.markdown("Adjust the mathematically compressed Principal Component (PCA) values below to simulate a transaction. The radar chart visualizes the data footprint before it enters the quantum circuit.")

st.write("") # Spacer

# Create a two-column layout: Controls on left, Visualization on right
col1, space, col2 = st.columns([1.5, 0.1, 1.5])

ranges = [(-10.0, 10.0), (-8.0, 8.0), (-6.0, 6.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
inputs = []

with col1:
    st.subheader("Transaction Parameters")
    st.markdown("Use the sliders or exact number inputs to alter the PCA dimensions:")
    st.write("")
    
    # Generate synced sliders and number inputs
    for i in range(n_qubits):
        st.markdown(f"**Principal Component {i+1}**")
        sl_col, num_col = st.columns([3, 1.2])
        
        with sl_col:
            st.slider(
                f"PC{i+1} Slider",
                min_value=float(ranges[i][0]),
                max_value=float(ranges[i][1]),
                value=st.session_state[f"pc_{i}"],
                step=0.1,
                key=f"slider_{i}",
                on_change=update_from_slider,
                args=(i,),
                label_visibility="collapsed"
            )
        with num_col:
            st.number_input(
                f"PC{i+1} Number",
                min_value=float(ranges[i][0]),
                max_value=float(ranges[i][1]),
                value=st.session_state[f"pc_{i}"],
                step=0.1,
                key=f"num_{i}",
                on_change=update_from_num,
                args=(i,),
                label_visibility="collapsed"
            )
            
        inputs.append(st.session_state[f"pc_{i}"])
        st.write("") # Add a little spacing between components
        
    simulate_pressed = st.button("Initialize Quantum Simulation")

with col2:
    st.subheader("Multidimensional Footprint")
    
    # Create an interactive Radar Chart for the inputs
    categories = [f'PC1', f'PC2', f'PC3', f'PC4', f'PC5', f'PC6']
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=inputs + [inputs[0]], # close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0, 186, 190, 0.2)',
        line=dict(color='#00BABE', width=2),
        name='Transaction Footprint'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-10, 10], showticklabels=False),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# -------------------- Prediction Results Area --------------------
st.divider()

if simulate_pressed:
    with st.spinner("Entangling Qubits and Measuring State..."):
        prob, pred = predict_from_pca(np.array(inputs))
        
    res_col1, res_col2 = st.columns([1, 1.5])
    
    with res_col1:
        st.write("")
        st.write("")
        if pred == 1:
            st.error("### FRAUD DETECTED")
            st.write("The quantum circuit has flagged this multidimensional footprint as a highly probable anomaly.")
        else:
            st.success("### TRANSACTION VERIFIED")
            st.write("The quantum circuit recognizes this footprint as standard financial behavior.")
            
    with res_col2:
        # Create a beautiful Gauge Chart for the probability
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            title = {'text': "Fraud Probability Map", 'font': {'size': 20}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#00BABE"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, threshold], 'color': "rgba(0, 255, 0, 0.1)"},
                    {'range': [threshold, 1], 'color': "rgba(255, 0, 0, 0.1)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold
                }
            }
        ))
        
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
else:
    st.info("Adjust the transaction parameters and click 'Initialize Quantum Simulation' to see the VQC in action.")