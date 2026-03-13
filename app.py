import streamlit as st
import numpy as np
import joblib
import pennylane as qml
from pennylane import numpy as pnp

# -------------------- Load the saved model components --------------------
# If you didn't save pca, you can comment out the pca line and set n_qubits manually.
pca = joblib.load('pca.pkl')               # PCA transformer
scaler = joblib.load('scaler.pkl')          # MinMaxScaler fitted on PCA output
params = joblib.load('best_params.pkl')     # trained quantum circuit parameters
threshold = joblib.load('threshold.pkl')    # optimal decision threshold

# Convert params back to PennyLane array (required for the circuit)
params = pnp.array(params, requires_grad=False)

# -------------------- Quantum circuit definition --------------------
n_qubits = 6   # number of PCA components used
dev = qml.device("default.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev, interface="autograd")
def circuit(params, x):
    # Encode features (same as in Block 8)
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i] * 0.5, wires=i)
    # Variational layers
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
    # Measurement (expectation value of ZZ on first two qubits)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

def predict_from_pca(pca_components):
    """
    pca_components: array-like of 6 values (the 6 PCA components)
    Returns: (probability_of_fraud, binary_prediction)
    """
    # Scale the PCA components to [0, π] (the scaler was trained on PCA output)
    X_norm = scaler.transform([pca_components])[0]
    # Run quantum circuit
    exp = circuit(params, X_norm)
    prob = (exp + 1.0) / 2.0          # map from [-1,1] to [0,1]
    pred = int(prob >= threshold)
    return prob, pred

# -------------------- Streamlit User Interface --------------------
st.set_page_config(page_title="Quantum Fraud Detector", layout="centered")
st.title("🧠 Quantum Fraud Detection")
st.markdown("Adjust the six PCA components and click **Predict**.")

# Approximate ranges for the six PCA components – you can replace these with
# actual min/max from your dataset to make the sliders more realistic.
# For now we use sensible defaults.
ranges = [(-10.0, 10.0), (-8.0, 8.0), (-6.0, 6.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]

# Create three columns to arrange inputs nicely
cols = st.columns(3)
inputs = []
for i in range(n_qubits):
    with cols[i % 3]:
        val = st.number_input(
            f"PC{i+1}",
            min_value=ranges[i][0],
            max_value=ranges[i][1],
            value=0.0,
            step=0.1,
            format="%.2f"
        )
        inputs.append(val)

# Predict button
if st.button("🔍 Predict Transaction", type="primary"):
    prob, pred = predict_from_pca(np.array(inputs))
    st.divider()
    if pred == 1:
        st.error(f"### 🚨 FRAUD DETECTED\n**Probability of fraud:** {prob:.4f}")
    else:
        st.success(f"### ✅ NORMAL TRANSACTION\n**Probability of fraud:** {prob:.4f}")
    st.caption(f"Decision threshold used: {threshold:.4f}")