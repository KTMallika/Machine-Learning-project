import streamlit as st
import numpy as np
from utils.helper import load_data, train_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="ML Project", layout="wide")

# =========================
# CUSTOM STYLE (BETTER COLORS)
# =========================
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 10px;
}
.stButton>button:hover {
    background-color: #45a049;
}
.success-box {
    background-color: #d4edda;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🌸 Iris Flower Prediction System")

# =========================
# LOAD DATA
# =========================
X, y, names, feature_names = load_data()

# =========================
# AUTO BEST K
# =========================
k_values = range(1, 11)
accuracies = []

for k_val in k_values:
    _, _, _, _, acc_val = train_model(X, y, k_val)
    accuracies.append(acc_val)

best_k = k_values[accuracies.index(max(accuracies))]

# Train final model
model, xtest, ytest, pred, acc = train_model(X, y, best_k)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📊 Model Info")
st.sidebar.success(f"Best K Selected: {best_k}")
st.sidebar.metric("Accuracy", f"{acc*100:.2f}%")

# =========================
# SESSION STATE
# =========================
if "inputs" not in st.session_state:
    st.session_state.inputs = [5.1, 3.5, 1.4, 0.2]

# =========================
# SAMPLE BUTTONS
# =========================
st.subheader("⚡ Try Sample Data")

c1, c2, c3 = st.columns(3)

if c1.button("🌼 Setosa"):
    st.session_state.inputs = [5.1, 3.5, 1.4, 0.2]

if c2.button("🌿 Versicolor"):
    st.session_state.inputs = [6.0, 2.9, 4.5, 1.5]

if c3.button("🌺 Virginica"):
    st.session_state.inputs = [6.5, 3.0, 5.5, 2.0]

# =========================
# INPUT SECTION
# =========================
st.subheader("🔍 Enter Flower Details")

col1, col2 = st.columns(2)

with col1:
    st.session_state.inputs[0] = st.number_input(
        "Sepal Length (cm)", 4.0, 8.0, st.session_state.inputs[0]
    )
    st.caption("Range → Setosa: 4.3–5.8 | Versicolor: 4.9–7.0 | Virginica: 4.9–7.9")

    st.session_state.inputs[1] = st.number_input(
        "Sepal Width (cm)", 2.0, 4.5, st.session_state.inputs[1]
    )
    st.caption("Range → Setosa: 2.3–4.4 | Versicolor: 2.0–3.4 | Virginica: 2.2–3.8")

with col2:
    st.session_state.inputs[2] = st.number_input(
        "Petal Length (cm)", 1.0, 7.0, st.session_state.inputs[2]
    )
    st.caption("Range → Setosa: 1.0–1.9 | Versicolor: 3.0–5.1 | Virginica: 4.5–6.9")

    st.session_state.inputs[3] = st.number_input(
        "Petal Width (cm)", 0.1, 2.5, st.session_state.inputs[3]
    )
    st.caption("Range → Setosa: 0.1–0.6 | Versicolor: 1.0–1.8 | Virginica: 1.5–2.5")

# =========================
# PREDICTION
# =========================
st.subheader("🎯 Prediction")

if st.button("Predict Flower"):
    new = np.array([st.session_state.inputs])
    output = model.predict(new)
    flower = names[output[0]]

    if flower == "setosa":
        st.success(f"🌼 Predicted Flower: {flower}")
    elif flower == "versicolor":
        st.info(f"🌿 Predicted Flower: {flower}")
    else:
        st.warning(f"🌺 Predicted Flower: {flower}")