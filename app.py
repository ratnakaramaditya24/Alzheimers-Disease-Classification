import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("alzheimers_model.pkl")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Alzheimer’s Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Alzheimer’s Disease Risk Prediction")
st.markdown("AI-based clinical screening tool for early risk detection.")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Patient Clinical Inputs")

mmse = st.sidebar.number_input(
    "MMSE Score (0–30)",
    min_value=0,
    max_value=30,
    value=15,
    step=1
)

functional = st.sidebar.number_input(
    "Functional Assessment (0–10)",
    min_value=0,
    max_value=10,
    value=5,
    step=1
)

adl = st.sidebar.number_input(
    "ADL Score (0–10)",
    min_value=0,
    max_value=10,
    value=5,
    step=1
)

memory = st.sidebar.selectbox(
    "Memory Complaints",
    ["No", "Yes"]
)

behavior = st.sidebar.selectbox(
    "Behavioral Problems",
    ["No", "Yes"]
)

# Convert categorical to numeric
memory = 1 if memory == "Yes" else 0
behavior = 1 if behavior == "Yes" else 0

# -------------------------------
# Prediction Button
# -------------------------------
st.markdown("---")

if st.button("🔍 Predict Alzheimer’s Risk"):
    input_data = np.array([[mmse, functional, adl, memory, behavior]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    # -------------------------------
    # Risk Meter
    # -------------------------------
    st.write("Risk Probability Meter")

    if probability < 0.30:
        st.success(f"Low Risk ({probability:.2f})")
        st.progress(int(probability * 100))

    elif probability < 0.70:
        st.warning(f"Moderate Risk ({probability:.2f})")
        st.progress(int(probability * 100))

    else:
        st.error(f"High Risk ({probability:.2f})")
        st.progress(int(probability * 100))

    # -------------------------------
    # Text Result
    # -------------------------------
    if prediction == 1:
        st.error("⚠ The patient is at HIGH RISK of Alzheimer’s Disease.")
    else:
        st.success("The patient is at LOW RISK of Alzheimer’s Disease.")

    # -------------------------------
    # Clinical Note
    # -------------------------------
    st.markdown("---")
    st.info(
        "This tool is designed for clinical screening support. "
        "It should not replace professional neurological diagnosis."
    )