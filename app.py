import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("alzheimers_model.pkl")

st.title("🧠 Alzheimer’s Disease Risk Prediction")

st.write("Enter Patient Clinical Details:")

# Inputs
mmse = st.slider("MMSE Score", 0, 30, 15)
functional = st.slider("Functional Assessment Score", 0, 10, 5)
adl = st.slider("ADL Score", 0, 10, 5)
memory = st.selectbox("Memory Complaints", ["No", "Yes"])
behavior = st.selectbox("Behavioral Problems", ["No", "Yes"])

# Convert categorical
memory = 1 if memory == "Yes" else 0
behavior = 1 if behavior == "Yes" else 0

# Prediction
if st.button("Predict Alzheimer’s Risk"):
    input_data = np.array([[mmse, functional, adl, memory, behavior]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.error("⚠ High Risk of Alzheimer’s Disease")
    else:
        st.success("Low Risk of Alzheimer’s Disease")

    st.write(f"**Risk Probability:** {probability:.2f}")

    # Risk Levels
    if probability < 0.3:
        st.info("Risk Level: Low")
    elif probability < 0.7:
        st.warning("Risk Level: Moderate")
    else:
        st.error("Risk Level: High")