import streamlit as st
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "machine_failure_model.joblib"

st.set_page_config(page_title="Machine Failure Predictor", page_icon="⚙️")
st.title("Industrial Machine Failure Prediction")
st.write("Educational ML prototype: predicts whether a machine is likely to fail within the defined operating window.")

model = joblib.load(MODEL_PATH)

st.sidebar.header("Machine inputs")
temperature = st.sidebar.number_input("Temperature", min_value=40.0, max_value=110.0, value=72.0, step=0.5)
vibration = st.sidebar.number_input("Vibration", min_value=0.5, max_value=12.0, value=4.8, step=0.1)
pressure = st.sidebar.number_input("Pressure", min_value=2.0, max_value=9.0, value=5.5, step=0.1)
runtime = st.sidebar.number_input("Runtime", min_value=50.0, max_value=6000.0, value=1500.0, step=50.0)
maintenance_gap = st.sidebar.number_input("Maintenance gap", min_value=1.0, max_value=150.0, value=25.0, step=1.0)

if st.button("Predict"):
    row = pd.DataFrame([{
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "runtime": runtime,
        "maintenance_gap": maintenance_gap
    }])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    if prediction == 1:
        st.error("Prediction: Higher failure risk")
    else:
        st.success("Prediction: Lower failure risk")

    st.metric("Estimated failure probability", f"{probability:.1%}")
    st.caption("This is an educational prototype and should not be used as an operational maintenance decision system.")
