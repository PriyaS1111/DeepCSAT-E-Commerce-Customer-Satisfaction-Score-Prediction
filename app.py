import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load("best_xgb_deepcsat.joblib")

st.title("🛒 DeepCSAT – E-commerce Customer Satisfaction Predictor")

st.markdown("""
Predict the **Customer Satisfaction (CSAT)** score based on interaction details.
Fill in the fields below and click **Predict**.
""")

remarks = st.text_area("Customer Remarks", "The agent was polite and helpful.")
response_time_min = st.number_input("Response Time (minutes)", min_value=0.0, value=5.0)
item_price = st.number_input("Item Price", min_value=0.0, value=499.0)
agent_shift = st.selectbox("Agent Shift", ["Morning", "Evening", "Night"])
agent_avg_csat = st.slider("Agent's Average CSAT", 1.0, 5.0, 4.0)

if st.button("Predict CSAT"):
    X_new = pd.DataFrame({
        "response_time_min": [response_time_min],
        "item_price": [item_price],
        "agent_avg_csat": [agent_avg_csat],
        "remarks_len": [len(remarks)],
        "remarks_word_count": [len(remarks.split())],
    })
    pred = model.predict(X_new)[0] + 1
    st.success(f"Predicted Customer Satisfaction Score: ⭐ {int(pred)} / 5")
