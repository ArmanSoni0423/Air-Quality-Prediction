# AIR QUALITY PREDICTION SYSTEM
# Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌫️",
    layout="wide"
)

# Load Trained Model
model = joblib.load("air_quality_model.pkl")

# AQI Category Function
def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


# Dashboard Header
st.title("🌫️ Air Quality Prediction System")

st.write(
    "Machine Learning based Air Quality Index (AQI) Prediction "
    "using Linear Regression."
)

st.divider()

# Input Section
st.subheader("Enter Pollution Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input(
        "PM2.5",
        min_value=0.0,
        value=50.0
    )

    pm10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=80.0
    )

with col2:
    no2 = st.number_input(
        "NO₂",
        min_value=0.0,
        value=30.0
    )

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=1.0
    )

with col3:
    so2 = st.number_input(
        "SO₂",
        min_value=0.0,
        value=10.0
    )

    o3 = st.number_input(
        "O₃",
        min_value=0.0,
        value=30.0
    )


st.divider()

# Prediction
if st.button("🔮 Predict AQI", use_container_width=True):

    input_data = pd.DataFrame({
        "PM2.5": [pm25],
        "PM10": [pm10],
        "NO2": [no2],
        "CO": [co],
        "SO2": [so2],
        "O3": [o3]
    })

    prediction = model.predict(input_data)[0]

    # AQI cannot be negative
    prediction = max(0, prediction)

    category = get_aqi_category(prediction)

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted AQI",
            f"{prediction:.2f}"
        )

    with result_col2:
        st.metric(
            "Air Quality",
            category
        )

    st.success(
        f"The predicted AQI is {prediction:.2f}, "
        f"which falls under the '{category}' category."
    )

# Footer
st.divider()

st.caption(
    "Air Quality Prediction System | "
    "Machine Learning Project | Linear Regression"
)