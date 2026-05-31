import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")

# Load model and scaler
try:
    model = load_model("model.ann.h5")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# User Inputs
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41.0)
total_rooms = st.number_input("Total Rooms", value=880.0)
population = st.number_input("Population", value=322.0)
households = st.number_input("Households", value=126.0)
median_income = st.number_input("Median Income", value=8.3252)
ocean_proximity = st.number_input("Ocean Proximity (Encoded Value)", value=3.0)

# Prediction
if st.button("Predict House Price"):

    try:
        features = np.array([[
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            population,
            households,
            median_income,
            ocean_proximity
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled, verbose=0)

        predicted_price = float(prediction[0][0])

        st.success(
            f"Predicted House Price: ₹ {predicted_price:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
