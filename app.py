import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model.ann.h5")
scaler = joblib.load("scaler.pkl")

st.title("House Price Prediction")

longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")
ocean_proximity = st.number_input("Ocean Proximity")

if st.button("Predict"):

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

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    st.success(f"Predicted House Price: ${prediction[0][0]:,.2f}")