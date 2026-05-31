import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Page Config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction")

# Load files
try:
    model = load_model("model.ann.h5")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# User Inputs
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41.0)
total_rooms = st.number_input("Total Rooms", value=880.0)
population = st.number_input("Population", value=322.0)
households = st.number_input("Households", value=126.0)
median_income = st.number_input("Median Income", value=8.3252)

# Ocean Proximity Dropdown
ocean_proximity = st.selectbox(
    "Ocean Proximity",
    le.classes_
)

# Encode Category
ocean_encoded = le.transform([ocean_proximity])[0]

# Prediction Button
if st.button("Predict House Price"):

    features = np.array([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        population,
        households,
        median_income,
        ocean_encoded
    ]])

    try:
        # Scale Input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled, verbose=0)

        predicted_price = float(prediction[0][0])

        st.success(
            f"Predicted House Price: ₹ {predicted_price:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
