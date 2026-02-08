import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from PIL import Image

# Load trained model
model = pk.load(open('model.pkl', 'rb'))


# App title

st.header('Cars Price Prediction')

# Load dataset
cars_data = pd.read_csv('Cardetails.csv')

# Function to extract brand name


def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()


# Preprocess dataset
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Streamlit inputs
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox(
    'Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 2, 10)

# Conversion rate from INR to NPR (approximate)
INR_TO_NPR = 1.6  # 1 INR ≈ 1.6 NPR, adjust as needed

# Programmer Info in sidebar
logo = Image.open('subodh.png')
col1, col2, col3 = st.sidebar.columns([1, 2, 1])  # middle column is bigger
with col2:
    st.image(logo, width=100)

st.sidebar.markdown(
    """
    <div style="text-align: center; margin-top: 10px; font-size: 12px; color: gray;">
        KCE081BCT043<br>Subodh Madai
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-top: 5px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Prediction button
if st.button("Predict"):
    # Prepare input dataframe
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission,
          owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission',
                 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encode categorical values
    input_data_model['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner',
            'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5], inplace=True
    )
    input_data_model['fuel'].replace(
        ['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True
    )
    input_data_model['seller_type'].replace(
        ['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True
    )
    input_data_model['transmission'].replace(
        ['Manual', 'Automatic'], [1, 2], inplace=True
    )
    input_data_model['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
         'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
         'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
         'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
        list(range(1, 32)), inplace=True
    )

    # Predict car price
    predicted_price_in_inr = model.predict(input_data_model)[0]

    # Clip negative or unrealistically low prices
    predicted_price_in_inr = max(
        predicted_price_in_inr, 50000)  # Minimum realistic price

    # Convert to Nepali Rupees
    predicted_price_in_npr = predicted_price_in_inr * INR_TO_NPR

    # Display prediction neatly
    st.success(
        f"**Predicted Car Price:** ₹{predicted_price_in_inr:,.0f} (INR) ≈ NPR {predicted_price_in_npr:,.0f}")
