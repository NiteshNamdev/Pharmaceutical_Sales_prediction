import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import joblib

# Load your trained model
model_path = r"C:\Users\NITESH NAMDEV\Nexthike projects\Pharmaceutical_Sales_Project 6\18-11-2024-23-31-40-476.pkl"
model = joblib.load(model_path)

# Title
st.title("Rossmann Sales Prediction")

# Sidebar for input
st.sidebar.header("Input Features")
store = st.sidebar.number_input("Store", min_value=1, step=1)
day_of_week = st.sidebar.number_input("Day of Week", min_value=1, max_value=7, step=1)
open_status = st.sidebar.selectbox("Open Status", [0, 1])
promo = st.sidebar.selectbox("Promo", [0, 1])
state_holiday = st.sidebar.selectbox("State Holiday", [0, 1])
school_holiday = st.sidebar.selectbox("School Holiday", [0, 1])
store_type = st.sidebar.number_input("Store Type", min_value=1, step=1)
assortment = st.sidebar.number_input("Assortment", min_value=1, step=1)
competition_distance = st.sidebar.number_input("Competition Distance", min_value=0.0, step=1.0)
competition_open_month = st.sidebar.number_input("Competition Open Month", min_value=1, max_value=12, step=1)
competition_open_year = st.sidebar.number_input("Competition Open Year", min_value=1900, max_value=2100, step=1)
promo2 = st.sidebar.selectbox("Promo2", [0, 1])
promo2_since_week = st.sidebar.number_input("Promo2 Since Week", min_value=1, max_value=52, step=1)
promo2_since_year = st.sidebar.number_input("Promo2 Since Year", min_value=1900, max_value=2100, step=1)
promo_interval = st.sidebar.text_input("Promo Interval", value="None")
weekday = st.sidebar.number_input("Weekday", min_value=1, max_value=7, step=1)
is_weekend = st.sidebar.selectbox("Is Weekend", [0, 1])
sales_per_customer = st.sidebar.number_input("Sales Per Customer", min_value=0.0, step=0.1)
is_month_start = st.sidebar.selectbox("Is Month Start", [0, 1])
is_month_middle = st.sidebar.selectbox("Is Month Middle", [0, 1])
is_month_end = st.sidebar.selectbox("Is Month End", [0, 1])

# Handle promo_interval separately
if promo_interval == "None":
    promo_interval = 0
else:
    promo_interval = float(promo_interval)

# Prediction button
if st.button("Predict Sales for Next 6 Months"):
    # Feature preparation
    input_features_dict = {
        'Store': store,
        'DayOfWeek': day_of_week,
        'Open': open_status,
        'Promo': promo,
        'StateHoliday': state_holiday,
        'SchoolHoliday': school_holiday,
        'StoreType': store_type,
        'Assortment': assortment,
        'CompetitionDistance': competition_distance,
        'CompetitionOpenSinceMonth': competition_open_month,
        'CompetitionOpenSinceYear': competition_open_year,
        'Promo2': promo2,
        'Promo2SinceWeek': promo2_since_week,
        'Promo2SinceYear': promo2_since_year,
        'PromoInterval': promo_interval,
        'weekday': weekday,
        'is_weekend': is_weekend,
        'IsMonthStart': is_month_start,
        'IsMonthMiddle': is_month_middle,
        'IsMonthEnd': is_month_end,
    }

    selected_feature_names = list(input_features_dict.keys())

    feature_indices = {feature: idx for idx, feature in enumerate(selected_feature_names)}

    input_features = np.zeros(len(selected_feature_names))
    for feature, value in input_features_dict.items():
        idx = feature_indices.get(feature)
        if idx is not None:
            input_features[idx] = value

    input_features = input_features.reshape(1, 1, -1)

    # Predict for the next 6 months
    predictions_next_6_months = []
    start_date = pd.to_datetime("2023-09-01")
    for i in range(6):
        predicted_values = model.predict(input_features)
        predicted_value = predicted_values[0][0]
        predictions_next_6_months.append(predicted_value)

        # Update input features
        input_features[0][0][1] += 7
        input_features[0][0][17] += 1
        input_features[0][0][19] += 1

    plot_dates = [start_date + pd.DateOffset(months=i) for i in range(6)]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(plot_dates, predictions_next_6_months, marker='o', linestyle='-', color='green')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Rossmann Sales Prediction for Next 6 Months")
    plt.grid(True)

    st.pyplot(plt)

    # Show predictions
    for date, prediction in zip(plot_dates, predictions_next_6_months):
        st.write(f"Date: {date.strftime('%Y-%m')} - Predicted Sales: {prediction:.2f}")