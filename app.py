import streamlit as st
import joblib
import pandas as pd

model_bundle = joblib.load("churn_model_bundle.pkl")
model = model_bundle["model"]
top_features = model_bundle["features"]
label_encoders = model_bundle["label_encoders"]
num_imputer = model_bundle["num_imputer"]

st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 E-Commerce Customer Churn Predictor")
st.markdown("Predict customer churn likelihood using behavior data.")
st.markdown("---")

st.subheader("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0)
    complain = st.selectbox("Filed Any Complaint?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cashback = st.number_input("Cashback Received", min_value=0.0)
    satisfaction = st.selectbox("Satisfaction Score (1-5)", [1, 2, 3, 4, 5])
    last_order_days = st.number_input("Days Since Last Order", min_value=0)

with col2:
    address_count = st.number_input("Number of Addresses", min_value=1)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    distance = st.number_input("Distance from Warehouse (km)", min_value=0)
    preferred_cat = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile", "Mobile Phone", "Others", "Fashion"])
    device_count = st.number_input("Devices Registered", min_value=1)

# Backend-used default values
default_values = {
    "CouponUsed": 1,
    "HourSpendOnApp": 3,
    "OrderAmountHikeFromlastYear": 20,
    "OrderCount": 10,
}

user_input = {
    "Tenure": tenure,
    "Complain": complain,
    "CashbackAmount": cashback,
    "SatisfactionScore": satisfaction,
    "DaySinceLastOrder": last_order_days,
    "NumberOfAddress": address_count,
    "CityTier": city_tier,
    "WarehouseToHome": distance,
    "PreferedOrderCat": preferred_cat,
    "NumberOfDeviceRegistered": device_count,
    **default_values,
}

input_df = pd.DataFrame([user_input])

if st.button("🚀 Predict Churn"):
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    numeric_cols = num_imputer.feature_names_in_
    input_df[numeric_cols] = num_imputer.transform(input_df[numeric_cols])

    input_df_reduced = input_df[top_features]
    prediction = model.predict(input_df_reduced)[0]
    probability = model.predict_proba(input_df_reduced)[0][1]

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error("🔴 Customer is likely to churn")
    else:
        st.success("🟢 Customer is likely to stay")

    st.info(f"📊 Churn Probability: {probability:.2%}")
    st.caption("🔍 Prediction based on behavior & transaction data.")
