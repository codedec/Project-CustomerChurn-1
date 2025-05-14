import streamlit as st
import joblib
import pandas as pd

# Load model bundle
model_bundle = joblib.load("churn_model_bundle.pkl")
model = model_bundle["model"]
top_features = model_bundle["features"]
label_encoders = model_bundle["label_encoders"]
num_imputer = model_bundle["num_imputer"]

# Page Config
st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="auto"
)

# Title and Instructions
st.title("🛒 E-Commerce Customer Churn Predictor")
st.markdown("Use this tool to predict whether a customer is likely to churn based on their activity and behavior.")

st.markdown("---")

st.subheader("📋 Customer Information")

# Organize inputs into columns
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (in months)", min_value=0)
    complain = st.selectbox("Filed Any Complaint?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cashback = st.number_input("Cashback Received", min_value=0.0)
    satisfaction = st.selectbox("Satisfaction Score (1 - Worst, 5 - Best)", [1, 2, 3, 4, 5])
    last_order_days = st.number_input("Days Since Last Order", min_value=0)
    
with col2:
    address_count = st.number_input("Number of Addresses", min_value=1)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    distance = st.number_input("Distance from Warehouse (in km)", min_value=0)
    preferred_cat = st.selectbox("Preferred Order Category", [
        "Laptop & Accessory", "Mobile", "Mobile Phone", "Others", "Fashion"
    ])
    device_count = st.number_input("Devices Registered", min_value=1)

# Optional backend-used fields
default_values = {
    "CouponUsed": 1,
    "HourSpendOnApp": 3,
    "OrderAmountHikeFromlastYear": 20,
    "OrderCount": 10,
}

# Combine all input
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
    **default_values
}

input_df = pd.DataFrame([user_input])

# Prediction trigger
if st.button("🚀 Predict Churn"):
    # Label encoding
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Impute missing values
    numeric_cols = num_imputer.feature_names_in_
    input_df[numeric_cols] = num_imputer.transform(input_df[numeric_cols])

    # Select top features
    input_df_reduced = input_df[top_features]

    # Prediction
    prediction = model.predict(input_df_reduced)[0]
    probability = model.predict_proba(input_df_reduced)[0][1]

    # Output
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error("🔴 **Customer is likely to churn**")
    else:
        st.success("🟢 **Customer is likely to stay**")

    st.info(f"📊 **Churn Probability:** {probability:.2%}")

    st.markdown("---")
    st.caption("🔍 This prediction is based on behavioral and transaction data. Use it to take proactive actions.")

