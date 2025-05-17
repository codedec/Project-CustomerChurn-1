import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Custom CSS for black background and modern UI ---
st.markdown("""
    <style>
    .stApp {
        background: #111111;
        min-height: 100vh;
    }
    h1, h3, h5, p, label, .stMarkdown, .stDataFrame, .stSelectbox, .stNumberInput, .stButton>button {
        color: #f8fafc !important;
    }
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%);
        color: #f8fafc !important;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        margin-top: 1rem;
    }
    .stSelectbox>div>div {
        border-radius: 8px !important;
        color: #111 !important;
    }
    .stNumberInput>div>div>input {
        border-radius: 8px !important;
        color: #111 !important;
    }
    .stDataFrame {
        background: #222 !important;
        color: #f8fafc !important;
    }
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #f8fafc;'>🛒 E-Commerce Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:1.2rem; color: #f8fafc;'>Enter customer details to predict churn probability.</p>", unsafe_allow_html=True)
st.markdown("---")

# Load model bundle
model_bundle = joblib.load("churn_model_bundle.pkl")
model = model_bundle["model"]
top_features = model_bundle["features"]
label_encoders = model_bundle["label_encoders"]
num_imputer = model_bundle["num_imputer"]

# Centered layout using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h3 style='text-align:center; color: #f8fafc;'>📋 Customer Features</h3>", unsafe_allow_html=True)
    with st.form("churn_form", clear_on_submit=False):
        user_input = {}
        # Split features into two columns if many features
        if len(top_features) > 6:
            fcol1, fcol2 = st.columns(2)
            cols = [fcol1, fcol2]
        else:
            cols = [st]
        for idx, feat in enumerate(top_features):
            help_text = f"Enter value for {feat.replace('_', ' ')}"
            col = cols[idx % len(cols)]
            with col:
                if feat in label_encoders:
                    options = list(label_encoders[feat].classes_)
                    user_input[feat] = st.selectbox(f"{feat}", options, help=help_text, key=feat)
                else:
                    dtype = float
                    min_val = 0.0
                    max_val = 10000.0
                    default_val = 0.0
                    step_val = 0.01
                    fmt = "%.2f"
                    if hasattr(num_imputer, "feature_names_in_"):
                        try:
                            idx2 = list(num_imputer.feature_names_in_).index(feat)
                            stat_val = num_imputer.statistics_[idx2]
                            if isinstance(stat_val, (int, np.integer)):
                                dtype = int
                                min_val = int(0)
                                max_val = int(10000)
                                default_val = int(stat_val)
                                step_val = 1
                                fmt = "%d"
                            else:
                                dtype = float
                                min_val = float(0)
                                max_val = float(10000)
                                default_val = float(stat_val)
                                step_val = 0.01
                                fmt = "%.2f"
                        except Exception:
                            pass
                    user_input[feat] = st.number_input(
                        f"{feat}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step_val,
                        format=fmt,
                        help=help_text,
                        key=feat
                    )
        submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)

# Build a DataFrame with all columns the imputer expects
all_cols = list(num_imputer.feature_names_in_)
full_input = {col: np.nan for col in all_cols}
for feat in top_features:
    full_input[feat] = user_input[feat]
input_df = pd.DataFrame([full_input])

# Encode categorical features
for col, le in label_encoders.items():
    if col in input_df.columns and pd.notnull(input_df[col].iloc[0]):
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            st.error(f"Input value for '{col}' not recognized. Please select a valid option.")
            st.stop()

# Impute missing numeric values (now safe)
input_df[all_cols] = num_imputer.transform(input_df[all_cols])

if submitted:
    with col2:
        st.markdown("<h3 style='text-align:center; color: #f8fafc;'>🎯 Prediction Result</h3>", unsafe_allow_html=True)
        input_for_model = input_df[top_features]
        probability = model.predict_proba(input_for_model)[0][1]
        prediction = 1 if probability > 0.5 else 0

        if prediction == 1:
            st.error(f"🔴 Prediction: Churn", icon="🚨")
        else:
            st.success(f"🟢 Prediction: Stay", icon="✅")
        st.info(f"Churn Probability: {probability:.2%}")




