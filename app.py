import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor Pro", layout="wide")

# --- 2. PREMIUM COLORFUL CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
    }
    .main-title {
        color: #fbbf24;
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 5px;
    }
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #374151;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 40px !important;
    }
    .stMetric {
        background-color: #1f2937;
        border: 1px solid #38bdf8;
        padding: 20px;
        border-radius: 15px;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 12px;
        border-radius: 10px;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px #10b981;
    }
    .stMarkdown, p, label {
        color: #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
MODEL_PATH = "artifacts/final_model_pipeline.pkl"

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_my_model()

# --- 4. HEADER ---
st.markdown("<h1 class='main-title'>📊 Customer Churn Analytics Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Advanced AI Dashboard for Telecom Retention</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 5. SIDEBAR (SIM PROVIDER) ---
st.sidebar.markdown("<h2 style='color: #fbbf24;'>📱 SIM Provider</h2>", unsafe_allow_html=True)
sim = st.sidebar.radio("Select provider", ["Jio", "Airtel", "Vi", "BSNL"])
st.sidebar.info(f"Currently Analyzing: {sim}")

# --- 6. MAIN INPUT FORM ---
if model is None:
    st.error(f"⚠️ Model file not found at {MODEL_PATH}. Please check your 'artifacts' folder.")
else:
    with st.container():
        st.subheader("📝 Customer Information")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (Months)", 0, 100, 12)

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    # Prediction Section
    st.markdown("---")
    _, mid_col, _ = st.columns([1, 1, 1])
    
    with mid_col:
        predict_btn = st.button("🔮 ANALYZE RISK")

    if predict_btn:
        # Create Input DataFrame
        input_df = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
            # "SIM": sim # Add this if your model was trained with SIM column
        }])

        # Get Prediction
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        # --- 7. RESULTS DISPLAY ---
        st.markdown("### 📈 Analysis Results")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK: LIKELY TO CHURN")
                st.write("Customer shows signs of leaving. Recommendation: Offer a retention discount.")
            else:
                st.success("### ✅ LOW RISK: LIKELY TO STAY")
                st.write("Customer is satisfied with the current services.")

        with res_col2:
            st.metric(label="Churn Probability", value=f"{prob:.2%}")
            st.progress(prob)

# --- 8. FOOTER ---
st.markdown("---")
st.caption("Developed for Churn Prediction Project | ML Pipeline v1.0")
