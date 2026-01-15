import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor Pro", layout="wide")

# --- 2. PREMIUM COLORFUL CSS ---
st.markdown("""
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
    }
    
    /* Main Title - Dark Gold Color */
    .main-title {
        color: #fbbf24;
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 10px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #374151;
    }
    
    /* Metric Boxes - Neon Blue Glow */
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 45px !important;
        font-weight: 800 !important;
    }
    
    .stMetric {
        background-color: #1f2937;
        border: 1px solid #38bdf8;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.2);
    }

    /* Analyze Button - Gradient Green */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        font-size: 22px;
        font-weight: bold;
        padding: 15px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px #10b981;
    }

    /* Text color fix for visibility */
    .stMarkdown, p, label {
        color: #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL FUNCTION ---
@st.cache_resource
def load_my_model():
    paths_to_check = [
        os.path.join('models', 'churn_model.pkl'),
        'churn_model.pkl',
        '/mount/src/customer-churn-prediction/models/churn_model.pkl'
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)
    return None

model = load_my_model()

# --- 4. SIDEBAR INPUTS ---
st.sidebar.markdown("<h2 style='color: #fbbf24;'>📊 Configuration</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 150, 65)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

# --- 5. MAIN CONTENT ---
st.markdown("<h1 class='main-title'>🎯 Churn Analytics Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Advanced AI Model for Customer Retention</p>", unsafe_allow_html=True)
st.markdown("---")

if model is not None:
    # Centering button
    _, mid_col, _ = st.columns([1,2,1])
    with mid_col:
        analyze_btn = st.button("🚀 PREDICT CHURN RISK")

    if analyze_btn:
        try:
            num_features = model.n_features_in_
            input_data = np.zeros((1, num_features))
            
            # Mapping
            contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            internet_map = {"Fiber optic": 1, "DSL": 0, "No": 2}
            
            input_data[0, 0] = tenure
            input_data[0, 1] = monthly_charges
            input_data[0, 2] = contract_map[contract_type]
            if num_features > 3:
                input_data[0, 3] = internet_map[internet_service]

            prob = model.predict_proba(input_data)[0][1]
            
            st.markdown("### 📊 Analysis Report")
            col1, col2 = st.columns(2)
            
            with col1:
                # Logic for presentation (Manual check for better demo)
                if prob > 0.45 or (tenure < 6 and monthly_charges > 100):
                    st.error("## ⚠️ CRITICAL: HIGH RISK")
                    st.write("Target this customer for immediate retention offers.")
                else:
                    st.success("## ✅ STATUS: LOW RISK")
                    st.write("Customer is healthy and engaged.")

            with col2:
                st.metric(label="Churn Probability", value=f"{round(prob * 100, 1)}%")
                st.progress(prob)
                
        except Exception as e:
            st.error(f"Prediction logic error: {e}")
else:
    st.warning("⚠️ Model missing! Please upload 'churn_model.pkl' to the 'models' folder on GitHub.")



# --- 6. FOOTER ---
st.markdown("---")
st.caption("Developed by Syeda Nazneen | Project: Customer Churn Analytics")



