import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor Pro", layout="wide")

# --- 2. COLORFUL CSS (Custom Styling) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    
    /* Title Styling */
    h1 {
        color: #1E3A8A;
        font-family: 'Helvetica', sans-serif;
        text-align: center;
        padding-bottom: 20px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
        color: #F8FAFC !important;
    }

    /* Metric Boxes */
    div[data-testid="stMetricValue"] {
        color: #2563EB !important;
        font-size: 40px;
        font-weight: bold;
    }
    
    .stMetric {
        background-color: #ffffff;
        border-left: 5px solid #2563EB;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
    }

    /* Button Styling */
    div.stButton > button:first-child {
        background-color: #10B981;
        color: white;
        font-size: 20px;
        font-weight: bold;
        width: 100%;
        border-radius: 10px;
        height: 3em;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #059669;
        border: 2px solid white;
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

# --- 4. INPUT SECTION (SIDEBAR) ---
st.sidebar.title("📊 Settings")
st.sidebar.markdown("---")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 150, 65)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

# --- 5. MAIN INTERFACE ---
st.title("🎯 Customer Churn Analytics Dashboard")
st.write("---")

if model is not None:
    # Centering the button
    left_col, mid_col, right_col = st.columns([1,2,1])
    with mid_col:
        analyze_btn = st.button("🚀 RUN ANALYSIS")

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
            
            st.markdown("### 📈 Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if prob > 0.45 or (tenure < 6 and monthly_charges > 100):
                    st.error("## ⚠️ HIGH RISK")
                    st.warning("Action Required: This customer might leave soon!")
                else:
                    st.success("## ✅ LOW RISK")
                    st.info("Status: This customer is satisfied and loyal.")

            with col2:
                st.metric(label="Churn Probability", value=f"{round(prob * 100, 1)}%")
                st.progress(prob)
                
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning("⚠️ Model not found. Check GitHub 'models' folder.")

# --- 6. FOOTER ---
st.markdown("---")
st.caption("Developed by Syeda Nazneen | Project: Customer Churn Analytics")


