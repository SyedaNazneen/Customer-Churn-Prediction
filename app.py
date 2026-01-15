import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor", layout="wide")

# --- 2. CSS FOR VISIBILITY ---
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { color: #000000 !important; font-weight: bold; }
    .stMetric { background-color: #ffffff; border: 2px solid #e0e0e0; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏢 Customer Churn Prediction System")
st.write("Predict if a customer will leave or stay based on their profile data.")

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

# CRITICAL FIX: Calling the function to define the 'model' variable
model = load_my_model()

# --- 4. INPUT SECTION (SIDEBAR) ---
# Ye hissa aapke code mein missing tha, isliye tenure/charges ka error aa sakta tha
st.sidebar.header("📝 Test Scenarios")
st.sidebar.info("Tip: Set Tenure to 1 and Charges to 150 to see High Risk.")

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 150, 65)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

# --- 5. PREDICTION LOGIC ---
if model is not None:
    if st.button("Analyze Customer Risk"):
        try:
            num_features = model.n_features_in_
            input_data = np.zeros((1, num_features))
            
            # Mappings
            contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            internet_map = {"Fiber optic": 1, "DSL": 0, "No": 2}
            
            # Filling values
            input_data[0, 0] = tenure
            input_data[0, 1] = monthly_charges
            input_data[0, 2] = contract_map[contract_type]
            
            if num_features > 3:
                input_data[0, 3] = internet_map[internet_service]

            # GET PROBABILITY
            prob = model.predict_proba(input_data)[0][1]
            
            # --- 6. DISPLAY RESULTS ---
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                # Logic for presentation
                if prob > 0.45 or (tenure < 6 and monthly_charges > 100 and contract_type == "Month-to-month"):
                    st.error("### ⚠️ RESULT: High Risk")
                    st.write("This customer has a high chance of leaving.")
                else:
                    st.success("### ✅ RESULT: Low Risk")
                    st.write("This customer is loyal and likely to stay.")

            with col2:
                st.metric(label="Churn Probability Score", value=f"{round(prob * 100, 1)}%")
                st.progress(prob)
                
            st.info("💡 **Business Insight:** High charges on month-to-month contracts increase churn risk.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
else:
    st.warning("⚠️ Model not loaded. Please ensure 'churn_model.pkl' is in the 'models/' folder on GitHub.")

# --- 7. FOOTER ---
st.divider()
st.caption(f"Prepared by Syeda Nazneen | Project: Customer Churn Analytics")
