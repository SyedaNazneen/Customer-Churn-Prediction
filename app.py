import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- 1. PAGE SETTINGS ---
st.set_page_config(page_title="Churn Predictor", layout="wide")

# --- 2. HEADER & STYLE FIX ---
# Yahan maine CSS change kiya hai taaki text hamesha black dikhe aur saaf nazar aaye
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #ddd;
    }
    /* Result text color fix */
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏢 Customer Churn Prediction System")
st.write("Predict if a customer will leave or stay based on their profile data.")

# --- 3. LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_my_model():
    model_path = os.path.join('models', 'churn_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found! Please run the training script first.")
        return None

model = load_my_model()

# --- 4. INPUT SECTION (SIDEBAR) ---
st.sidebar.header("📝 Customer Details")

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 15.0, 150.0, 65.0)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
tech_support = st.sidebar.radio("Has Tech Support?", ["Yes", "No"])

# --- 5. PREDICTION LOGIC ---
if model:
    if st.button("Analyze Customer Risk"):
        try:
            num_features = model.n_features_in_
            input_data = np.full((1, num_features), 0.0) # Base 0.0
            
            # Mappings
            contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            internet_map = {"Fiber optic": 1, "DSL": 0, "No": 2}
            tech_map = {"Yes": 1, "No": 0}
            
            # Filling values
            input_data[0, 0] = tenure
            input_data[0, 1] = monthly_charges
            input_data[0, 2] = contract_map[contract_type]
            if num_features > 3:
                input_data[0, 3] = internet_map[internet_service]
                input_data[0, 4] = tech_map[tech_support]
            
            # Predict
            probability = model.predict_proba(input_data)[0][1]
            prediction = 1 if probability > 0.5 else 0
            
            # --- 6. DISPLAY RESULTS ---
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("### ⚠️ RESULT: High Risk")
                    st.markdown("**This customer is likely to CHURN.**")
                else:
                    st.success("### ✅ RESULT: Low Risk")
                    st.markdown("**This customer is likely to STAY.**")
            
            with col2:
                # Metric display with clear labels
                st.metric(label="Churn Probability", value=f"{round(probability * 100, 1)}%")
                st.progress(probability)
                
            st.info("💡 **Tip:** High charges and short contracts usually increase churn risk.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")


# --- 7. FOOTER ---
st.divider()
st.caption(f"Prepared by Syeda Nazneen | Project: Customer Churn Analytics")

