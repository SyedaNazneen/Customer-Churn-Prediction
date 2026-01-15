import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- 1. PAGE SETTINGS ---
st.set_page_config(page_title="Churn Predictor", layout="wide")

# --- 2. HEADER & STYLE ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏢 Customer Churn Prediction System")
st.write("Predict if a customer will leave or stay based on their profile data.")

# --- 3. LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_my_model():
    # Looking for the model in the 'models' folder
    model_path = os.path.join('models', 'churn_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found! Please run the main training script first.")
        return None

model = load_my_model()

# --- 4. INPUT SECTION (SIDEBAR) ---
st.sidebar.header("📝 Customer Details")

# Collecting main features mentioned in the dataset
tenure = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=65.0)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Additional features to make the model more responsive
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
tech_support = st.sidebar.radio("Has Tech Support?", ["Yes", "No"])

# --- 5. PREDICTION LOGIC ---
if model:
    if st.button("Analyze Customer Risk"):
        try:
            # Preparing input to match the number of features the model expects
            num_features = model.n_features_in_
            
            # Using a neutral base (0.5) instead of 0 for better sensitivity
            input_data = np.full((1, num_features), 0.5)
            
            # Mapping categorical values to numbers as done in Stage 2
            contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            internet_map = {"Fiber optic": 1, "DSL": 0, "No": 2}
            tech_map = {"Yes": 1, "No": 0}
            
            # Assigning values to specific indices (matching training order)
            input_data[0, 0] = tenure
            input_data[0, 1] = monthly_charges
            input_data[0, 2] = contract_map[contract_type]
            
            # Additional mapping for better logic
            if num_features > 3:
                input_data[0, 3] = internet_map[internet_service]
                input_data[0, 4] = tech_map[tech_support]
            
            # Getting probability and prediction
            probability = model.predict_proba(input_data)[0][1]
            prediction = 1 if probability > 0.5 else 0
            
            # --- 6. DISPLAY RESULTS ---
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("### ⚠️ RESULT: High Risk")
                    st.write("This customer is very likely to **CHURN** soon.")
                else:
                    st.success("### ✅ RESULT: Low Risk")
                    st.write("This customer is likely to **STAY** with the company.")
            
            with col2:
                # This metric will now change as you move the sliders
                st.metric(label="Churn Probability Score", value=f"{round(probability * 100, 2)}%")
                st.progress(probability)
                
            st.info("💡 **Insight:** Lowering monthly charges or offering a 2-year contract usually reduces churn risk.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- 7. FOOTER ---
st.divider()
st.caption(f"Prepared by Syeda Nazneen | Project: Customer Churn Analytics")
