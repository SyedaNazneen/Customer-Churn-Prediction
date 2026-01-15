import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("🏢 Customer Churn Prediction System")
st.write("This app predicts if a customer will leave or stay based on their data.")

# --- LOAD THE MODEL ---
@st.cache_resource
def load_my_model():
    # Make sure your model file is in 'models/' folder
    model_path = os.path.join('models', 'churn_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found! Please run main.py first.")
        return None

model = load_my_model()

# --- INPUT SECTION ---
st.sidebar.header("Customer Information")

# Main inputs
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 10.0, 200.0, 50.0)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# --- PREDICTION LOGIC ---
if model:
    if st.button("Calculate Churn Probability"):
        try:
            # 1. Map the contract type to numbers
            contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            
            # 2. Match the model's expected feature count
            # This creates a row of zeros for all expected columns
            num_features = model.n_features_in_
            input_array = np.zeros((1, num_features))
            
            # 3. Fill the inputs we have (Tenure, Charges, Contract)
            # Assuming these were the first 3 features during training
            input_array[0, 0] = tenure
            input_array[0, 1] = monthly_charges
            input_array[0, 2] = contract_mapping[contract_type]
            
            # 4. Make prediction
            prediction = model.predict(input_array)
            probability = model.predict_proba(input_array)[0][1]
            
            # --- DISPLAY RESULTS ---
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 1:
                    st.error("### ⚠️ RESULT: High Risk")
                    st.write("This customer is likely to **CHURN**.")
                else:
                    st.success("### ✅ RESULT: Low Risk")
                    st.write("This customer is likely to **STAY**.")
            
            with col2:
                st.metric(label="Churn Probability", value=f"{round(probability * 100, 2)}%")
                st.progress(probability)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info(f"Model expects {model.n_features_in_} features.")

st.caption("Prepared by Vihara Tech Student : Syeda Nazneen | Project Title: Customer Churn Prediction")

