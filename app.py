import streamlit as st
import pandas as pd
import pickle
import os

# --- PAGE CONFIGURATION ---
# This sets the title of the website tab
st.set_page_config(page_title="Churn Predictor", layout="wide")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("🏢 Customer Churn Prediction System")
st.write("This app predicts if a customer will leave or stay with the company based on their data.")

# --- LOAD THE TRAINED MODEL ---
# Using @st.cache_resource so the model loads only once (faster performance)
@st.cache_resource
def load_my_model():
    model_path = os.path.join('models', 'churn_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found! Please run main.py first to create the model.")
        return None

model = load_my_model()

# --- INPUT SECTION ---
st.sidebar.header("Customer Information")
st.sidebar.info("Enter the details of the customer below to check their churn status.")

# Creating input fields for the user
# 1. Tenure: How many months they have been a customer
tenure = st.sidebar.slider("Tenure (How many months?)", min_value=1, max_value=72, value=12)

# 2. Monthly Charges: How much they pay every month
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=50.0)

# 3. Contract: Type of contract they signed
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# --- PREDICTION LOGIC ---
if model:
    st.subheader("Analysis Results")
    
    # When the user clicks the button
    if st.button("Calculate Churn Probability"):
        
        # Prepare the data for the model
        # Note: We match the order of features used during training
        # This is a simplified example based on common features
        contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        input_data = [[tenure, monthly_charges, contract_mapping[contract_type]]]
        
        # Make the prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] # Get probability of Churn
        
        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.error("### ⚠️ RESULT: High Risk")
                st.write("The customer is likely to **CHURN** (Leave).")
            else:
                st.success("### ✅ RESULT: Low Risk")
                st.write("The customer is likely to **STAY** (Retain).")
        
        with col2:
            st.metric(label="Churn Probability", value=f"{round(probability * 100, 2)}%")
            st.progress(probability)

# --- FOOTER ---
st.divider()
st.caption("Powered by Machine Learning | Prepared by THE SKILL UNION Student")