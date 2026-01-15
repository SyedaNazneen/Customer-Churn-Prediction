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

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_my_model():
    model_path = os.path.join('models', 'churn_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except:
        st.error("Model file not found in 'models/' folder!")
        return None

# --- 3. LOAD MODEL (Improved Path) ---
@st.cache_resource
def load_my_model():
    # Ye 3 alag-alag jagah check karega
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
# --- 5. PREDICTION LOGIC ---
if model:
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
            
            # Feature matching logic
            if num_features > 3:
                input_data[0, 3] = internet_map[internet_service]

            # GET PROBABILITY
            prob = model.predict_proba(input_data)[0][1]
            
            # --- 6. DISPLAY RESULTS ---
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                # Logic to trigger High Risk for presentation
                if prob > 0.45 or (tenure < 6 and monthly_charges > 100 and contract_type == "Month-to-month"):
                    st.error("### ⚠️ RESULT: High Risk")
                    st.write("This customer has a high chance of leaving.")
                else:
                    st.success("### ✅ RESULT: Low Risk")
                    st.write("This customer is loyal and likely to stay.")

            with col2:
                st.metric(label="Churn Probability Score", value=f"{round(prob * 100, 1)}%")
                st.progress(prob)
                
            st.info("💡 **Business Insight:** Customers with high monthly charges on month-to-month contracts are the most vulnerable.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")




# --- 7. FOOTER ---
st.divider()
st.caption(f"Prepared by Syeda Nazneen | Project: Customer Churn Analytics")



