# 📊 Telecom Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Project%20Status-Completed-brightgreen?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge&logo=streamlit)

## 🎯 Project Roadmap & Progress
This project helps a telecom company identify customers who are likely to leave. I have used a professional **Modular (OOPS)** approach to build this system.

| Stage | What I did | Status |
| :--- | :--- | :--- |
| **Stage 1** | **Data Visualization (EDA)** | ✅ Done |
| **Stage 2** | **Data Cleaning & Imputation** | ✅ Done |
| **Stage 3** | **Model Training (Random Forest)** | ✅ Done |
| **Stage 4** | **Performance Evaluation** | ✅ Done |
| **Stage 5** | **Live Web App (Streamlit)** | ✅ Done |

---

## 🔍 Stage 1: Data Analysis (Insights)
I analyzed the data to find patterns. Key findings include:
- **Contract:** Month-to-month users churn the most.
- **Tenure:** New customers leave more often than old ones.
- **Charges:** Higher monthly bills increase the risk of leaving.

## 🛠️ Stage 2: Data Cleaning & Imputation
In this stage, I prepared the data for the Machine Learning model:
- **TotalCharges Fix:** Converted `TotalCharges` from text to numbers.
- **Missing Values:** Used **Mean Imputation** to fill empty data points automatically.
- **Encoding:** Converted categories (like Gender, Contract) into numbers so the computer can understand them.



## 🤖 Stage 3: Model Training
I used the **Random Forest Classifier** for this project:
- **Why Random Forest?** It is an ensemble method that combines multiple "Decision Trees" to give more accurate and stable predictions.
- **Data Splitting:** Divided data into **80% Training** and **20% Testing** to ensure the model is tested on unseen data.



## 📈 Stage 4: Performance Evaluation
After training, I checked how good the model is:
- **Accuracy:** The model achieved **79.2%**, meaning it correctly predicts 79 out of 100 customers.
- **Logging:** Every step of the training and evaluation was recorded in the `logs/` folder for professional tracking.



## 🌐 Stage 5: Live Web App
I built a user-friendly interface using **Streamlit**:
- Users can input customer details like Tenure and Contract type.
- The app instantly shows the **Churn Probability** and tells if the customer is "High Risk" or "Low Risk".

---

## ⚙️ How the Code Works
The code is written in separate **Classes** to keep it professional:
* **DataHandler:** Handles data loading and cleaning.
* **EDAProcessor:** Generates visual charts.
* **MissingValueTechniques:** Fixes empty data.
* **ModelTrainer:** Trains and saves the AI model.

## 🚀 How to Run
1. Clone the repo.
2. Install requirements: `pip install -r requirements.txt`
3. Run the analysis: `python main.py`
4. Run the Web App: `streamlit run app.py`

## 🌐 Live Demo
You can try the live application here: [Customer Churn Predictor App](https://customer-churn-prediction-site.streamlit.app/)


Prepared by: **Syeda Nazneen** 


Date - **15 January 2026**
