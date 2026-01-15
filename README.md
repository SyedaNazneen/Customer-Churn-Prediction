# 🏢 Customer Churn Prediction Project

## 1. What is this project?
This project helps businesses find out which customers might stop using their services. In business, it is much cheaper to keep an old customer than to find a new one. This system uses **Machine Learning** to give an "Early Warning" so the company can talk to at-risk customers and keep them.

## 2. Why is this important?
When customers leave (Churn), the company loses money. Many companies don't know why customers are leaving. This project looks at data like:
- How long they have been a customer (**Tenure**)
- Their type of **Contract**
- How much they pay (**Monthly Charges**)

## 3. How the project is organized (OOPS Style)
We used **Object-Oriented Programming (OOPS)**. This means the code is divided into different "Classes" to keep it clean and professional.



### Folder Structure:
- `data/`: Holds the dataset file.
- `logs/`: Keeps a record of everything that happens when the code runs.
- `models/`: Saves the final trained "Brain" of the project.
- `visuals/`: Saves the charts and graphs.
- `src/`: The main coding files:
    - `logger.py`: For tracking actions.
    - `data_handler.py`: For loading and cleaning data.
    - `eda_processor.py`: For making graphs.
    - `missing_values.py`: For fixing empty data.
    - `model_trainer.py`: For training the AI.

## 4. Technology Used
- **Python**: The main language.
- **Pandas & NumPy**: For handling data.
- **Scikit-Learn**: For the Machine Learning model.
- **Matplotlib & Seaborn**: For making charts.



## 5. Main Results
- **Accuracy**: Our model is **79.2%** accurate.
- **What we found**:
    - People with **Month-to-month** contracts leave the most.
    - **New customers** are more likely to leave than old ones.
    - Customers with **high bills** leave more often.



## 7. Future Goals
- Create a website for this project using **Streamlit**.
- Use more advanced AI like **Deep Learning**.
- Add a feature to read customer feedback.

---
**Prepared by:** Syeda Nazneen  
**Date:** 15 January 2026