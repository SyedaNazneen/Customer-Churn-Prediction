import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.original_df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            self.original_df = self.df.copy() # Visuals ke liye original copy
            print("✅ Data Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def save_visuals(self, output_folder='visuals'):
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 1. Numerical Histograms
            num_features = ['tenure', 'MonthlyCharges']
            plt.figure(figsize=(12, 5))
            for i, col in enumerate(num_features):
                plt.subplot(1, 2, i+1)
                sns.histplot(self.original_df[col], kde=True, color='blue')
                plt.title(f'Distribution of {col}')
            plt.savefig(f'{output_folder}/numerical_dist.png')
            plt.close()

            # 2. Categorical Churn Comparison (Top important ones)
            cat_features = ['gender', 'Contract', 'InternetService', 'PaymentMethod']
            for col in cat_features:
                plt.figure(figsize=(8, 5))
                sns.countplot(data=self.original_df, x=col, hue='Churn', palette='viridis')
                plt.title(f'Churn Comparison by {col}')
                plt.savefig(f'{output_folder}/{col}_vs_churn.png')
                plt.close()

            print(f"✅ All Visuals saved in '{output_folder}' folder!")
        except Exception as e:
            print(f"❌ Visualization Error: {e}")

    def clean_data(self):
        try:
            # Step 1: TotalCharges fixed
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            self.df.dropna(inplace=True)

            # Step 2: STRICTLY DROP customerID (Forcefully)
            # Hum columns ki list check karke drop karenge
            cols_to_drop = ['customerID']
            for col in cols_to_drop:
                if col in self.df.columns:
                    self.df = self.df.drop(columns=[col])
                    print(f"🗑️ Successfully removed: {col}")

            # Step 3: Churn Encoding
            if 'Churn' in self.df.columns:
                self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})

            # Step 4: One-Hot Encoding (Object to Numbers)
            self.df = pd.get_dummies(self.df, drop_first=True)
            
            # Step 5: Final Check - Ensure everything is numeric
            self.df = self.df.select_dtypes(include=[np.number])

            print("✅ Data Cleaning Completed!")
            return self.df
        except Exception as e:
            print(f"❌ Cleaning Error: {e}")
            return None