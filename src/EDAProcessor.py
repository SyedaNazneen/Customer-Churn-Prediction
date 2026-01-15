import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from src.logger import setup_logging

logger = setup_logging('eda_processor')

class EDAProcessor:
    @staticmethod
    def perform_eda(df):
        try:
            logger.info("EDA process is Starting...")
            
            # checking Visuals folder 
            output_dir = 'visuals'
            os.makedirs(output_dir, exist_ok=True)

            # 1. Churn (Dependent) vs Gender (Independent)
            plt.figure(figsize=(8, 5))
            sns.countplot(x='gender', hue='Churn', data=df, palette='viridis')
            plt.title('Churn Count by Gender')
            plt.savefig(f'{output_dir}/gender_vs_churn.png')
            plt.close()

            # 2. Tenure (Independent) vs Churn (Dependent) - Distribution Plot
            plt.figure(figsize=(8, 5))
            sns.histplot(data=df, x='tenure', hue='Churn', kde=True, palette='magma')
            plt.title('Tenure Distribution by Churn')
            plt.savefig(f'{output_dir}/tenure_vs_churn.png')
            plt.close()

            # 3. Monthly Charges vs Churn
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
            plt.title('Monthly Charges vs Churn')
            plt.savefig(f'{output_dir}/monthly_charges_vs_churn.png')
            plt.close()

            # 4. Contract Type vs Churn (most important feature)
            plt.figure(figsize=(8, 5))
            sns.countplot(x='Contract', hue='Churn', data=df)
            plt.title('Contract Type vs Churn')
            plt.savefig(f'{output_dir}/contract_vs_churn.png')
            plt.close()

            logger.info(f"Saare visuals '{output_dir}/' folder mein save ho gaye hain.")
        
        except Exception as e:
            _, _, error_line = sys.exc_info()
            logger.error(f"EDA failed at line {error_line.tb_lineno}: {str(e)}")