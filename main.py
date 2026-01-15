import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing custom modules from the src folder
from src.logger import setup_logging
from src.data_handler import DataHandler
from src.EDAProcessor import EDAProcessor
from src.missing_values import MISSING_VALUE_TECHNIQUES
from src.model_trainer import ModelTrainer

# Initialize the main logger for the pipeline
logger = setup_logging('main_pipeline')

def run_project():
    try:
        logger.info("================ Project Execution Started ================")
        
        # 1. Define the path for the dataset
        data_path = os.path.join("data", "churn prediction.csv")
        
        # 2. Load the dataset using DataHandler
        df = DataHandler.load_data(data_path)
        if df is None:
            logger.error("Failed to load data. Script execution stopped.")
            return

        # 3. Perform Exploratory Data Analysis (EDA)
        # This will generate and save plots in the 'visuals/' folder
        logger.info("Starting EDA process...")
        EDAProcessor.perform_eda(df)
        print("📊 Visuals generated successfully. Check the 'visuals/' folder.")

        # 4. Clean the data (Handle data types and drop unnecessary columns)
        logger.info("Starting data cleaning process...")
        df_cleaned = DataHandler.clean_data(df)
        
        # 5. Handle Missing Values
        # Split the data into features (X) and target (y) before imputation
        logger.info("Splitting data for missing value imputation...")
        X = df_cleaned.drop('Churn', axis=1)
        y = df_cleaned['Churn']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply Mean Imputation technique from missing_values module
        X_train, X_test = MISSING_VALUE_TECHNIQUES.mean_imputation(X_train, X_test)
        logger.info("Missing values handled using Mean Imputation.")

        # 6. Model Training
        # Combine X_train and y_train back into one dataframe for the trainer
        final_training_df = pd.concat([X_train, y_train], axis=1)
        
        logger.info("Starting machine learning model training...")
        ModelTrainer.train_model(final_training_df)
        
        print("✅ Project executed successfully! Check 'logs' and 'models' folders.")
        logger.info("================ Project Execution Completed Successfully ================")

    except Exception as e:
        # Capture error details including the line number
        _, _, error_line = sys.exc_info()
        logger.error(f"Main execution failed at line {error_line.tb_lineno}: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    run_project()