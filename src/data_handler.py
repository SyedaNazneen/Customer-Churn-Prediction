import pandas as pd
import sys
from src.logger import setup_logging

logger = setup_logging('data_handler')

class DataHandler:
    @staticmethod
    def load_data(file_path):
        try:
            logger.info(f"Loading file from: {file_path}")
            df = pd.read_csv(file_path)
            logger.info("Data loaded successfully.")
            return df
        except Exception as e:
            _, _, error_line = sys.exc_info()
            logger.error(f"Error at line {error_line.tb_lineno}: {str(e)}")
            return None

    @staticmethod
    def clean_data(df):
        try:
            logger.info("Cleaning data...")
            # TotalCharges ko numeric banana (kuch empty strings hoti hain isliye)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df.dropna(inplace=True)
            if 'customerID' in df.columns:
                df.drop('customerID', axis=1, inplace=True)
            logger.info("Cleaning successful.")
            return df
        except Exception as e:
            logger.error(f"Cleaning failed: {str(e)}")
            return None