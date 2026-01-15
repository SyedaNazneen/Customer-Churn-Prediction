import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from src.logger import setup_logging

logger = setup_logging('model_trainer')

class ModelTrainer:
    @staticmethod
    def train_model(df):
        try:
            logger.info("Splitting data into Train and Test")
            # Encoding categorical values
            df_encoded = pd.get_dummies(df, drop_first=True)
            
            X = df_encoded.drop('Churn_Yes', axis=1)
            y = df_encoded['Churn_Yes']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            
            logger.info(f"Model trained with accuracy: {model.score(X_test, y_test)}")
            
            # Save model (File Handling)
            os.makedirs('models', exist_ok=True)
            with open('models/churn_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            logger.info("Model saved in models/ folder")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")