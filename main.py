from src.data_handler import DataHandler
from src.model_trainer import ModelTrainer

if __name__ == "__main__":
    # Apni file ka sahi path yahan likhein
    path = r'C:\Users\Nazneen\OneDrive\Desktop\Customer_Churn_Project\data\WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # 1. Data Loading
    handler = DataHandler(path)
    handler.load_data()
    
    if handler.df is not None:
        # 2. Save All Visuals
        handler.save_visuals()
        
        # 3. Clean Data and Capture the result
        clean_df = handler.clean_data()
        
        # 4. Model Training
        if clean_df is not None:
            print(f"Data Shape for Training: {clean_df.shape}") # Debugging line
            trainer = ModelTrainer(clean_df)
            trainer.run_pipeline()
    else:
        print("❌ Error: Process ruk gaya kyunki data load nahi hua.")