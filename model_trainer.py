from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, dataframe):
        self.df = dataframe
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def run_pipeline(self):
        try:
            X = self.df.drop('Churn', axis=1)
            y = self.df['Churn']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scaling
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Training
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluation
            y_pred = self.model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            print(f"✅ Model Trained! Accuracy: {acc:.2f}")
            
        except Exception as e:
            print(f"❌ Model Error: {e}")