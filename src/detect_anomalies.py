import pandas as pd
import joblib

class AnomalyDetector:
    def __init__(self, model_path="models/isolation_forest.pkl"):
        self.model = joblib.load(model_path)
        self.features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

    def predict(self, input_data):
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data], columns=self.features)
        else:
            input_df = input_data[self.features]
        
        prediction = self.model.predict(input_df)
        return 1 if prediction[0] == -1 else 0

# Example usage
detector = AnomalyDetector()
user_input = {
    'Transaction_Amount': 1500,
    'Average_Transaction_Amount': 200,
    'Frequency_of_Transactions': 5
}
print("Anomaly Detected!" if detector.predict(user_input) else "Normal Transaction")