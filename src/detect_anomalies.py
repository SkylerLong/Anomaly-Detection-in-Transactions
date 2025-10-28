import pandas as pd
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, model_path="models/isolation_forest.pkl"):
        try:
            self.model = joblib.load(model_path)
            self.features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
            logger.info(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, input_data):
        """
        Predict if input data is an anomaly.
        
        Args:
            input_data: Dictionary or DataFrame with transaction features
        
        Returns:
            int: 1 for anomaly, 0 for normal transaction
        """
        try:
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data], columns=self.features)
            else:
                input_df = input_data[self.features]
            
            prediction = self.model.predict(input_df)
            result = 1 if prediction[0] == -1 else 0
            logger.info(f"Prediction completed: {'Anomaly' if result else 'Normal'}")
            return result
        except KeyError as e:
            logger.error(f"Missing required feature: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

# Example usage
detector = AnomalyDetector()
user_input = {
    'Transaction_Amount': 1500,
    'Average_Transaction_Amount': 200,
    'Frequency_of_Transactions': 5
}
print("Anomaly Detected!" if detector.predict(user_input) else "Normal Transaction")