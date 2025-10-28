import pandas as pd
import joblib
import logging
from typing import Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, model_path: str = "models/isolation_forest.pkl") -> None:
        """
        Initialize the AnomalyDetector with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        try:
            self.model = joblib.load(model_path)
            self.features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model supports {len(self.features)} features")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> int:
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
            
            # Validate input shape
            if len(input_df) == 0:
                raise ValueError("Input data is empty")
            
            # Check for missing values in features
            if input_df.isnull().any().any():
                logger.warning("Input data contains missing values")
            
            prediction = self.model.predict(input_df)
            result = 1 if prediction[0] == -1 else 0
            prediction_type = 'Anomaly' if result else 'Normal'
            logger.info(f"Prediction completed: {prediction_type}")
            return result
        except KeyError as e:
            logger.error(f"Missing required feature: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        detector = AnomalyDetector()
        user_input = {
            'Transaction_Amount': 1500,
            'Average_Transaction_Amount': 200,
            'Frequency_of_Transactions': 5
        }
        result = detector.predict(user_input)
        print("Anomaly Detected!" if result else "Normal Transaction")
    except Exception as e:
        print(f"Error: {e}")