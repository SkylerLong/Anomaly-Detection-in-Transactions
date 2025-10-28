import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for transaction data with anomaly detection features.
    
    This transformer applies statistical methods to identify potential anomalies
    based on transaction amounts using z-score methodology.
    """
    
    def __init__(self, threshold_std=2.0):
        """
        Initialize the preprocessor.
        
        Args:
            threshold_std: Number of standard deviations for anomaly threshold (default: 2.0)
        """
        self.threshold_std = threshold_std
        self.anomaly_threshold = None
        self.mean_amount = None
        self.std_amount = None

    def fit(self, X, y=None):
        """
        Calculate statistics from training data.
        
        Args:
            X: DataFrame with transaction data
            y: Optional target variable (not used)
        
        Returns:
            self: Fitted preprocessor instance
        """
        if 'Transaction_Amount' not in X.columns:
            raise ValueError("Missing required column: Transaction_Amount")
        
        self.mean_amount = X['Transaction_Amount'].mean()
        self.std_amount = X['Transaction_Amount'].std()
        self.anomaly_threshold = self.mean_amount + self.threshold_std * self.std_amount
        
        logger.info(f"Preprocessor fitted - Threshold: {self.anomaly_threshold:.2f}")
        return self

    def transform(self, X):
        """
        Transform data by adding anomaly indicators and engineered features.
        
        Args:
            X: DataFrame with transaction data
        
        Returns:
            DataFrame with additional features
        """
        if self.anomaly_threshold is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = X.copy()
        
        # Add anomaly indicator
        X['Is_Anomaly'] = (X['Transaction_Amount'] > self.anomaly_threshold).astype(int)
        
        # Add z-score feature with protection against division by zero
        if self.std_amount > 0:
            X['Transaction_ZScore'] = (X['Transaction_Amount'] - self.mean_amount) / self.std_amount
        else:
            X['Transaction_ZScore'] = 0
            logger.warning("Standard deviation is zero, z-score set to 0")
        
        # Handle missing values
        X = X.fillna(0)
        
        logger.info(f"Data transformed - {X['Is_Anomaly'].sum()} anomalies detected")
        return X

# Example usage
if __name__ == "__main__":
    try:
        data = pd.read_csv("data/raw_transactions.csv")
        preprocessor = DataPreprocessor()
        preprocessed_data = preprocessor.fit_transform(data)
        logger.info("Preprocessing completed successfully")
    except FileNotFoundError:
        logger.warning("Sample data file not found - skipping example")