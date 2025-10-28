import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for transaction data with anomaly detection features.
    
    This transformer applies statistical methods to identify potential anomalies
    based on transaction amounts using z-score methodology.
    """
    
    def __init__(self, threshold_std: float = 2.0, handle_nulls: bool = True) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            threshold_std: Number of standard deviations for anomaly threshold (default: 2.0)
            handle_nulls: Whether to fill missing values (default: True)
        """
        self.threshold_std = threshold_std
        self.handle_nulls = handle_nulls
        self.anomaly_threshold = None
        self.mean_amount = None
        self.std_amount = None

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'DataPreprocessor':
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
        
        # Calculate statistics, handling potential NaN values
        self.mean_amount = X['Transaction_Amount'].mean()
        self.std_amount = X['Transaction_Amount'].std()
        
        if pd.isna(self.mean_amount) or pd.isna(self.std_amount):
            raise ValueError("Cannot compute statistics: data contains invalid values")
        
        self.anomaly_threshold = self.mean_amount + self.threshold_std * self.std_amount
        
        logger.info(f"Preprocessor fitted successfully")
        logger.info(f"Statistics - Mean: {self.mean_amount:.2f}, Std: {self.std_amount:.2f}")
        logger.info(f"Anomaly threshold: {self.anomaly_threshold:.2f}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        
        # Add anomaly indicator based on threshold
        X['Is_Anomaly'] = (X['Transaction_Amount'] > self.anomaly_threshold).astype(int)
        anomaly_count = X['Is_Anomaly'].sum()
        anomaly_rate = anomaly_count / len(X) if len(X) > 0 else 0
        
        # Add z-score feature with protection against division by zero
        if self.std_amount > 0:
            X['Transaction_ZScore'] = (X['Transaction_Amount'] - self.mean_amount) / self.std_amount
        else:
            X['Transaction_ZScore'] = 0
            logger.warning("Standard deviation is zero, z-score set to 0")
        
        # Handle missing values
        if self.handle_nulls:
            X = X.fillna(0)
        
        logger.info(f"Data transformation completed")
        logger.info(f"Anomalies detected: {anomaly_count} ({anomaly_rate:.2%})")
        logger.debug(f"Data shape after transformation: {X.shape}")
        return X

# Example usage
if __name__ == "__main__":
    try:
        data = pd.read_csv("data/raw_transactions.csv")
        preprocessor = DataPreprocessor(threshold_std=2.0)
        preprocessed_data = preprocessor.fit_transform(data)
        logger.info("Preprocessing completed successfully")
        logger.info(f"Processed {len(preprocessed_data)} records")
    except FileNotFoundError:
        logger.warning("Sample data file not found - skipping example")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")