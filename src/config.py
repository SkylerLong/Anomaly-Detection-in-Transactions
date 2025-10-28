"""
Configuration settings for the Anomaly Detection system.

This module centralizes all configuration parameters for easy maintenance
and consistent behavior across the application.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Data files
TRANSACTION_DATA_FILE = DATA_DIR / "transaction_anomalies_dataset.csv"
RAW_TRANSACTION_FILE = DATA_DIR / "raw_transactions.csv"

# Model files
ISOLATION_FOREST_MODEL = MODEL_DIR / "isolation_forest.pkl"

# Model hyperparameters - Isolation Forest configuration
MODEL_CONFIG: Dict[str, Any] = {
    'contamination': 0.02,  # Expected proportion of anomalies
    'random_state': 42,  # Reproducibility seed
    'n_estimators': 200,  # Number of trees in the forest
    'max_samples': 'auto',  # Sample size for each tree
    'n_jobs': -1,  # Use all available cores
    'max_features': 1.0  # Use all features
}

# Feature configuration - Core features required for anomaly detection
REQUIRED_FEATURES: List[str] = [
    'Transaction_Amount',
    'Average_Transaction_Amount',
    'Frequency_of_Transactions'
]

# Engineered features created during preprocessing
ENGINEERED_FEATURES: List[str] = [
    'Is_Anomaly',
    'Transaction_ZScore'
]

# Preprocessing configuration
PREPROCESSING_CONFIG: Dict[str, Any] = {
    'threshold_std': 2.0,
    'fill_missing_value': 0,
    'normalize_features': True  # Enable feature normalization
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'enable_file_logging': False  # Set to True to enable file logging
}

# Evaluation metrics
METRIC_NAMES: List[str] = ['accuracy', 'precision', 'recall', 'f1_score']
TARGET_NAMES: List[str] = ['Normal', 'Anomaly']

# Validation thresholds for data and model parameters
VALIDATION_THRESHOLDS: Dict[str, Any] = {
    'min_contamination': 0.001,
    'max_contamination': 0.5,
    'min_data_rows': 10,
    'max_features': 100,  # Maximum number of features
    'min_estimators': 50,  # Minimum number of estimators
    'max_estimators': 500,  # Maximum number of estimators
    'min_std_dev': 0.1  # Minimum standard deviation threshold
}

