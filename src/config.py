"""
Configuration settings for the Anomaly Detection system.

This module centralizes all configuration parameters for easy maintenance
and consistent behavior across the application.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Data files
TRANSACTION_DATA_FILE = DATA_DIR / "transaction_anomalies_dataset.csv"
RAW_TRANSACTION_FILE = DATA_DIR / "raw_transactions.csv"

# Model files
ISOLATION_FOREST_MODEL = MODEL_DIR / "isolation_forest.pkl"

# Model hyperparameters
MODEL_CONFIG = {
    'contamination': 0.02,
    'random_state': 42,
    'n_estimators': 200,
    'max_samples': 'auto',
    'n_jobs': -1,  # Use all available cores
    'max_features': 1.0  # Use all features
}

# Feature configuration
REQUIRED_FEATURES = [
    'Transaction_Amount',
    'Average_Transaction_Amount',
    'Frequency_of_Transactions'
]

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'threshold_std': 2.0,
    'fill_missing_value': 0,
    'normalize_features': True  # Enable feature normalization
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

# Evaluation metrics
METRIC_NAMES = ['accuracy', 'precision', 'recall', 'f1_score']
TARGET_NAMES = ['Normal', 'Anomaly']

# Validation thresholds
VALIDATION_THRESHOLDS = {
    'min_contamination': 0.001,
    'max_contamination': 0.5,
    'min_data_rows': 10,
    'max_features': 100  # Maximum number of features
}

