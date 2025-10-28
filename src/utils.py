import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib
from sklearn.metrics import classification_report
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load transaction data from CSV with validation.
    
    This function loads transaction data from a CSV file and performs basic
    validation checks including existence check and empty data check.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded transaction data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or malformed
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("Data file is empty")
        
        num_records, num_features = data.shape
        logger.info(f"Data loaded successfully - Records: {num_records}, Features: {num_features}")
        logger.info(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024:.2f} KB")
        logger.debug(f"Column names: {list(data.columns)}")
        return data
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or malformed")
        raise ValueError("Data file is empty or malformed")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise ValueError(f"Error loading data: {str(e)}")

def validate_data(data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """
    Validate data structure and required columns.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        bool: True if validation passes
    """
    if required_columns is None:
        required_columns = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values in required columns
    null_counts = data[required_columns].isnull().sum()
    if null_counts.any():
        affected_columns = null_counts[null_counts > 0].to_dict()
        logger.warning(f"Data contains missing values in: {affected_columns}")
    
    # Log validation results
    total_missing = null_counts.sum()
    if total_missing == 0:
        logger.info("Data validation passed - no missing values")
    else:
        logger.info(f"Data validation passed with {total_missing} missing values across all columns")
    return True

def save_model(model: Any, path: str) -> None:
    """
    Save trained model to disk with validation.
    
    Args:
        model: Trained model to save
        path: Destination file path
    """
    try:
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {path}")
        joblib.dump(model, path)
        saved_file_size = Path(path).stat().st_size / 1024
        logger.info(f"Model saved successfully to {path} ({saved_file_size:.2f} KB)")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(path: str) -> Any:
    """
    Load trained model from disk with validation.
    
    Args:
        path: Path to the model file
    
    Returns:
        Model: Loaded scikit-learn model
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        logger.info(f"Loading model from {path}")
        model = joblib.load(path)
        file_size = Path(path).stat().st_size / 1024
        logger.info(f"Model loaded successfully ({file_size:.2f} KB)")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise FileNotFoundError(f"Model loading failed: {str(e)}")

def evaluate_model(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Generate classification metrics with validation.
    
    This function calculates comprehensive evaluation metrics for anomaly detection
    including accuracy, precision, recall, and F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Dictionary containing evaluation metrics
    
    Raises:
        ValueError: If input lengths don't match
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch between true labels ({len(y_true)}) and predictions ({len(y_pred)})")
    
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], output_dict=True)
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['Anomaly']['precision'],
        'recall': report['Anomaly']['recall'],
        'f1': report['Anomaly']['f1-score']
    }
    logger.info(f"Model evaluation completed - F1 Score: {metrics['f1']:.4f}")
    return metrics

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """
    Save evaluation metrics to JSON with validation.
    
    Args:
        metrics: Dictionary of metrics to save
        path: Destination file path
    """
    try:
        output_dir = Path(path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving metrics to {path}")
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved successfully")
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")
        raise