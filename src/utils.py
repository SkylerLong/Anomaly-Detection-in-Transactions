import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib
from sklearn.metrics import classification_report

def load_data(file_path):
    """Load transaction data from CSV"""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully with {data.shape[0]} records")
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

def save_model(model, path):
    """Save trained model to disk"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    """Load trained model from disk"""
    try:
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model
    except Exception as e:
        raise FileNotFoundError(f"Model loading failed: {str(e)}")

def evaluate_model(y_true, y_pred):
    """Generate classification metrics"""
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'precision': report['Anomaly']['precision'],
        'recall': report['Anomaly']['recall'],
        'f1': report['Anomaly']['f1-score']
    }

def save_metrics(metrics, path):
    """Save evaluation metrics to JSON"""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}")