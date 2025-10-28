from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_evaluate(X_train, X_test, y_test, contamination=0.02):
    """
    Train Isolation Forest model and evaluate its performance.
    
    Args:
        X_train: Training features
        X_test: Test features  
        y_test: Test labels
        contamination: Expected proportion of anomalies (default: 0.02)
    
    Returns:
        tuple: (trained_model, evaluation_metrics)
    """
    # Validate contamination parameter
    if not 0 < contamination < 0.5:
        raise ValueError("Contamination must be between 0 and 0.5")
    
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        max_samples='auto',
        max_features=1.0  # Use all features
    )
    model.fit(X_train)
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save model
    joblib.dump(model, "models/isolation_forest.pkl")
    logger.info("Model saved successfully")
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred == -1 else 0 for pred in y_pred]
    
    # Calculate comprehensive metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'precision': precision_score(y_test, y_pred_binary),
        'recall': recall_score(y_test, y_pred_binary),
        'f1_score': f1_score(y_test, y_pred_binary)
    }
    
    print(classification_report(y_test, y_pred_binary, target_names=['Normal', 'Anomaly']))
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    return model, metrics