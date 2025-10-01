from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score
import joblib

def train_and_evaluate(X_train, X_test, y_test, contamination=0.02):
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        max_samples='auto'
    )
    model.fit(X_train)
    
    # Save model
    joblib.dump(model, "models/isolation_forest.pkl")
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred == -1 else 0 for pred in y_pred]
    
    print(classification_report(y_test, y_pred_binary, target_names=['Normal', 'Anomaly']))
    print(f"F1 Score: {f1_score(y_test, y_pred_binary):.4f}")
    
    return model