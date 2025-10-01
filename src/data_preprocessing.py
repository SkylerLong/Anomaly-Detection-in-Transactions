import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.anomaly_threshold = None

    def fit(self, X, y=None):
        self.anomaly_threshold = X['Transaction_Amount'].mean() + 2 * X['Transaction_Amount'].std()
        return self

    def transform(self, X):
        X = X.copy()
        X['Is_Anomaly'] = (X['Transaction_Amount'] > self.anomaly_threshold).astype(int)
        return X

# Usage
data = pd.read_csv("data/raw_transactions.csv")
preprocessor = DataPreprocessor()
preprocessed_data = preprocessor.fit_transform(data)