import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

class FraudDataProcessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.smote = SMOTE(sampling_strategy=0.5, random_state=42)
        
    def load_and_split(self, filepath, test_size=0.3):
        """Load data and split"""
        df = pd.read_csv(filepath)
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def engineer_features(self, X):
        """Feature engineering"""
        X_new = X.copy()
        if 'Time' in X.columns:
            X_new['Hour'] = (X['Time'] / 3600) % 24
            X_new['Day'] = (X['Time'] / 86400).astype(int)
        if 'Amount' in X.columns:
            X_new['Amount_log'] = np.log1p(X['Amount'])
            X_new['Amount_scaled'] = self.scaler.fit_transform(X[['Amount']])
        return X_new
    
    def apply_smote(self, X_train, y_train):
        """Handle imbalance"""
        X_res, y_res = self.smote.fit_resample(X_train, y_train)
        print(f"✅ After SMOTE - Fraud cases: {y_res.sum()}")
        return X_res, y_res
    
    def save_processor(self, filepath):
        joblib.dump(self, filepath)
        print(f"🧩 Processor saved to {filepath}")
        
    @staticmethod
    def load_processor(filepath):
        return joblib.load(filepath)
