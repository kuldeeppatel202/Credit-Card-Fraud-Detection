import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

class FraudDetectionModels:
    def __init__(self):
        self.models = {}
        self.threshold = 0.35
        # Ensemble weights to balance supervised and unsupervised models
        self.weights = {
            "xgboost": 0.25,
            "lightgbm": 0.25,
            "random_forest": 0.15,
            "isolation_forest": 0.20,
            "autoencoder": 0.15
        }

    # ---------- SUPERVISED MODELS ----------
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric='auc', random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.models['xgboost'] = model
        return model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            is_unbalance=True, metric='auc', random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
        self.models['lightgbm'] = model
        return model

    def train_random_forest(self, X_train, y_train):
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model

    # ---------- UNSUPERVISED MODELS ----------
    def train_isolation_forest(self, X_train):
        model = IsolationForest(
            n_estimators=200, contamination=0.02, random_state=42
        )
        model.fit(X_train)
        self.models['isolation_forest'] = model
        return model

    def build_autoencoder(self, input_dim):
        encoder = keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(input_dim,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu')
        ])
        decoder = keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=(8,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        autoencoder = keras.Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train_autoencoder(self, X_train_normal, X_val, epochs=50):
        input_dim = X_train_normal.shape[1]
        autoencoder = self.build_autoencoder(input_dim)
        autoencoder.fit(
            X_train_normal, X_train_normal,
            epochs=epochs, batch_size=256, validation_data=(X_val, X_val),
            verbose=0
        )
        self.models['autoencoder'] = autoencoder
        return autoencoder

    # ---------- EVALUATION ----------
    def evaluate_model(self, model, X_test, y_test, name):
        print(f"\n{'='*40}\nEvaluating {name}\n{'='*40}")
        if name == 'isolation_forest':
            preds = np.where(model.predict(X_test) == -1, 1, 0)
        elif name == 'autoencoder':
            recon = model.predict(X_test, verbose=0)
            mse = np.mean(np.square(X_test - recon), axis=1)
            # Normalize MSE for better thresholding
            norm_mse = (mse - mse.min()) / (mse.max() - mse.min() + 1e-9)
            preds = (norm_mse > 0.5).astype(int)
        else:
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs > self.threshold).astype(int)

        print(classification_report(y_test, preds))
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print(f"F1 Score: {f1_score(y_test, preds):.4f}")
        return preds

    # ---------- ENSEMBLE ----------
    def ensemble_predict(self, X_test):
        n_samples = X_test.shape[0]
        ensemble_probs = np.zeros(n_samples)

        for name, model in self.models.items():
            weight = self.weights.get(name, 0.2)
            if name == 'isolation_forest':
                p = np.where(model.predict(X_test) == -1, 1, 0)
                ensemble_probs += p * weight
            elif name == 'autoencoder':
                recon = model.predict(X_test, verbose=0)
                mse = np.mean(np.square(X_test - recon), axis=1)
                norm_mse = (mse - mse.min()) / (mse.max() - mse.min() + 1e-9)
                ensemble_probs += norm_mse * weight
            else:
                probs = model.predict_proba(X_test)[:, 1]
                ensemble_probs += probs * weight

        # Amount-based heuristic: large transactions are more risky
        if 'Amount' in X_test.dtype.names:  # if structured array
            amounts = X_test['Amount']
        else:
            amounts = X_test[:, -1]  # assuming last column is Amount
        ensemble_probs = np.maximum(ensemble_probs, np.where(amounts > 10000, 0.7, ensemble_probs))

        # Convert to binary prediction
        preds = (ensemble_probs > self.threshold).astype(int)
        return preds, ensemble_probs

    # ---------- SAVE / LOAD ----------
    def save_models(self, model_dir='../models'):
        os.makedirs(model_dir, exist_ok=True)
        for name, model in self.models.items():
            if name == 'autoencoder':
                model.save(f'{model_dir}/{name}_model.h5')
            else:
                joblib.dump(model, f'{model_dir}/{name}_model.pkl')
        print(f"✅ Models saved to {model_dir}/")
