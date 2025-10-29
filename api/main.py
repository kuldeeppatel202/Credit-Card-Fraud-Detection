# api/main.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging
from fastapi import Body

# Optional: tensorflow may print lots of logs; suppress unless needed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Ensemble of supervised + unsupervised models for fraud detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Pydantic model for single transaction
# ----------------------------
class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float; V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

# ----------------------------
# Lazy loader for models and preprocessor
# ----------------------------
class ModelLoader:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.preprocessor = None
        self._loaded = False

    def load_all(self):
        if self._loaded:
            return
        # Load supervised models (pkl)
        mapping = {
            "xgboost": "xgboost_model.pkl",
            "lightgbm": "lightgbm_model.pkl",
            "random_forest": "random_forest_model.pkl",
            "isolation_forest": "isolation_forest_model.pkl",  # unsupervised
            # autoencoder will be saved as .h5
        }

        for name, fname in mapping.items():
            path = os.path.join(self.model_dir, fname)
            if os.path.exists(path):
                try:
                    self.models[name] = joblib.load(path)
                    logger.info(f"Loaded model: {name}")
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")

        # Load autoencoder if present
        ae_path = os.path.join(self.model_dir, "autoencoder_model.h5")
        try:
            if os.path.exists(ae_path):
                # Import lazily to avoid TF import errors when not present
                from tensorflow import keras
                self.models["autoencoder"] = keras.models.load_model(ae_path)
                logger.info("Loaded autoencoder model")
        except Exception as e:
            logger.error(f"Failed to load autoencoder: {e}")

        # Load preprocessor (joblib)
        preproc_path = os.path.join(self.model_dir, "preprocessor.pkl")
        if os.path.exists(preproc_path):
            try:
                self.preprocessor = joblib.load(preproc_path)
                logger.info("Loaded preprocessor")
            except Exception as e:
                logger.error(f"Failed to load preprocessor: {e}")
        else:
            logger.warning("Preprocessor file not found.")

        self._loaded = True

loader = ModelLoader()

# ----------------------------
# Helpers
# ----------------------------
def preprocess_row(df: pd.DataFrame):
    """
    Attempt to use the loaded preprocessor (if it exposes
    engineer_features or transform). Otherwise, apply minimal feature engineering.
    """
    if loader.preprocessor is not None:
        # try common method names used in our repo
        if hasattr(loader.preprocessor, "engineer_features"):
            df_proc = loader.preprocessor.engineer_features(df.copy())
            return df_proc
        if hasattr(loader.preprocessor, "transform"):
            df_proc = loader.preprocessor.transform(df.copy())
            return df_proc
    # fallback: basic feature engineering
    df2 = df.copy()
    if "Time" in df2.columns:
        df2["Hour"] = (df2["Time"] / 3600) % 24
        df2["Day"] = (df2["Time"] / 86400).astype(int)
    if "Amount" in df2.columns:
        df2["Amount_log"] = np.log1p(df2["Amount"])
        # try scale with RobustScaler if available
        try:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            df2["Amount_scaled"] = scaler.fit_transform(df2[["Amount"]])
        except Exception:
            df2["Amount_scaled"] = df2["Amount"]
    return df2

def score_autoencoder(autoencoder, X):
    recon = autoencoder.predict(X, verbose=0)
    mse = np.mean(np.square(X - recon), axis=1)
    # normalize to [0,1]
    norm = (mse - mse.min()) / (mse.max() - mse.min() + 1e-9)
    return norm

def get_risk_level(prob):
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

def get_recommendation(risk):
    return {
        "LOW": "Approve — no immediate action needed.",
        "MEDIUM": "Flag for manual review / ask for OTP.",
        "HIGH": "Block transaction and start investigation."
    }.get(risk, "No recommendation")

# Default model weights for ensemble (adjustable)
DEFAULT_WEIGHTS = {
    "xgboost": 0.30,
    "lightgbm": 0.30,
    "random_forest": 0.20,
    "isolation_forest": 0.10,
    "autoencoder": 0.10
}

# ----------------------------
# Endpoints
# ----------------------------
@app.on_event("startup")
def startup_event():
    loader.load_all()

@app.get("/")
def root():
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health():
    loader.load_all()
    return {
        "status": "healthy",
        "models_loaded": list(loader.models.keys()),
        "preprocessor_loaded": loader.preprocessor is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/models-info")
def models_info():
    loader.load_all()
    return {
        "models": list(loader.models.keys()),
        "ensemble_weights_default": DEFAULT_WEIGHTS
    }

@app.post("/predict")
def predict_transaction(transactions: Any = Body(..., example=[{"Time":0.0,"V1":0.0,"V2":0.0,"Amount":0.0}])):
    """
    Predict for one or more transactions.
    Accepts either a single JSON object or a list of JSON objects in the **body**.
    """
    loader.load_all()

    try:
        # Normalize input format: list of dicts
        if isinstance(transactions, dict):
            transactions = [transactions]
        elif not isinstance(transactions, list):
            raise HTTPException(status_code=400, detail="Input must be a dictionary or list of dictionaries.")

        # Rest of your prediction logic...
        df = pd.DataFrame(transactions)
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty input data.")
        
        df_proc = preprocess_row(df)
        X = df_proc.select_dtypes(include=[np.number]).values

        results: List[Dict[str, Any]] = []

        for i in range(len(X)):
            x_i = X[i:i+1]
            model_scores: Dict[str, float] = {}

            for name, model in loader.models.items():
                if name in ["xgboost", "lightgbm", "random_forest"]:
                    try:
                        proba = model.predict_proba(x_i)[:, 1][0]
                    except Exception:
                        proba = float(model.predict(x_i)[0])
                    model_scores[name] = float(proba)
                elif name == "isolation_forest":
                    pred = model.predict(x_i)[0]
                    score = 1.0 if pred == -1 else 0.0
                    model_scores[name] = float(score)
                elif name == "autoencoder":
                    try:
                        score = float(score_autoencoder(model, x_i)[0])
                    except Exception:
                        score = 0.0
                    model_scores[name] = float(score)

            available = [m for m in model_scores.keys()]
            total = sum(DEFAULT_WEIGHTS.get(m, 0) for m in available)
            weights = {m: (DEFAULT_WEIGHTS.get(m, 0) / total) for m in available} if total > 0 else {m: 1.0 / len(available) for m in available}

            fraud_prob = sum(model_scores[m] * weights[m] for m in available)
            risk = get_risk_level(fraud_prob)
            rec = get_recommendation(risk)

            results.append({
                "index": i,
                "is_fraud": fraud_prob > 0.5,
                "fraud_probability": round(float(fraud_prob), 4),
                "risk_level": risk,
                "recommendation": rec,
                "model_scores": model_scores
            })

        if len(results) == 1:
            return results[0]
        else:
            fraud_detected = sum(1 for r in results if r["is_fraud"])
            return {
                "total": len(results),
                "fraud_detected": fraud_detected,
                "fraud_percentage": round(fraud_detected / max(1, len(results)) * 100, 2),
                "predictions": results
            }

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts a CSV file of transactions, returns predictions for each row.
    """
    loader.load_all()
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        df_proc = preprocess_row(df)
        X = df_proc.select_dtypes(include=[np.number]).values

        results: List[Dict[str, Any]] = []
        model_scores_all = {m: [] for m in loader.models.keys()}

        # get model-wise scores for all rows
        for name, model in loader.models.items():
            if name in ["xgboost", "lightgbm", "random_forest"]:
                try:
                    probs = model.predict_proba(X)[:, 1].tolist()
                except Exception:
                    probs = model.predict(X).tolist()
                model_scores_all[name] = [float(p) for p in probs]
            elif name == "isolation_forest":
                preds = model.predict(X)  # -1 or 1
                model_scores_all[name] = [1.0 if p == -1 else 0.0 for p in preds]
            elif name == "autoencoder":
                try:
                    ae_scores = score_autoencoder(model, X).tolist()
                except Exception:
                    ae_scores = [0.0] * len(X)
                model_scores_all[name] = [float(s) for s in ae_scores]

        # compute ensemble probabilities per row
        available = list(model_scores_all.keys())
        total = sum(DEFAULT_WEIGHTS.get(m, 0) for m in available)
        if total <= 0:
            weights = {m: 1.0 / len(available) for m in available}
        else:
            weights = {m: (DEFAULT_WEIGHTS.get(m, 0) / total) for m in available}

        for i in range(len(X)):
            row_scores = {m: model_scores_all[m][i] for m in available}
            fraud_prob = sum(row_scores[m] * weights[m] for m in available)
            risk = get_risk_level(fraud_prob)
            results.append({
                "index": i,
                "fraud_probability": round(float(fraud_prob), 4),
                "is_fraud": fraud_prob > 0.5,
                "risk_level": risk,
                "model_scores": {m: row_scores[m] for m in available}
            })

        fraud_detected = sum(1 for r in results if r["is_fraud"])
        return {
            "total": len(results),
            "fraud_detected": fraud_detected,
            "fraud_percentage": round(fraud_detected / max(1, len(results)) * 100, 2),
            "predictions": results
        }
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(e))
