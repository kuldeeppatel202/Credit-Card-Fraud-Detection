# debug/debug_predict.py
import json, os, joblib, numpy as np, pandas as pd
from pathlib import Path

# Put your JSON array here (or load from file)
SYN_JSON = [
  {"Time":50000,"V1":2.8,"V2":2.4,"V3":2.6,"V4":2.1,"V5":-2.0,"V6":1.9,"V7":2.3,"V8":1.8,"V9":2.2,"V10":1.7,"V11":-2.4,"V12":2.5,"V13":-1.9,"V14":2.0,"V15":2.6,"V16":-2.1,"V17":2.3,"V18":-1.8,"V19":2.1,"V20":1.9,"V21":-2.0,"V22":2.2,"V23":-2.3,"V24":1.6,"V25":2.0,"V26":-2.1,"V27":2.5,"V28":-2.2,"Amount":4200.00}
]

MODEL_DIR = "models"

def load_models():
    models = {}
    for fn in os.listdir(MODEL_DIR):
        p = Path(MODEL_DIR)/fn
        if fn.endswith(".pkl"):
            name = fn.replace("_model.pkl","").replace(".pkl","")
            try:
                models[name] = joblib.load(p)
            except Exception as e:
                print("Failed load", p, e)
        elif fn.endswith(".h5"):
            try:
                from tensorflow import keras
                models["autoencoder"] = keras.models.load_model(str(p))
            except Exception as e:
                print("Failed load autoencoder", e)
    return models

def load_preproc():
    p = Path(MODEL_DIR)/"preprocessor.pkl"
    if p.exists():
        return joblib.load(p)
    return None

def main():
    models = load_models()
    pre = load_preproc()
    print("Loaded models:", list(models.keys()))
    print("Preprocessor loaded:", pre is not None)

    df = pd.DataFrame(SYN_JSON)
    print("\nRaw input ranges:")
    print(df.describe().loc[["min","max","mean","std"]])

    # Use preprocessor if available
    if pre is not None:
        # try engineer_features or transform
        if hasattr(pre, "engineer_features"):
            df_proc = pre.engineer_features(df.copy())
            print("\nUsed preprocessor.engineer_features()")
        elif hasattr(pre, "transform"):
            df_proc = pre.transform(df.copy())
            print("\nUsed preprocessor.transform()")
        else:
            df_proc = df.copy()
            print("\nPreprocessor loaded but no known method, using raw df")
    else:
        df_proc = df.copy()
        print("\nNo preprocessor, using fallback engineer (log amount + hour)")

        # fallback features
        if "Time" in df_proc.columns:
            df_proc["Hour"] = (df_proc["Time"]/3600)%24
        if "Amount" in df_proc.columns:
            df_proc["Amount_log"] = np.log1p(df_proc["Amount"])

    print("\nPost-preprocess ranges / sample:")
    print(df_proc.head().T)

    # Numeric matrix for models
    X = df_proc.select_dtypes(include=[np.number]).values
    print("\nNumeric feature count:", X.shape[1])

    # Per-model outputs
    from sklearn.ensemble import IsolationForest
    for name, m in models.items():
        print("\n--- Model:", name)
        try:
            if name in ["xgboost","lightgbm","random_forest"]:
                try:
                    proba = m.predict_proba(X)[:,1]
                    print("predict_proba:", proba)
                except Exception:
                    pred = m.predict(X)
                    print("predict:", pred)
            elif name == "isolation_forest":
                pred = m.predict(X)  # -1 anomaly, 1 normal
                print("isolation preds:", pred)
            elif name == "autoencoder":
                recon = m.predict(X, verbose=0)
                mse = np.mean((X - recon)**2, axis=1)
                print("autoencoder mse:", mse)
            else:
                # generic
                try:
                    print("model predict_proba:", m.predict_proba(X)[:,1])
                except Exception:
                    print("model predict:", m.predict(X))
        except Exception as e:
            print("Error scoring:", e)

if __name__ == "__main__":
    main()
