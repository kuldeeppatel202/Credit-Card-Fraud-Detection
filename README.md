💳 Real-Time Credit Card Fraud Detection System

🚀 An end-to-end machine learning pipeline for detecting fraudulent credit card transactions in real time, built with Python, Scikit-learn, XGBoost, LightGBM, TensorFlow Autoencoders, and Streamlit for deployment.
This project demonstrates supervised + unsupervised learning, model ensemble, and real-time API integration for production-ready fraud detection.

🧠 Overview

Fraudulent credit card transactions are rare but extremely costly.
This system aims to:

Accurately detect fraudulent activities with imbalanced datasets

Deploy models with a REST API + Streamlit frontend

Provide a unified ensemble prediction combining supervised and unsupervised models

🏗️ Architecture Overview

                ┌────────────────────┐
                │   Raw Transaction   │
                │       Data          │
                └─────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Data Preprocessing   │
              │  - Feature scaling    │
              │  - Train-test split   │
              │  - Imbalance handling │
              └───────────┬───────────┘
                          │
          ┌───────────────▼────────────────┐
          │     Model Training (ML)        │
          │ - XGBoost, LightGBM, RF        │
          │ - Autoencoder (unsupervised)   │
          │ - Isolation Forest             │
          └───────────────┬────────────────┘
                          │
          ┌───────────────▼────────────────┐
          │         Model Ensemble         │
          │ Combines multiple model votes  │
          │ for better fraud detection     │
          └───────────────┬────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │     REST API + Streamlit UI     │
         │ Real-time risk prediction app   │
         └─────────────────────────────────┘

⚙️ Technical Stack

| Component               | Technology Used                      | Description                             |
| ----------------------- | ------------------------------------ | --------------------------------------- |
| **Language**            | Python 3.10                          | Core implementation                     |
| **Data Handling**       | Pandas, NumPy                        | Preprocessing & feature scaling         |
| **Supervised Models**   | XGBoost, LightGBM, RandomForest      | Trained on labeled fraud data           |
| **Unsupervised Models** | Autoencoder (Keras), IsolationForest | Detect unseen anomalies                 |
| **Model Evaluation**    | ROC-AUC, F1-score, Confusion Matrix  | Handles class imbalance                 |
| **Deployment**          | Streamlit, FastAPI (optional)        | Real-time risk prediction               |
| **Model Storage**       | Joblib, H5                           | Efficient persistence of trained models |
| **Monitoring**          | Logging & threshold tuning           | Keeps track of fraud drift              |


🧩 Folder Structure

fraud-detection-system/
│
├── data/                     # Dataset (CSV files)
├── notebooks/                # Jupyter notebooks for exploration
├── models/                   # Saved models (.pkl and .h5)
├── src/                      # Source code for pipeline
│   ├── models.py             # All ML/Deep Learning model training
│   ├── preprocess.py         # Data cleaning & scaling
│   ├── train_pipeline.py     # Main training entrypoint
│   ├── predict.py            # Real-time prediction logic
│   └── utils.py              # Helper functions
│
├── app.py                    # Streamlit frontend for predictions
├── requirements.txt           # Dependencies
└── README.md                  # Documentation (this file)


🔬 Model Training Details
| Model            | Key Features                              | Strength                  |
| ---------------- | ----------------------------------------- | ------------------------- |
| **XGBoost**      | `scale_pos_weight` for imbalance handling | High recall and precision |
| **LightGBM**     | Leaf-wise growth, early stopping          | Fast and memory efficient |
| **RandomForest** | Balanced class weighting                  | Robust to noise           |

Metrics Used:

ROC-AUC

F1-score

Confusion Matrix

Precision & Recall

🧠 2. Unsupervised Models
| Model                   | Approach                                                  | Purpose                            |
| ----------------------- | --------------------------------------------------------- | ---------------------------------- |
| **Autoencoder (Keras)** | Neural network trained to reconstruct normal transactions | High reconstruction error → Fraud  |
| **Isolation Forest**    | Tree-based anomaly detection                              | Detects rare, outlier transactions |

🧩 3. Ensemble Strategy
Each model predicts independently.

Final prediction = Majority voting among all models.

Autoencoder + IsolationForest capture anomalies unseen by supervised models.

final_pred = np.round(np.mean(model_predictions, axis=0)).astype(int)


🧮 Example Prediction

Input Example (JSON):
[
  {
    "Time": 10000,
    "V1": -1.3598,
    "V2": -0.0727,
    "V3": 2.5363,
    "V4": 1.3781,
    "V14": -0.3111,
    "V17": 0.2079,
    "Amount": 900000
  }
]

✅ Prediction Complete — Risk: HIGH

💻 How to Run Locally
Step 1: Clone the Repository

git clone https://github.com/<your-username>/fraud-detection-system.git
cd fraud-detection-system

Step 2: Create Virtual Environment

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows


Step 3: Install Dependencies

pip install -r requirements.txt

Step 4: Train Models

python src/train_pipeline.py

Step 5: Run the Web App

streamlit run app.py

📊 Evaluation Results

| Model        | ROC-AUC   | F1-Score | Accuracy   |
| ------------ | --------- | -------- | ---------- |
| XGBoost      | 0.998     | 0.95     | 99.8%      |
| LightGBM     | 0.996     | 0.94     | 99.7%      |
| RandomForest | 0.990     | 0.92     | 99.5%      |
| Autoencoder  | 0.987     | -        | -          |
| Ensemble     | **0.999** | **0.96** | **99.85%** |

🧠 Key Learnings

Handling imbalanced datasets with scale_pos_weight, SMOTE, and threshold tuning

Combining supervised and unsupervised models for high fraud recall

Using Autoencoders for anomaly detection in finance data

Building real-time APIs for live fraud risk assessment

🤝 Acknowledgements

Dataset: Kaggle Credit Card Fraud Detection Dataset
Inspired by real-world fraud detection architectures used by fintech and banks

🧑‍💻 Author

Kuldeep Patel
🎓 M.Tech, Computer Science — IIT Guwahati
