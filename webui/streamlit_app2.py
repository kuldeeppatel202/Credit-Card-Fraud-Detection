# webui/streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("Interact with the FastAPI backend for real-time and batch predictions.")

API_URL = "http://127.0.0.1:8000/predict"          
BATCH_API_URL = "http://127.0.0.1:8000/predict-batch"

# ------------------------------
# Sidebar: Input options
# ------------------------------
st.sidebar.header("Input Options")
mode = st.sidebar.radio("Select input mode", ["Single Transaction", "Batch CSV Upload"])

# ------------------------------
# Helper: send single/multiple transactions
# ------------------------------
def get_predictions(transactions):
    try:
        response = requests.post(API_URL, json=transactions)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def get_batch_predictions(uploaded_file):
    try:
        response = requests.post(BATCH_API_URL, files={"file": uploaded_file})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        return None

# ------------------------------
# Single Transaction Mode
# ------------------------------
if mode == "Single Transaction":
    st.subheader("📦 Enter Transaction JSON")
    sample_json = [
        {
            "Time": 10000,
            "V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.53634673796914,
            "V4": 1.37815522427443, "V5": -0.338320769942518, "V6": 0.462387777762292,
            "V7": 0.239598554061257, "V8": 0.0986979012610507, "V9": 0.363786969611213,
            "V10": 0.0907941719789316, "V11": -0.551599533260813, "V12": -0.617800855762348,
            "V13": -0.991389847235408, "V14": -0.311169353699879, "V15": 1.46817697209427,
            "V16": -0.470400525259478, "V17": 0.207971241929242, "V18": 0.0257905801985591,
            "V19": 0.403992960255733, "V20": 0.251412098239705, "V21": -0.018306777944153,
            "V22": 0.277837575558899, "V23": -0.110473910188767, "V24": 0.0669280749146731,
            "V25": 0.128539358273528, "V26": -0.189114843888824, "V27": 0.133558376740387,
            "V28": -0.0210530534538215, "Amount": 149.62
        }
    ]
    input_json = st.text_area("Paste or edit JSON transactions here 👇", value=json.dumps(sample_json, indent=4), height=300)

    if st.button("Predict"):
        try:
            transactions = json.loads(input_json)
            result = get_predictions(transactions)

            if result:
                # Display results
                if isinstance(result, list):
                    df_results = pd.DataFrame(result)
                elif "predictions" in result:
                    df_results = pd.DataFrame(result["predictions"])
                else:
                    df_results = pd.DataFrame([result])

                st.success(f"✅ Prediction Complete — Risk: {df_results.iloc[0]['risk_level'] if not df_results.empty else 'N/A'}")
                st.dataframe(df_results)

                # Model contribution chart for each transaction
                for idx, row in df_results.iterrows():
                    st.subheader(f"Transaction {row['index']} — Risk: {row['risk_level']}")
                    model_scores = row["model_scores"]
                    fig = px.bar(
                        x=list(model_scores.keys()),
                        y=list(model_scores.values()),
                        labels={"x": "Model", "y": "Fraud Probability"},
                        title="Model Contributions"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"JSON parsing error: {e}")

# ------------------------------
# Batch CSV Upload Mode
# ------------------------------
if mode == "Batch CSV Upload":
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV file with transactions", type=["csv"])

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded file:")
        st.dataframe(df_input.head())

        result = get_batch_predictions(uploaded_file)

        if result:
            st.success(f"✅ Batch Prediction Complete — Fraud Percentage: {result['fraud_percentage']}%")
            df_results = pd.DataFrame(result["predictions"])
            st.dataframe(df_results)

            # Average model contributions
            st.subheader("Average Model Scores Across Batch")
            avg_scores = pd.DataFrame(df_results["model_scores"].apply(pd.Series).mean()).reset_index()
            avg_scores.columns = ["Model", "Avg Fraud Probability"]
            fig = px.bar(avg_scores, x="Model", y="Avg Fraud Probability", title="Average Model Contributions")
            st.plotly_chart(fig, use_container_width=True)
