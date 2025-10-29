import streamlit as st
import requests
import pandas as pd
import json
from io import StringIO

# ---- PAGE CONFIG ----
st.set_page_config(page_title="💳 Fraud Detection Dashboard", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("Interact with your FastAPI backend for **real-time** and **batch** predictions.")

# ---- SIDEBAR ----
st.sidebar.header("🧠 Choose Mode")
mode = st.sidebar.radio("Prediction Mode", ["Single / Multiple JSON", "Batch CSV Upload"])

# ======================================
# 1️⃣ JSON MODE
# ======================================
if mode == "Single / Multiple JSON":
    st.subheader("📦 Enter Transaction JSON")

    example = [
        {
            "Time": 10000,
            "V1": -1.3598071336738,
            "V2": -0.0727811733098497,
            "V3": 2.53634673796914,
            "V4": 1.37815522427443,
            "V5": -0.338320769942518,
            "V6": 0.462387777762292,
            "V7": 0.239598554061257,
            "V8": 0.0986979012610507,
            "V9": 0.363786969611213,
            "V10": 0.0907941719789316,
            "V11": -0.551599533260813,
            "V12": -0.617800855762348,
            "V13": -0.991389847235408,
            "V14": -0.311169353699879,
            "V15": 1.46817697209427,
            "V16": -0.470400525259478,
            "V17": 0.207971241929242,
            "V18": 0.0257905801985591,
            "V19": 0.403992960255733,
            "V20": 0.251412098239705,
            "V21": -0.018306777944153,
            "V22": 0.277837575558899,
            "V23": -0.110473910188767,
            "V24": 0.0669280749146731,
            "V25": 0.128539358273528,
            "V26": -0.189114843888824,
            "V27": 0.133558376740387,
            "V28": -0.0210530534538215,
            "Amount": 149.62
        }
    ]

    json_input = st.text_area(
        "Paste or edit JSON transactions here 👇",
        value=json.dumps(example, indent=4),
        height=400,
    )

    if st.button("🚀 Predict Fraud (JSON)"):
        try:
            transactions = json.loads(json_input)
            resp = requests.post(f"{API_URL}/predict", json=transactions)

            if resp.status_code == 200:
                result = resp.json()

                # If single transaction
                if isinstance(result, dict) and "fraud_probability" in result:
                    st.success(f"✅ Prediction Complete — Risk: {result['risk_level']}")
                    st.json(result)
                else:
                    df = pd.DataFrame(result["predictions"])
                    st.success("✅ Batch JSON Prediction Complete")
                    st.dataframe(df, use_container_width=True)
            else:
                st.error(f"❌ API Error: {resp.text}")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ======================================
# 2️⃣ CSV UPLOAD MODE
# ======================================
else:
    st.subheader("📁 Upload a CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.write("Preview of uploaded file:")
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("🚀 Predict Fraud (CSV)"):
            try:
                files = {"file": uploaded_file.getvalue()}
                resp = requests.post(f"{API_URL}/predict-batch", files=files)

                if resp.status_code == 200:
                    result = resp.json()
                    df_pred = pd.DataFrame(result["predictions"])
                    st.success(
                        f"✅ Processed {result['total']} records — Fraud detected in {result['fraud_detected']} transactions ({result['fraud_percentage']}%)."
                    )
                    st.dataframe(df_pred, use_container_width=True)
                else:
                    st.error(f"❌ API Error: {resp.text}")

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# ======================================
# Footer
# ======================================
st.markdown("---")
st.caption("Built with ❤️ using **FastAPI + Streamlit**")
