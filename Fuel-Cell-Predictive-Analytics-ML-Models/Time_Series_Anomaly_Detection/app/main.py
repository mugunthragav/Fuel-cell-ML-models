from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pywt
import io
import base64
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fuel Cell Time-Series Anomaly Detection API")

# Load models
try:
    scaler = joblib.load("models/scaler.pkl")
    logger.info(f"Scaler loaded successfully, n_features_in_: {scaler.n_features_in_}")
    autoencoder = tf.saved_model.load("models/transformer_autoencoder")
    logger.info("Autoencoder loaded successfully")
    iso_forest = joblib.load("models/isolation_forest.pkl")
    logger.info(f"Isolation Forest loaded successfully, n_features_in_: {getattr(iso_forest, 'n_features_in_', 'Unknown')}")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise Exception(f"Failed to load models: {str(e)}")

def preprocess_data(df):
    try:
        required_cols = ["time", "current", "voltage"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {set(required_cols) - set(df.columns)}")
        if "power" in df.columns:
            df.drop(columns=["power"], inplace=True)
        if pd.api.types.is_numeric_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", origin="1970-01-01")
        else:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            if df["time"].isna().any():
                raise ValueError("Invalid 'time' values detected")
        df.set_index("time", inplace=True)
        df.interpolate(method="linear", inplace=True)
        df["current_lag1"] = df["current"].shift(1)
        df["current_lag2"] = df["current"].shift(2)
        df["voltage_lag1"] = df["voltage"].shift(1)
        df["current_rolling_mean"] = df["current"].rolling(window=10).mean()
        df["current_rolling_std"] = df["current"].rolling(window=10).std()
        coeffs = pywt.wavedec(df["current"], "db4", level=3)
        df["wavelet_current"] = pywt.waverec(coeffs, "db4")[:len(df)]
        df.dropna(inplace=True)
        logger.info(f"Preprocessed columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise ValueError(f"Preprocessing failed: {str(e)}")

def create_sequences(X, seq_length=30):
    Xs = []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
    return np.array(Xs)

def batch_predict(autoencoder_predict, X_seq, batch_size=1000):
    predictions = []
    n_samples = X_seq.shape[0]
    output_key = list(autoencoder_predict.structured_outputs.keys())[0]
    logger.info(f"Using output key: {output_key}")
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = tf.convert_to_tensor(X_seq[start:end], dtype=tf.float32)
        batch_pred = autoencoder_predict(inputs=batch)[output_key]
        predictions.append(np.array(batch_pred))
    return np.concatenate(predictions, axis=0)

def generate_plot(data, anomalies, n_seconds):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(n_seconds), data, label="Current (A)", color="blue")
    ax.scatter(anomalies, data[anomalies], color="red", label="Anomalies", marker="x")
    ax.set_title("Time-Series Anomaly Detection")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Current (A)")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return plot_base64

@app.get("/health")
async def health():
    return {"status": "API is running"}

@app.post("/detect_anomalies")
async def detect_anomalies(file: UploadFile = File(...), n_seconds: int = Form(..., ge=1)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        logger.info("Processing uploaded file")
        df = pd.read_csv(file.file)
        df_processed = preprocess_data(df)
        X = df_processed.drop(columns=["current"])
        y = df_processed["current"]
        X_scaled = scaler.transform(X)
        X_seq = create_sequences(X_scaled, seq_length=30)
        logger.info(f"X_seq shape: {X_seq.shape}")
        autoencoder_predict = autoencoder.signatures["serving_default"]
        recon = batch_predict(autoencoder_predict, X_seq, batch_size=1000)
        recon_errors = np.mean(np.power(X_seq - recon, 2), axis=(1, 2))
        recon_threshold = np.percentile(recon_errors, 90)
        recon_anomalies = recon_errors > recon_threshold

        reference_seq = np.median(X_seq[:1000], axis=0)  # Corrected
        dtw_distances = [fastdtw(seq, reference_seq, dist=euclidean)[0] for seq in X_seq]
        dtw_threshold = np.percentile(dtw_distances, 90)
        dtw_anomalies = np.array(dtw_distances) > dtw_threshold

        X_iso = X_scaled[30:]
        if X_iso.ndim != 2:
            X_iso = X_iso.reshape(X_iso.shape[0], -1)
        if np.any(~np.isfinite(X_iso)):
            logger.warning("NaNs or infinities detected in X_iso, replacing with zeros")
            X_iso = np.nan_to_num(X_iso, nan=0.0, posinf=0.0, neginf=0.0)

        if X_iso.shape[1] != getattr(iso_forest, 'n_features_in_', 20):
            logger.warning("Feature mismatch in IsolationForest input, skipping ISO prediction")
            iso_anomalies = np.ones_like(recon_anomalies, dtype=bool)
        else:
            iso_anomalies = iso_forest.predict(X_iso) == -1

        anomalies = np.logical_and.reduce([recon_anomalies, dtw_anomalies, iso_anomalies])
        anomaly_indices = np.where(anomalies)[0]
        plot_base64 = generate_plot(y[30:], anomaly_indices, len(y[30:]))

        return {
            "message": f"Detected {len(anomaly_indices)} anomalies",
            "anomaly_indices": anomaly_indices.tolist(),
            "anomaly_plot": f"data:image/png;base64,{plot_base64}",
            "summary": {
                "total_rows": len(y[30:]),
                "anomaly_percentage": len(anomaly_indices) / len(y[30:]) * 100
            }
        }
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
