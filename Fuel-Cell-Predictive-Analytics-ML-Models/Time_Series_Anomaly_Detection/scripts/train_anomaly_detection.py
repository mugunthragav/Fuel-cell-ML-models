import pandas as pd
import numpy as np
import pywt
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Reshape, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load and preprocess data
logger.info("Loading and preprocessing data")
df = pd.read_excel("data/combined_data.xlsx")
df.drop(columns=["power"], inplace=True)
df["time"] = pd.to_datetime(df["time"], unit="s", origin="1970-01-01")
df.set_index("time", inplace=True)
df.interpolate(method="linear", inplace=True)

# Feature engineering
logger.info("Performing feature engineering")
df["current_lag1"] = df["current"].shift(1)
df["current_lag2"] = df["current"].shift(2)
df["voltage_lag1"] = df["voltage"].shift(1)
df["current_rolling_mean"] = df["current"].rolling(window=10).mean()
df["current_rolling_std"] = df["current"].rolling(window=10).std()
coeffs = pywt.wavedec(df["current"], "db4", level=3)
df["wavelet_current"] = pywt.waverec(coeffs, "db4")[:len(df)]
df.dropna(inplace=True)
df.to_csv("data/preprocessed_data.csv")

# Train-test split
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]
X_train, X_test = train.drop(columns=["current"]), test.drop(columns=["current"])
y_train, y_test = train["current"], test["current"]

# Scale features
logger.info("Scaling features")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# Prepare sequences (length=30)
def create_sequences(X, y, seq_length=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y.iloc[i + seq_length])
    return np.array(Xs), np.array(ys)

seq_length = 30
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)

# Transformer-Autoencoder
def transformer_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Lambda(lambda x: x[:, -1, :])(x)
    x = Dense(8, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(input_shape[1] * input_shape[0])(x)
    outputs = Reshape((input_shape[0], input_shape[1]))(x)
    return Model(inputs, outputs)

logger.info("Training Transformer-Autoencoder")
autoencoder = transformer_autoencoder((seq_length, X_train_seq.shape[2]))
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_train_seq, X_train_seq, epochs=10, batch_size=128, validation_split=0.1, verbose=1)
tf.saved_model.save(autoencoder, "models/transformer_autoencoder")

# Isolation Forest
logger.info("Training Isolation Forest")
iso_forest = IsolationForest(contamination=0.05, n_estimators=100)
iso_forest.fit(X_train_scaled)
joblib.dump(iso_forest, "models/isolation_forest.pkl")

# DTW for sequence-based anomaly detection
logger.info("Computing DTW distances")
reference_seq = np.median(X_train_seq[:1000], axis=0)  # Corrected
dtw_distances = [fastdtw(seq, reference_seq, dist=euclidean)[0] for seq in X_test_seq]
dtw_threshold = np.percentile(dtw_distances, 90)
dtw_anomalies = np.array(dtw_distances) > dtw_threshold

# Anomaly detection
logger.info("Detecting anomalies")
recon = autoencoder.predict(X_test_seq)
recon_errors = np.mean(np.power(X_test_seq - recon, 2), axis=(1, 2))
recon_threshold = np.percentile(recon_errors, 90)
recon_anomalies = recon_errors > recon_threshold
iso_anomalies = iso_forest.predict(X_test_scaled[seq_length:]) == -1
anomalies = np.logical_and.reduce([recon_anomalies, dtw_anomalies, iso_anomalies])
anomaly_indices = np.where(anomalies)[0]

# Metrics
with open("outputs/metrics.txt", "w") as f:
    f.write(f"Anomaly Percentage: {len(anomaly_indices) / len(y_test_seq) * 100:.2f}%\n")

# Visualizations
plt.figure(figsize=(12, 6))
plt.plot(y_test_seq, label="Current (A)", color="blue")
plt.scatter(anomaly_indices, y_test_seq[anomaly_indices], color="red", label="Anomalies", marker="x")
plt.title("Time-Series Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Current (A)")
plt.legend()
plt.savefig(f"outputs/anomaly_plot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.close()

explainer = shap.KernelExplainer(iso_forest.decision_function, X_train_scaled[:100])
shap_values = explainer.shap_values(X_test_scaled[seq_length:][:100])
shap.summary_plot(shap_values, X_test.iloc[seq_length:][:100], show=False)
plt.savefig(f"outputs/feature_importance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.close()

# Save anomaly report
test["anomaly"] = 0
test.iloc[seq_length:]["anomaly"].iloc[anomaly_indices] = 1
test.to_csv(f"outputs/anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
logger.info("Training completed")
