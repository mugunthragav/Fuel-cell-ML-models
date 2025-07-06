import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import datetime
from io import BytesIO
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pywt
import base64
from joblib import Parallel, delayed
import time
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Combined Fuel Cell Models API", description="Unified API for all five fuel cell models")

# Dynamically determine BASE_DIR by searching for 'FC-final' from the script's location
script_dir = Path(__file__).resolve().parent
base_dir = next((p for p in script_dir.parents if p.name == 'FC-final'), script_dir)
if base_dir.name != 'FC-final':
    logger.warning(f"Could not find 'FC-final' in parent directories. Using script directory as base: {script_dir}")
BASE_DIR = base_dir

# Define model directories relative to BASE_DIR
MODEL_DIRS = {
    'model1': BASE_DIR / 'Anomaly_Detection_Stack_Voltage',
    'model2': BASE_DIR / 'Health_State_Classification',
    'model3': BASE_DIR / 'Time_Series_Anomaly_Detection',
    'model4': BASE_DIR / 'Transient_Current_Detection',
    'model5': BASE_DIR / 'Voltage_Prediction_Digital_Twin'
}

# Ensure combined_api's output directory exists
COMBINED_API_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / 'outputs'
os.makedirs(COMBINED_API_OUTPUTS_DIR, exist_ok=True)
logger.info(f"Ensured combined_api outputs directory exists at: {COMBINED_API_OUTPUTS_DIR}")

# Global variables for loaded models and artifacts
model1_dbscan = None
model1_scaler = None
model1_pca = None
model1_features = []

model2_rf = None
model2_rf_top = None
model2_scaler = None
model2_top_features = []
model2_critical_threshold = 3.1
model2_degraded_threshold = 3.25
model2_all_features = []

model3_scaler = None
model3_autoencoder = None
model3_iso_forest = None

model4_best_model = None
model4_scaler = None
model4_kbest_transformer = None
model4_initial_feature_names = []

model5_xgb = None
model5_xgb_top = None
model5_scaler = None
model5_top_features = []
model5_all_features = []

# Load models and required files
def load_all_models():
    global model1_dbscan, model1_scaler, model1_pca, model1_features
    global model2_rf, model2_rf_top, model2_scaler, model2_top_features, model2_critical_threshold, model2_degraded_threshold, model2_all_features
    global model3_scaler, model3_autoencoder, model3_iso_forest
    global model4_best_model, model4_scaler, model4_kbest_transformer, model4_initial_feature_names
    global model5_xgb, model5_xgb_top, model5_scaler, model5_top_features, model5_all_features

    try:
        # Model 1: Anomaly Detection Stack Voltage
        model1_dbscan = joblib.load(MODEL_DIRS['model1'] / 'models' / 'dbscan_model1.pkl')
        model1_scaler = joblib.load(MODEL_DIRS['model1'] / 'models' / 'scaler_model1.pkl')
        model1_pca = joblib.load(MODEL_DIRS['model1'] / 'models' / 'pca_model1.pkl')
        model1_features = ['Load', 'Loaddensity', 'H2Flow', 'TstackIn', 'TstackOut',
                          'CoolantFlow', 'AnodePressureDiff', 'CathodePressureDiff',
                          'RH_An', 'RH_Cat'] + [f'Cell{i}' for i in range(1, 97)] + ['Cell_Voltage_Std']
        logger.info("Loaded Model 1 files successfully.")

        # Model 2: Health State Classification
        model2_rf = joblib.load(MODEL_DIRS['model2'] / 'models' / 'rf_health_model3.pkl')
        model2_rf_top = joblib.load(MODEL_DIRS['model2'] / 'models' / 'rf_health_top_model3.pkl')
        model2_scaler = joblib.load(MODEL_DIRS['model2'] / 'models' / 'scaler_model3.pkl')
        model2_top_features = joblib.load(MODEL_DIRS['model2'] / 'models' / 'top_features_model3.pkl')
        try:
            with open(MODEL_DIRS['model2'] / 'outputs' / 'metrics_model3.txt', 'r') as f:
                import ast
                model2_metrics = ast.literal_eval(f.read())
            model2_critical_threshold = model2_metrics.get('critical_threshold', 3.1)
            model2_degraded_threshold = model2_metrics.get('degraded_threshold', 3.25)
        except FileNotFoundError:
            logger.warning("Model 2 metrics file not found, using fallback thresholds.")
        except Exception as e:
            logger.warning(f"Error loading Model 2 metrics: {e}, using fallback thresholds.")
        model2_all_features = ['Time (h)', 'TinH2 (°C)', 'ToutH2 (°C)', 'TinAIR (°C)', 'ToutAIR (°C)',
                             'TinWAT (°C)', 'ToutWAT (°C)', 'PinAIR (mbara)', 'PoutAIR (mbara)',
                             'PoutH2 (mbara)', 'PinH2 (mbara)', 'DinH2 (l/mn)', 'DoutH2 (l/mn)',
                             'DinAIR (l/mn)', 'DoutAIR (l/mn)', 'DWAT (l/mn)', 'HrAIRFC (%)']
        logger.info("Loaded Model 2 files successfully.")

        # Model 3: Time Series Anomaly Detection
        model3_scaler = joblib.load(MODEL_DIRS['model3'] / 'models' / 'scaler.pkl')
        model3_autoencoder = tf.saved_model.load(str(MODEL_DIRS['model3'] / 'models' / 'transformer_autoencoder'))
        model3_iso_forest = joblib.load(MODEL_DIRS['model3'] / 'models' / 'isolation_forest.pkl')
        logger.info(f"Loaded Model 3 files successfully - Scaler features: {getattr(model3_scaler, 'n_features_in_', 'Unknown')}.")

        # Model 4: Transient Current Detection
        model4_best_model = joblib.load(MODEL_DIRS['model4'] / 'models' / 'best_model.pkl')
        model4_scaler = joblib.load(MODEL_DIRS['model4'] / 'models' / 'scaler_model1.pkl')
        model4_kbest_transformer = joblib.load(MODEL_DIRS['model4'] / 'models' / 'kbest_transformer.pkl')
        model4_initial_feature_names = joblib.load(MODEL_DIRS['model4'] / 'models' / 'initial_feature_names.pkl')
        logger.info("Loaded Model 4 files successfully (best_model, scaler, kbest_transformer, initial_feature_names).")

        # Model 5: Voltage Prediction Digital Twin
        model5_xgb = joblib.load(MODEL_DIRS['model5'] / 'models' / 'xgb_utot_model2.pkl')
        model5_xgb_top = joblib.load(MODEL_DIRS['model5'] / 'models' / 'xgb_utot_top_model2.pkl')
        model5_scaler = joblib.load(MODEL_DIRS['model5'] / 'models' / 'scaler_model2.pkl')
        model5_top_features = joblib.load(MODEL_DIRS['model5'] / 'models' / 'top_features_model2.pkl')
        logger.info("Loaded Model 5 files successfully.")

    except FileNotFoundError as e:
        logger.error(f"A model file was not found: {e}. Please ensure all training scripts have been run and models are in their correct 'models/' directories.")
        raise HTTPException(status_code=500, detail=f"Failed to load a model file: {e}. Ensure training scripts were run.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load models due to an unexpected error: {e}")

# Call load_all_models when the application starts
@app.on_event("startup")
async def startup_event():
    load_all_models()

# Pydantic models
class Model1FuelCellData(BaseModel):
    Date_Time: str
    Load: float
    Loaddensity: float
    H2Flow: float
    TstackIn: float
    TstackOut: float
    CoolantFlow: float
    AnodePressureDiff: float
    CathodePressureDiff: float
    RH_An: float
    RH_Cat: float
    Cell_Voltages: List[float]

class Model2FuelCellData(BaseModel):
    Time_h: float
    TinH2: float
    ToutH2: float
    TinAIR: float
    ToutAIR: float
    TinWAT: float
    ToutWAT: float
    PinAIR: float
    PoutAIR: float
    PoutH2: float
    PinH2: float
    DinH2: float
    DoutH2: float
    DinAIR: float
    DoutAIR: float
    DWAT: float
    HrAIRFC: float

class Model4PredictionInput(BaseModel):
    features: Dict[str, float]

class Model5FuelCellData(BaseModel):
    Time_h: float
    TinH2: float
    ToutH2: float
    TinAIR: float
    ToutAIR: float
    TinWAT: float
    ToutWAT: float
    PinAIR: float
    PoutAIR: float
    PoutH2: float
    PinH2: float
    DinH2: float
    DoutH2: float
    DinAIR: float
    DoutAIR: float
    DWAT: float
    HrAIRFC: float

# Preprocessing functions
def preprocess_model1_data(df):
    df = df.drop_duplicates()
    if 'Date_Time' in df.columns:
        df.loc[:, 'Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        df.loc[:, 'Date_Time'] = df['Date_Time'].fillna('1970-01-01 00:00:00')
    cell_columns = [f'Cell{i}' for i in range(1, 97)]
    df.loc[:, 'Cell_Voltage_Std'] = df[cell_columns].std(axis=1)
    X = df[model1_features].copy()
    for col in ['Load', 'Loaddensity', 'H2Flow', 'TstackIn', 'TstackOut',
                'CoolantFlow', 'AnodePressureDiff', 'CathodePressureDiff',
                'RH_An', 'RH_Cat', 'Cell_Voltage_Std']:
        mean_val = X[col][X[col] != 0].mean()
        if pd.isna(mean_val): mean_val = 0.0
        X.loc[:, col] = X[col].replace(0, mean_val).fillna(mean_val)
    for col in cell_columns:
        median_val = X[col][X[col] != 0].median()
        if pd.isna(median_val): median_val = 0.0
        X.loc[:, col] = X[col].replace(0, median_val).fillna(median_val)
    X = X.fillna(X.mean(numeric_only=True))
    X_scaled = model1_scaler.transform(X)
    return X_scaled, df

def preprocess_model2_data(df, features):
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    for col in features:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df.loc[:, col] = df[col].clip(lower, upper)
        else:
            logger.warning(f"Feature '{col}' not found in DataFrame for clipping. Setting to 0.")
            df.loc[:, col] = 0.0
    X = df[features].copy()
    X_scaled = model2_scaler.transform(X)
    return X_scaled, df

def preprocess_model3_data(df):
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
        
        if len(df["current"]) >= (2**3) * 2:
            coeffs = pywt.wavedec(df["current"], "db4", level=3)
            df["wavelet_current"] = pywt.waverec(coeffs, "db4")[:len(df)]
        else:
            logger.warning("Not enough data points for wavelet transform level 3. Skipping wavelet feature.")
            df["wavelet_current"] = 0.0
            
        df.dropna(inplace=True)
        logger.info(f"Preprocessed columns for Model 3: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error in preprocess_model3_data: {str(e)}")
        raise ValueError(f"Preprocessing failed: {str(e)}")

def preprocess_model4_data(input_df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Ensure all required columns are present
        missing_cols = [col for col in model4_initial_feature_names if col not in input_df.columns]
        if missing_cols:
            logger.warning(f"Missing features in input: {missing_cols}. Filling with 0.")
            for col in missing_cols:
                input_df[col] = 0.0

        input_df = input_df[model4_initial_feature_names].copy()

        # Fill missing values using scaler means as a dictionary or fallback to DataFrame mean
        fill_values = {}
        if hasattr(model4_scaler, 'mean_'):
            for col, mean_val in zip(model4_initial_feature_names, model4_scaler.mean_):
                fill_values[col] = mean_val
        else:
            fill_values = input_df.mean(numeric_only=True).to_dict()
        input_df = input_df.fillna(value=fill_values)
        logger.info("Filled missing values with column means.")

        # Skip outlier removal for single-row input (z-score requires multiple samples)
        if len(input_df) > 1 and input_df.select_dtypes(include=np.number).shape[1] > 0:
            z_scores = np.abs(stats.zscore(input_df.select_dtypes(include=np.number)))
            mask = (z_scores < 3).all(axis=1)
            input_df = input_df[mask]
            logger.info("Removed outliers using z-score (threshold 3).")
        else:
            logger.info("Skipped outlier removal: Single row or no numeric columns detected.")

        if input_df.empty:
            raise ValueError("All rows were filtered out during preprocessing. Check input data or outlier removal.")

        # Apply scaling
        input_scaled = model4_scaler.transform(input_df)
        logger.info("Applied StandardScaler to input data.")

        # Apply feature selection
        input_transformed = model4_kbest_transformer.transform(input_scaled)
        logger.info(f"Applied SelectKBest transformation to select {input_transformed.shape[1]} features.")

        return pd.DataFrame(input_transformed, columns=[f'feature_{i}' for i in range(input_transformed.shape[1])])

    except Exception as e:
        logger.error(f"Error in preprocess_model4_data: {str(e)}", exc_info=True)
        raise

def preprocess_model5_data(df, features):
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    for col in features:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df.loc[:, col] = df[col].clip(lower, upper)
        else:
            logger.warning(f"Feature '{col}' not found in DataFrame for clipping. Setting to 0.")
            df.loc[:, col] = 0.0
    X = df[features].copy()
    X_scaled = model5_scaler.transform(X)
    return X_scaled, df

# Model 3 helper functions
def create_sequences(X, seq_length=30):
    Xs = []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
    return np.array(Xs)

def batch_predict(autoencoder_predict_func, X_seq, batch_size=1000):
    predictions = []
    n_samples = X_seq.shape[0]
    
    input_key = list(autoencoder_predict_func.inputs)[0].name.split(':')[0]
    output_key = list(autoencoder_predict_func.structured_outputs.keys())[0]
    logger.debug(f"Model 3 Autoencoder - Using input key: {input_key}, output key: {output_key}")

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = tf.convert_to_tensor(X_seq[start:end], dtype=tf.float32)
        batch_pred = autoencoder_predict_func(inputs={input_key: batch})[output_key]
        predictions.append(np.array(batch_pred))
    return np.concatenate(predictions, axis=0)

def generate_plot(data_series, anomalies_indices, total_points):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(data_series)), data_series, label="Current (A)", color="blue")
    ax.scatter(anomalies_indices, data_series.iloc[anomalies_indices], color="red", label="Anomalies", marker="x")
    ax.set_title("Time-Series Anomaly Detection")
    ax.set_xlabel("Time (data points)")
    ax.set_ylabel("Current (A)")
    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return plot_base64

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "API is running"}

# Model 1 Endpoints
@app.post("/model1/detect_anomalies/file")
async def model1_detect_anomalies_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        file_like = BytesIO(file_content)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file_like, engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or XLSX.")
        
        missing_features = [f for f in model1_features[:-1] if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        
        X_scaled, df_processed = preprocess_model1_data(df)
        cluster_labels = model1_dbscan.fit_predict(X_scaled)
        df_processed.loc[:, 'Cluster'] = cluster_labels
        
        cell_columns = [f'Cell{i}' for i in range(1, 97)]
        existing_cell_cols = [col for col in cell_columns if col in df_processed.columns]
        if existing_cell_cols:
            df_processed.loc[:, 'Min_Cell_Voltage'] = df_processed[existing_cell_cols].min(axis=1)
            df_processed.loc[:, 'Min_Cell_Index'] = df_processed[existing_cell_cols].idxmin(axis=1)
        else:
            df_processed.loc[:, 'Min_Cell_Voltage'] = np.nan
            df_processed.loc[:, 'Min_Cell_Index'] = "N/A"

        try:
            X_pca = model1_pca.transform(X_scaled)
            df_processed.loc[:, 'PCA1'] = X_pca[:, 0]
            df_processed.loc[:, 'PCA2'] = X_pca[:, 1]
        except Exception as pca_e:
            logger.error(f"Model 1 PCA transformation failed: {pca_e}. X_scaled shape: {X_scaled.shape}, PCA expected features: {model1_pca.n_features_in_}")
            df_processed.loc[:, 'PCA1'] = np.nan
            df_processed.loc[:, 'PCA2'] = np.nan
            
        anomalies = df_processed[df_processed['Cluster'] == -1].copy()
        if 'Date_Time' in anomalies.columns:
            anomalies.loc[:, 'Date_Time'] = pd.to_datetime(anomalies['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            anomalies.loc[:, 'Date_Time'] = anomalies['Date_Time'].fillna('1970-01-01 00:00:00')
            
        report_cols = ['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index',
                      'Load', 'Loaddensity', 'H2Flow', 'TstackIn',
                      'TstackOut', 'CoolantFlow', 'AnodePressureDiff',
                      'CathodePressureDiff', 'RH_An', 'RH_Cat']
        existing_report_cols = [col for col in report_cols if col in anomalies.columns]
        anomaly_report = anomalies[existing_report_cols].copy()

        if 'Date_Time' in anomaly_report.columns:
            anomaly_report.loc[:, 'Date_Time'] = pd.to_datetime(anomaly_report['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            anomaly_report.loc[:, 'Date_Time'] = anomaly_report['Date_Time'].fillna('1970-01-01 00:00:00')
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = COMBINED_API_OUTPUTS_DIR / f'realtime_anomalies_dbscan_model1_{timestamp}.csv'
        anomaly_report.to_csv(report_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        logger.info(f"Model 1 anomaly report saved to {report_path}")
            
        for cell in [f'Cell{i}' for i in range(1, 97)]:
            cell_anomalies = anomalies[anomalies['Min_Cell_Index'] == cell][['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index']]
            if not cell_anomalies.empty and 'Date_Time' in cell_anomalies.columns:
                cell_anomalies.loc[:, 'Date_Time'] = pd.to_datetime(cell_anomalies['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                cell_anomalies.loc[:, 'Date_Time'] = cell_anomalies['Date_Time'].fillna('1970-01-01 00:00:00')
                cell_anomalies.to_csv(COMBINED_API_OUTPUTS_DIR / f'realtime_anomalies_cell_{cell}_{timestamp}.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
        
        if 'PCA1' in df_processed.columns and 'PCA2' in df_processed.columns:
            plt.figure(figsize=(12, 8))
            normal_points = df_processed[df_processed['Cluster'] != -1]
            plt.scatter(normal_points['PCA1'], normal_points['PCA2'], c=normal_points['Cluster'],
                       cmap='viridis', s=50, alpha=0.6, label='Normal (Clustered)')
            plt.scatter(anomalies['PCA1'], anomalies['PCA2'], c='red', s=100, marker='x',
                       label='Anomalies', linewidths=2)
            plt.xlabel('PCA Component 1', fontsize=12)
            plt.ylabel('PCA Component 2', fontsize=12)
            plt.title('DBSCAN Clustering: Real-Time Anomalies', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plot_path = COMBINED_API_OUTPUTS_DIR / f'realtime_cluster_anomaly_plot_{timestamp}.png'
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info(f"Model 1 anomaly plot saved to {plot_path}")
        else:
            plot_path = "No plot generated due to missing PCA components."
            logger.warning("Skipping Model 1 anomaly plot: PCA components (PCA1, PCA2) not available.")
            
        anomaly_summary = {
            "total_rows": len(df_processed),
            "total_anomalies": len(anomalies),
            "anomaly_percentage": (len(anomalies) / len(df_processed)) * 100 if len(df_processed) > 0 else 0,
            "most_affected_cell": anomalies['Min_Cell_Index'].value_counts().idxmax() if not anomalies.empty and 'Min_Cell_Index' in anomalies.columns else "None",
            "min_cell_voltage_range": [float(anomalies['Min_Cell_Voltage'].min()) if not anomalies.empty and 'Min_Cell_Voltage' in anomalies.columns else 0,
                                      float(anomalies['Min_Cell_Voltage'].max()) if not anomalies.empty and 'Min_Cell_Voltage' in anomalies.columns else 0],
            "key_operating_conditions": {
                "avg_load": float(anomalies['Load'].mean()) if not anomalies.empty and 'Load' in anomalies.columns else 0,
                "avg_h2flow": float(anomalies['H2Flow'].mean()) if not anomalies.empty and 'H2Flow' in anomalies.columns else 0,
                "avg_tstackin": float(anomalies['TstackIn'].mean()) if not anomalies.empty and 'TstackIn' in anomalies.columns else 0
            }
        }
        
        anomaly_records = anomaly_report.to_dict(orient='records')
        for record in anomaly_records:
            if 'Date_Time' in record and isinstance(record['Date_Time'], pd.Timestamp):
                record['Date_Time'] = record['Date_Time'].strftime('%Y-%m-%d %H:%M:%S')
            elif 'Date_Time' in record and pd.isna(record['Date_Time']):
                record['Date_Time'] = '1970-01-01 00:00:00'
                
        return JSONResponse(content={
            "message": f"Detected {len(anomalies)} anomalies",
            "anomalies": anomaly_records,
            "plot_path": str(plot_path),
            "report_path": str(report_path),
            "summary": anomaly_summary
        })
    except Exception as e:
        logger.error(f"Error in model1_detect_anomalies_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model1/detect_anomalies/json")
async def model1_detect_anomalies_json(data: Model1FuelCellData):
    try:
        df = pd.DataFrame([{
            'Date_Time': data.Date_Time,
            'Load': data.Load,
            'Loaddensity': data.Loaddensity,
            'H2Flow': data.H2Flow,
            'TstackIn': data.TstackIn,
            'TstackOut': data.TstackOut,
            'CoolantFlow': data.CoolantFlow,
            'AnodePressureDiff': data.AnodePressureDiff,
            'CathodePressureDiff': data.CathodePressureDiff,
            'RH_An': data.RH_An,
            'RH_Cat': data.RH_Cat,
            **{f'Cell{i+1}': data.Cell_Voltages[i] for i in range(96)}
        }])
        
        X_scaled, df_processed = preprocess_model1_data(df)
        cluster_labels = model1_dbscan.fit_predict(X_scaled)
        df_processed.loc[:, 'Cluster'] = cluster_labels
        
        cell_columns = [f'Cell{i}' for i in range(1, 97)]
        existing_cell_cols = [col for col in cell_columns if col in df_processed.columns]
        if existing_cell_cols:
            df_processed.loc[:, 'Min_Cell_Voltage'] = df_processed[existing_cell_cols].min(axis=1)
            df_processed.loc[:, 'Min_Cell_Index'] = df_processed[existing_cell_cols].idxmin(axis=1)
        else:
            df_processed.loc[:, 'Min_Cell_Voltage'] = np.nan
            df_processed.loc[:, 'Min_Cell_Index'] = "N/A"

        result = df_processed[['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index',
                             'Load', 'Loaddensity', 'H2Flow', 'TstackIn',
                             'TstackOut', 'CoolantFlow', 'AnodePressureDiff',
                             'CathodePressureDiff', 'RH_An', 'RH_Cat']].iloc[0].to_dict()
        result['Is_Anomaly'] = cluster_labels[0] == -1
        
        if 'Date_Time' in result and isinstance(result['Date_Time'], pd.Timestamp):
            result['Date_Time'] = result['Date_Time'].strftime('%Y-%m-%d %H:%M:%S')
        elif 'Date_Time' in result and pd.isna(result['Date_Time']):
            result['Date_Time'] = '1970-01-01 00:00:00'
            
        return JSONResponse(content={"result": result})
    except Exception as e:
        logger.error(f"Error in model1_detect_anomalies_json: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model1/detect_anomalies/cell/{cell_number}")
async def model1_detect_anomalies_cell(cell_number: int):
    try:
        cell = f'Cell{cell_number}'
        if cell not in [f'Cell{i}' for i in range(1, 97)]:
            raise HTTPException(status_code=400, detail=f"Invalid cell number: {cell_number}. Must be between 1 and 96.")
        
        anomaly_report_path = MODEL_DIRS['model1'] / 'outputs' / 'anomalies_dbscan_model1.csv'
        if not anomaly_report_path.exists():
            list_of_files = sorted([f for f in (MODEL_DIRS['model1'] / 'outputs').glob('realtime_anomalies_dbscan_model1_*.csv')], key=os.path.getctime, reverse=True)
            if list_of_files:
                anomaly_report_path = list_of_files[0]
                logger.info(f"Found latest timestamped anomaly report for Model 1: {anomaly_report_path}")
            else:
                raise HTTPException(status_code=500, detail="No anomaly reports found. Run Model 1 file upload endpoint first.")

        try:
            anomaly_report = pd.read_csv(anomaly_report_path)
            if 'Date_Time' in anomaly_report.columns:
                anomaly_report.loc[:, 'Date_Time'] = pd.to_datetime(anomaly_report['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                anomaly_report.loc[:, 'Date_Time'] = anomaly_report['Date_Time'].fillna('1970-01-01 00:00:00')
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail=f"Anomaly report not found at {anomaly_report_path}. Run Model 1 training script or file upload endpoint first.")
        except Exception as e:
            logger.error(f"Error loading Model 1 anomaly report for cell endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error loading Model 1 anomaly report: {e}")
            
        cell_anomalies = anomaly_report[anomaly_report['Min_Cell_Index'] == cell][['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index']].copy()
        
        if 'Date_Time' in cell_anomalies.columns:
            cell_anomalies.loc[:, 'Date_Time'] = cell_anomalies['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, pd.Timestamp) else '1970-01-01 00:00:00')

        return JSONResponse(content={
            "message": f"Detected {len(cell_anomalies)} anomalies for {cell}",
            "anomalies": cell_anomalies.to_dict(orient='records')
        })
    except Exception as e:
        logger.error(f"Error in model1_detect_anomalies_cell: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Model 2 Endpoints
@app.post("/model2/predict_health/file")
async def model2_predict_health_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        file_like = BytesIO(file_content)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file_like, engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or XLSX.")
        
        missing_features = [f for f in model2_top_features if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        
        X_scaled, df_processed = preprocess_model2_data(df, model2_top_features)
        df_processed.loc[:, 'Health_State_Predicted'] = model2_rf_top.predict(X_scaled)
        df_processed.loc[:, 'Health_State_Label'] = df_processed['Health_State_Predicted'].map({0: 'Critical', 1: 'Degraded', 2: 'Healthy'})
        
        summary = {
            "total_rows": len(df_processed),
            "health_state_counts": df_processed['Health_State_Label'].value_counts().to_dict(),
            "thresholds": {
                "Critical": f"< {model2_critical_threshold:.3f} V",
                "Degraded": f"{model2_critical_threshold:.3f}–{model2_degraded_threshold:.3f} V",
                "Healthy": f"≥ {model2_degraded_threshold:.3f} V"
            }
        }
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = COMBINED_API_OUTPUTS_DIR / f'health_predictions_model2_{timestamp}.csv'
        
        cols_to_save = ['Time (h)', 'Health_State_Predicted', 'Health_State_Label'] + model2_top_features
        existing_cols_to_save = [col for col in cols_to_save if col in df_processed.columns]
        df_processed[existing_cols_to_save].to_csv(report_path, index=False)
        logger.info(f"Model 2 health predictions report saved to {report_path}")
        
        return JSONResponse(content={
            "message": f"Predicted Health State for {len(df_processed)} rows",
            "predictions": df_processed[existing_cols_to_save].to_dict(orient='records'),
            "report_path": str(report_path),
            "summary": summary
        })
    except Exception as e:
        logger.error(f"Error in model2_predict_health_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model2/predict_health/json")
async def model2_predict_health_json(data: Model2FuelCellData):
    try:
        df = pd.DataFrame([{
            'Time (h)': data.Time_h,
            'TinH2 (°C)': data.TinH2,
            'ToutH2 (°C)': data.ToutH2,
            'TinAIR (°C)': data.TinAIR,
            'ToutAIR (°C)': data.ToutAIR,
            'TinWAT (°C)': data.TinWAT,
            'ToutWAT (°C)': data.ToutWAT,
            'PinAIR (mbara)': data.PinAIR,
            'PoutAIR (mbara)': data.PoutAIR,
            'PoutH2 (mbara)': data.PoutH2,
            'PinH2 (mbara)': data.PinH2,
            'DinH2 (l/mn)': data.DinH2,
            'DoutH2 (l/mn)': data.DoutH2,
            'DinAIR (l/mn)': data.DinAIR,
            'DoutAIR (l/mn)': data.DoutAIR,
            'DWAT (l/mn)': data.DWAT,
            'HrAIRFC (%)': data.HrAIRFC
        }])
        
        missing_features = [f for f in model2_top_features if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        
        X_scaled, df_processed = preprocess_model2_data(df, model2_top_features)
        df_processed.loc[:, 'Health_State_Predicted'] = model2_rf_top.predict(X_scaled)
        df_processed.loc[:, 'Health_State_Label'] = df_processed['Health_State_Predicted'].map({0: 'Critical', 1: 'Degraded', 2: 'Healthy'})
        
        result_cols = ['Time (h)', 'Health_State_Predicted', 'Health_State_Label'] + model2_top_features
        existing_result_cols = [col for col in result_cols if col in df_processed.columns]
        result = df_processed[existing_result_cols].iloc[0].to_dict()
        
        return JSONResponse(content={
            "result": result,
            "thresholds": {
                "Critical": f"< {model2_critical_threshold:.3f} V",
                "Degraded": f"{model2_critical_threshold:.3f}–{model2_degraded_threshold:.3f} V",
                "Healthy": f"≥ {model2_degraded_threshold:.3f} V"
            }
        })
    except Exception as e:
        logger.error(f"Error in model2_predict_health_json: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Model 3 Endpoints
@app.post("/model3/detect_anomalies")
async def model3_detect_anomalies(file: UploadFile = File(...), n_seconds: int = Form(..., ge=1)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    try:
        logger.info("Processing uploaded file for Model 3 Time-Series Anomaly Detection")
        df = pd.read_csv(file.file)
        df_processed = preprocess_model3_data(df)
        
        features_for_scaler = [col for col in df_processed.columns if col != 'current']
        X_for_scaling = df_processed[features_for_scaler]
        
        if np.any(~np.isfinite(X_for_scaling)):
            logger.warning("NaNs or infinities detected in X_for_scaling for Model 3, replacing with zeros.")
            X_for_scaling = np.nan_to_num(X_for_scaling, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = model3_scaler.transform(X_for_scaling)
        y_current = df_processed["current"]
        
        seq_length = 30
        if len(X_scaled) < seq_length:
            raise HTTPException(status_code=400, detail=f"Input data too short for sequence creation. Requires at least {seq_length} data points.")

        X_seq = create_sequences(X_scaled, seq_length=seq_length)
        logger.info(f"Model 3 X_seq shape: {X_seq.shape}")
        
        autoencoder_predict = model3_autoencoder.signatures["serving_default"]
        recon = batch_predict(autoencoder_predict, X_seq, batch_size=1000)
        recon_errors = np.mean(np.power(X_seq - recon, 2), axis=(1, 2))
        recon_threshold = np.percentile(recon_errors, 90)
        recon_anomalies = recon_errors > recon_threshold

        if X_seq.shape[0] > 0:
            reference_seq = X_seq[0]
            if reference_seq.ndim == 2 and reference_seq.shape[0] < seq_length:
                reference_seq = np.pad(reference_seq, ((0, seq_length - reference_seq.shape[0]), (0, reference_seq.shape[1])), 'constant')
            elif reference_seq.ndim == 1 and reference_seq.shape[0] < seq_length:
                reference_seq = np.pad(reference_seq, (0, seq_length - reference_seq.shape[0]), 'constant')
            
            dtw_distances = [fastdtw(seq, reference_seq, dist=euclidean)[0] for seq in X_seq]
            dtw_threshold = np.percentile(dtw_distances, 90)
            dtw_anomalies = np.array(dtw_distances) > dtw_threshold
        else:
            logger.warning("No sequences created for DTW in Model 3. Skipping DTW anomaly detection.")
            dtw_anomalies = np.zeros_like(recon_anomalies, dtype=bool)
            dtw_distances = []

        X_iso = X_scaled[seq_length:]
        if X_iso.ndim != 2:
            X_iso = X_iso.reshape(X_iso.shape[0], -1)
        if np.any(~np.isfinite(X_iso)):
            logger.warning("NaNs or infinities detected in X_iso for Model 3, replacing with zeros.")
            X_iso = np.nan_to_num(X_iso, nan=0.0, posinf=0.0, neginf=0.0)

        if hasattr(model3_iso_forest, 'n_features_in_') and X_iso.shape[1] != model3_iso_forest.n_features_in_:
            logger.warning(f"Feature mismatch in Model 3 IsolationForest input. Expected {model3_iso_forest.n_features_in_} features, got {X_iso.shape[1]}. Skipping ISO prediction.")
            iso_anomalies = np.ones_like(recon_anomalies, dtype=bool)
        else:
            if X_iso.shape[0] > 0:
                iso_anomalies = model3_iso_forest.predict(X_iso) == -1
            else:
                logger.warning("X_iso is empty for Model 3. Skipping ISO prediction.")
                iso_anomalies = np.zeros_like(recon_anomalies, dtype=bool)

        min_len = min(len(recon_anomalies), len(dtw_anomalies), len(iso_anomalies))
        anomalies = np.logical_and.reduce([
            recon_anomalies[:min_len],
            dtw_anomalies[:min_len],
            iso_anomalies[:min_len]
        ])
        anomaly_indices = np.where(anomalies)[0]
        
        plot_base64 = generate_plot(y_current[seq_length:], anomaly_indices, len(y_current[seq_length:]))

        return JSONResponse(content={
            "message": f"Detected {len(anomaly_indices)} anomalies for Model 3",
            "anomaly_indices": anomaly_indices.tolist(),
            "anomaly_plot": f"data:image/png;base64,{plot_base64}",
            "summary": {
                "total_rows": len(y_current[seq_length:]),
                "anomaly_percentage": len(anomaly_indices) / len(y_current[seq_length:]) * 100 if len(y_current[seq_length:]) > 0 else 0
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in model3_detect_anomalies: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Model 4 Endpoints
@app.post("/model4/predict")
async def model4_predict(input_data: Model4PredictionInput):
    try:
        # Convert input dictionary to DataFrame with a single row
        input_df = pd.DataFrame([input_data.features])
        if input_df.empty or input_df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Input data is empty. Please provide at least one set of feature values.")

        # Log the raw input for debugging
        logger.info(f"Raw input DataFrame for Model 4: {input_df.to_dict()}")

        # Preprocess the data
        input_transformed = preprocess_model4_data(input_df)
        
        if input_transformed.empty or input_transformed.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Preprocessed data is empty. Check input features or preprocessing steps.")

        # Ensure input_transformed is 2D for prediction
        if input_transformed.ndim == 1:
            input_transformed = input_transformed.reshape(1, -1)

        # Make prediction
        prediction_proba = model4_best_model.predict_proba(input_transformed)
        prediction = model4_best_model.predict(input_transformed)[0]
        confidence = prediction_proba[0, prediction] if prediction_proba.size > 0 else 0.0

        # Determine label based on three-class model, mapping 1 and 2 to Faulty
        label = "Normal" if prediction == 0 else "Faulty"  # Map 1 and 2 to Faulty

        response = {
            "prediction_status": "Success",
            "prediction": label,
            "confidence": float(confidence),
            "features_provided": list(input_data.features.keys()),
            "predicted_class": int(prediction)  # Return the raw predicted class (0, 1, or 2) for transparency
        }
        logger.info(f"Model 4 Prediction made: {response}")
        
        return JSONResponse(content=response)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in model4_predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error in Model 4: {str(e)}")

# Model 5 Endpoints
@app.post("/model5/predict_utot/file")
async def model5_predict_utot_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        file_like = BytesIO(file_content)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file_like, engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or XLSX.")
        
        missing_features = [f for f in model5_top_features if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        
        X_scaled, df_processed = preprocess_model5_data(df, model5_top_features)
        df_processed.loc[:, 'Utot_Predicted'] = model5_xgb_top.predict(X_scaled)
        
        summary = {
            "total_rows": len(df_processed),
            "mean_utot_predicted": float(df_processed['Utot_Predicted'].mean()),
            "utot_range": [float(df_processed['Utot_Predicted'].min()), float(df_processed['Utot_Predicted'].max())]
        }
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = COMBINED_API_OUTPUTS_DIR / f'utot_predictions_model5_{timestamp}.csv'
        
        cols_to_save = ['Time (h)', 'Utot_Predicted'] + model5_top_features
        existing_cols_to_save = [col for col in cols_to_save if col in df_processed.columns]
        df_processed[existing_cols_to_save].to_csv(report_path, index=False)
        logger.info(f"Model 5 Utot predictions report saved to {report_path}")
        
        return JSONResponse(content={
            "message": f"Predicted Utot for {len(df_processed)} rows",
            "predictions": df_processed[existing_cols_to_save].to_dict(orient='records'),
            "report_path": str(report_path),
            "summary": summary
        })
    except Exception as e:
        logger.error(f"Error in model5_predict_utot_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model5/predict_utot/json")
async def model5_predict_utot_json(data: Model5FuelCellData):
    try:
        df = pd.DataFrame([{
            'Time (h)': data.Time_h,
            'TinH2 (°C)': data.TinH2,
            'ToutH2 (°C)': data.ToutH2,
            'TinAIR (°C)': data.TinAIR,
            'ToutAIR (°C)': data.ToutAIR,
            'TinWAT (°C)': data.TinWAT,
            'ToutWAT (°C)': data.ToutWAT,
            'PinAIR (mbara)': data.PinAIR,
            'PoutAIR (mbara)': data.PoutAIR,
            'PoutH2 (mbara)': data.PoutH2,
            'PinH2 (mbara)': data.PinH2,
            'DinH2 (l/mn)': data.DinH2,
            'DoutH2 (l/mn)': data.DoutH2,
            'DinAIR (l/mn)': data.DinAIR,
            'DoutAIR (l/mn)': data.DoutAIR,
            'DWAT (l/mn)': data.DWAT,
            'HrAIRFC (%)': data.HrAIRFC
        }])
        
        X_scaled, df_processed = preprocess_model5_data(df, model5_top_features)
        df_processed.loc[:, 'Utot_Predicted'] = model5_xgb_top.predict(X_scaled)
        result = df_processed[['Time (h)', 'Utot_Predicted'] + model5_top_features].iloc[0].to_dict()
        
        return JSONResponse(content={"result": result})
    except Exception as e:
        logger.error(f"Error in model5_predict_utot_json: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))