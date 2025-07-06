from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import os
from pydantic import BaseModel
from typing import List
import datetime
from io import BytesIO
import logging
from pandas import Timestamp

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Fuel Cell Anomaly Detection API", description="API for real-time anomaly detection in fuel cell data")

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

# Load pre-trained models
try:
    dbscan = joblib.load('models/dbscan_model1.pkl')
    scaler = joblib.load('models/scaler_model1.pkl')
    pca = joblib.load('models/pca_model1.pkl')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model files not found in 'models/' directory")

# Define features
features = ['Load', 'Loaddensity', 'H2Flow', 'TstackIn', 'TstackOut', 
            'CoolantFlow', 'AnodePressureDiff', 'CathodePressureDiff', 
            'RH_An', 'RH_Cat'] + [f'Cell{i}' for i in range(1, 97)] + ['Cell_Voltage_Std']

# Pydantic model for JSON input
class FuelCellData(BaseModel):
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
    Cell_Voltages: List[float]  # List of 96 cell voltages

# Preprocess data
def preprocess_data(df):
    df = df.drop_duplicates()
    
    # Convert Date_Time to string early
    if 'Date_Time' in df.columns:
        logger.debug(f"Preprocess - Date_Time type before conversion: {df['Date_Time'].dtype}")
        df.loc[:, 'Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        df.loc[:, 'Date_Time'] = df['Date_Time'].fillna('1970-01-01 00:00:00')
        logger.debug(f"Preprocess - Date_Time type after conversion: {df['Date_Time'].dtype}")
    
    # Add Cell_Voltage_Std using .loc to avoid SettingWithCopyWarning
    cell_columns = [f'Cell{i}' for i in range(1, 97)]
    df.loc[:, 'Cell_Voltage_Std'] = df[cell_columns].std(axis=1)
    
    # Select features
    X = df[features].copy()
    
    # Handle zeros and NaNs
    for col in ['Load', 'Loaddensity', 'H2Flow', 'TstackIn', 'TstackOut', 
                'CoolantFlow', 'AnodePressureDiff', 'CathodePressureDiff', 
                'RH_An', 'RH_Cat', 'Cell_Voltage_Std']:
        X.loc[:, col] = X[col].replace(0, X[col].mean())
    for col in cell_columns:
        X.loc[:, col] = X[col].replace(0, X[col].median())
    X = X.fillna(X.mean())
    
    # Normalize
    X_scaled = scaler.transform(X)
    
    return X_scaled, df

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "API is running"}

# File upload endpoint
@app.post("/detect_anomalies/file")
async def detect_anomalies_file(file: UploadFile = File(...)):
    try:
        # Read file contents into BytesIO
        file_content = await file.read()
        file_like = BytesIO(file_content)
        
        # Read file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file_like, engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or XLSX.")
        
        logger.debug(f"Initial df columns: {df.columns.tolist()}")
        logger.debug(f"Initial Date_Time type: {df['Date_Time'].dtype if 'Date_Time' in df.columns else 'Not present'}")
        
        # Convert Date_Time immediately after reading
        if 'Date_Time' in df.columns:
            df.loc[:, 'Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            df.loc[:, 'Date_Time'] = df['Date_Time'].fillna('1970-01-01 00:00:00')
            logger.debug(f"After initial read - Date_Time type: {df['Date_Time'].dtype}")
        
        # Verify features
        missing_features = [f for f in features[:-1] if f not in df.columns]  # Exclude Cell_Voltage_Std
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        
        # Preprocess
        X_scaled, df = preprocess_data(df)
        
        # Predict anomalies
        cluster_labels = dbscan.fit_predict(X_scaled)
        df.loc[:, 'Cluster'] = cluster_labels
        
        # Add Min_Cell_Voltage, Min_Cell_Index, and PCA columns
        df.loc[:, 'Min_Cell_Voltage'] = df[[f'Cell{i}' for i in range(1, 97)]].min(axis=1)
        df.loc[:, 'Min_Cell_Index'] = df[[f'Cell{i}' for i in range(1, 97)]].idxmin(axis=1)
        X_pca = pca.transform(X_scaled)
        df.loc[:, 'PCA1'] = X_pca[:, 0]
        df.loc[:, 'PCA2'] = X_pca[:, 1]
        
        # Filter anomalies
        anomalies = df[df['Cluster'] == -1]
        if 'Date_Time' in anomalies.columns:
            anomalies.loc[:, 'Date_Time'] = pd.to_datetime(anomalies['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            anomalies.loc[:, 'Date_Time'] = anomalies['Date_Time'].fillna('1970-01-01 00:00:00')
            logger.debug(f"Anomalies - Date_Time type: {anomalies['Date_Time'].dtype}")
        
        # Generate anomaly report
        anomaly_report = anomalies[['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index', 
                                   'Load', 'Loaddensity', 'H2Flow', 'TstackIn', 
                                   'TstackOut', 'CoolantFlow', 'AnodePressureDiff', 
                                   'CathodePressureDiff', 'RH_An', 'RH_Cat']]
        if 'Date_Time' in anomaly_report.columns:
            anomaly_report.loc[:, 'Date_Time'] = pd.to_datetime(anomaly_report['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            anomaly_report.loc[:, 'Date_Time'] = anomaly_report['Date_Time'].fillna('1970-01-01 00:00:00')
            logger.debug(f"Anomaly report - Date_Time type before serialization: {anomaly_report['Date_Time'].dtype}")
            logger.debug(f"Anomaly report - First few Date_Time values: {anomaly_report['Date_Time'].head().tolist()}")
        
        # Save anomaly report to CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'outputs/realtime_anomalies_dbscan_model1_{timestamp}.csv'
        anomaly_report.to_csv(report_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        
        # Generate cell-specific anomaly reports
        for cell in [f'Cell{i}' for i in range(1, 97)]:
            cell_anomalies = anomalies[anomalies['Min_Cell_Index'] == cell][['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index']]
            if not cell_anomalies.empty and 'Date_Time' in cell_anomalies.columns:
                cell_anomalies.loc[:, 'Date_Time'] = pd.to_datetime(cell_anomalies['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                cell_anomalies.loc[:, 'Date_Time'] = cell_anomalies['Date_Time'].fillna('1970-01-01 00:00:00')
                logger.debug(f"Cell {cell} anomalies - Date_Time type: {cell_anomalies['Date_Time'].dtype}")
            cell_anomalies.to_csv(f'outputs/realtime_anomalies_cell_{cell}_{timestamp}.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
        
        # Visualize
        plt.figure(figsize=(12, 8))
        normal_points = df[df['Cluster'] != -1]
        if 'Date_Time' in normal_points.columns:
            normal_points.loc[:, 'Date_Time'] = pd.to_datetime(normal_points['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            normal_points.loc[:, 'Date_Time'] = normal_points['Date_Time'].fillna('1970-01-01 00:00:00')
            logger.debug(f"Normal points - Date_Time type: {normal_points['Date_Time'].dtype}")
        plt.scatter(normal_points['PCA1'], normal_points['PCA2'], c=normal_points['Cluster'], 
                    cmap='viridis', s=50, alpha=0.6, label='Normal (Clustered)')
        plt.scatter(anomalies['PCA1'], anomalies['PCA2'], c='red', s=100, marker='x', 
                    label='Anomalies', linewidths=2)
        plt.xlabel('PCA Component 1', fontsize=12)
        plt.ylabel('PCA Component 2', fontsize=12)
        plt.title('DBSCAN Clustering: Real-Time Anomalies', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plot_path = f'outputs/realtime_cluster_anomaly_plot_{timestamp}.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        # Generate anomaly summary
        anomaly_summary = {
            "total_rows": len(df),
            "total_anomalies": len(anomalies),
            "anomaly_percentage": (len(anomalies) / len(df)) * 100 if len(df) > 0 else 0,
            "most_affected_cell": anomalies['Min_Cell_Index'].value_counts().idxmax() if not anomalies.empty else "None",
            "min_cell_voltage_range": [float(anomalies['Min_Cell_Voltage'].min()) if not anomalies.empty else 0, 
                                      float(anomalies['Min_Cell_Voltage'].max()) if not anomalies.empty else 0],
            "key_operating_conditions": {
                "avg_load": float(anomalies['Load'].mean()) if not anomalies.empty else 0,
                "avg_h2flow": float(anomalies['H2Flow'].mean()) if not anomalies.empty else 0,
                "avg_tstackin": float(anomalies['TstackIn'].mean()) if not anomalies.empty else 0
            }
        }
        
        # Final serialization check
        anomaly_records = anomaly_report.to_dict(orient='records')
        for record in anomaly_records:
            if 'Date_Time' in record:
                if isinstance(record['Date_Time'], Timestamp):
                    logger.warning(f"Found Timestamp in record: {record['Date_Time']}")
                    record['Date_Time'] = record['Date_Time'].strftime('%Y-%m-%d %H:%M:%S')
                elif pd.isna(record['Date_Time']):
                    record['Date_Time'] = '1970-01-01 00:00:00'
                logger.debug(f"Record Date_Time value: {record['Date_Time']} (type: {type(record['Date_Time']).__name__})")
        
        return JSONResponse(content={
            "message": f"Detected {len(anomalies)} anomalies",
            "anomalies": anomaly_records,
            "plot_path": plot_path,
            "report_path": report_path,
            "summary": anomaly_summary
        })
    except Exception as e:
        logger.error(f"Error in detect_anomalies_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# JSON input endpoint
@app.post("/detect_anomalies/json")
async def detect_anomalies_json(data: FuelCellData):
    try:
        # Convert JSON to DataFrame
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
        
        logger.debug(f"JSON input - Date_Time type: {df['Date_Time'].dtype}")
        logger.debug(f"JSON input - Cell_Voltages length: {len(data.Cell_Voltages)}")
        logger.debug(f"JSON input - Data: {df.iloc[0].to_dict()}")
        
        # Preprocess
        X_scaled, df = preprocess_data(df)
        
        # Predict anomaly
        cluster_labels = dbscan.fit_predict(X_scaled)
        df.loc[:, 'Cluster'] = cluster_labels
        
        # Generate result
        df.loc[:, 'Min_Cell_Voltage'] = df[[f'Cell{i}' for i in range(1, 97)]].min(axis=1)
        df.loc[:, 'Min_Cell_Index'] = df[[f'Cell{i}' for i in range(1, 97)]].idxmin(axis=1)
        result = df[['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index', 
                     'Load', 'Loaddensity', 'H2Flow', 'TstackIn', 
                     'TstackOut', 'CoolantFlow', 'AnodePressureDiff', 
                     'CathodePressureDiff', 'RH_An', 'RH_Cat']].iloc[0].to_dict()
        result['Is_Anomaly'] = cluster_labels[0] == -1
        
        logger.debug(f"JSON result - Date_Time type: {result['Date_Time'].__class__.__name__}")
        
        return JSONResponse(content={"result": result})
    except Exception as e:
        logger.error(f"Error in detect_anomalies_json: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Cell-specific anomaly endpoint
@app.get("/detect_anomalies/cell/{cell_number}")
async def detect_anomalies_cell(cell_number: int):
    try:
        cell = f'Cell{cell_number}'
        if cell not in [f'Cell{i}' for i in range(1, 97)]:
            raise HTTPException(status_code=400, detail=f"Invalid cell number: {cell_number}. Must be between 1 and 96.")
        
        # Load original anomaly report
        try:
            anomaly_report = pd.read_csv('outputs/anomalies_dbscan_model1.csv')
            if 'Date_Time' in anomaly_report.columns:
                logger.debug(f"Cell anomaly report - Date_Time type before conversion: {anomaly_report['Date_Time'].dtype}")
                anomaly_report.loc[:, 'Date_Time'] = pd.to_datetime(anomaly_report['Date_Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                anomaly_report.loc[:, 'Date_Time'] = anomaly_report['Date_Time'].fillna('1970-01-01 00:00:00')
                logger.debug(f"Cell anomaly report - Date_Time type after conversion: {anomaly_report['Date_Time'].dtype}")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Anomaly report not found. Run training script first.")
        
        # Filter anomalies for the specified cell
        cell_anomalies = anomaly_report[anomaly_report['Min_Cell_Index'] == cell][['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index']]
        
        return JSONResponse(content={
            "message": f"Detected {len(cell_anomalies)} anomalies for {cell}",
            "anomalies": cell_anomalies.to_dict(orient='records')
        })
    except Exception as e:
        logger.error(f"Error in detect_anomalies_cell: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))