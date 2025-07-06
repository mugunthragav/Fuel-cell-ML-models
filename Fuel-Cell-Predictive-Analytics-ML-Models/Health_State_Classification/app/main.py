from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pydantic import BaseModel
import datetime
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Fuel Cell Model API", description="Digital Twin for Health State Classification")

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

# Load pre-trained models and features
try:
    rf_health_model = joblib.load('models/rf_health_model3.pkl')
    rf_health_top_model = joblib.load('models/rf_health_top_model3.pkl')
    scaler_health = joblib.load('models/scaler_model3.pkl')
    top_features_health = joblib.load('models/top_features_model3.pkl')
except FileNotFoundError:
    logger.error("Model 3 files not found in 'models/' directory")
    raise HTTPException(status_code=500, detail="Model 3 files not found in 'models/' directory")

# Load thresholds from metrics
try:
    with open('outputs/metrics_model3.txt', 'r') as f:
        metrics = eval(f.read())
    critical_threshold = metrics['critical_threshold']
    degraded_threshold = metrics['degraded_threshold']
except FileNotFoundError:
    critical_threshold, degraded_threshold = 3.1, 3.25  # Fallback
    logger.warning("Metrics file not found, using fallback thresholds")

# Define all features
all_features = ['Time (h)', 'TinH2 (°C)', 'ToutH2 (°C)', 'TinAIR (°C)', 'ToutAIR (°C)', 
                'TinWAT (°C)', 'ToutWAT (°C)', 'PinAIR (mbara)', 'PoutAIR (mbara)', 
                'PoutH2 (mbara)', 'PinH2 (mbara)', 'DinH2 (l/mn)', 'DoutH2 (l/mn)', 
                'DinAIR (l/mn)', 'DoutAIR (l/mn)', 'DWAT (l/mn)', 'HrAIRFC (%)']

# Pydantic model for JSON input
class FuelCellData(BaseModel):
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

# Preprocess data
def preprocess_data(df, features, scaler):
    df = df.drop_duplicates()
    df = df.fillna(df.mean())
    for col in features:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df.loc[:, col] = df[col].clip(lower, upper)
    X = df[features].copy()
    X_scaled = scaler.transform(X)
    return X_scaled, df

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "API is running"}

# File upload endpoint for Health State prediction
@app.post("/predict_health/file")
async def predict_health_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        file_like = BytesIO(file_content)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file_like, engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or XLSX.")
        
        logger.debug(f"Initial df columns: {df.columns.tolist()}")
        missing_features = [f for f in top_features_health if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        
        X_scaled, df = preprocess_data(df, top_features_health, scaler_health)
        df.loc[:, 'Health_State_Predicted'] = rf_health_top_model.predict(X_scaled)
        df.loc[:, 'Health_State_Label'] = df['Health_State_Predicted'].map({0: 'Critical', 1: 'Degraded', 2: 'Healthy'})
        
        summary = {
            "total_rows": len(df),
            "health_state_counts": df['Health_State_Label'].value_counts().to_dict(),
            "thresholds": {
                "Critical": f"< {critical_threshold:.3f} V",
                "Degraded": f"{critical_threshold:.3f}–{degraded_threshold:.3f} V",
                "Healthy": f"≥ {degraded_threshold:.3f} V"
            }
        }
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'outputs/health_predictions_model3_{timestamp}.csv'
        df[['Time (h)', 'Health_State_Predicted', 'Health_State_Label'] + top_features_health].to_csv(report_path, index=False)
        
        return JSONResponse(content={
            "message": f"Predicted Health State for {len(df)} rows",
            "predictions": df[['Time (h)', 'Health_State_Predicted', 'Health_State_Label'] + top_features_health].to_dict(orient='records'),
            "report_path": report_path,
            "summary": summary
        })
    except Exception as e:
        logger.error(f"Error in predict_health_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# JSON input endpoint for Health State prediction
@app.post("/predict_health/json")
async def predict_health_json(data: FuelCellData):
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
        
        logger.debug(f"JSON input - Data: {df.iloc[0].to_dict()}")
        missing_features = [f for f in top_features_health if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        
        X_scaled, df = preprocess_data(df, top_features_health, scaler_health)
        df.loc[:, 'Health_State_Predicted'] = rf_health_top_model.predict(X_scaled)
        df.loc[:, 'Health_State_Label'] = df['Health_State_Predicted'].map({0: 'Critical', 1: 'Degraded', 2: 'Healthy'})
        result = df[['Time (h)', 'Health_State_Predicted', 'Health_State_Label'] + top_features_health].iloc[0].to_dict()
        
        return JSONResponse(content={
            "result": result,
            "thresholds": {
                "Critical": f"< {critical_threshold:.3f} V",
                "Degraded": f"{critical_threshold:.3f}–{degraded_threshold:.3f} V",
                "Healthy": f"≥ {degraded_threshold:.3f} V"
            }
        })
    except Exception as e:
        logger.error(f"Error in predict_health_json: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))