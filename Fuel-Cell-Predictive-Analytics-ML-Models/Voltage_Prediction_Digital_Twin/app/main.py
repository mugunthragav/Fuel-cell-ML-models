from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
from pydantic import BaseModel
import datetime
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Fuel Cell Model 2 API", description="Digital Twin for Utot Prediction")

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

# Load pre-trained models and features
try:
    xgb_model = joblib.load('models/xgb_utot_model2.pkl')
    xgb_model_top = joblib.load('models/xgb_utot_top_model2.pkl')
    scaler = joblib.load('models/scaler_model2.pkl')
    top_features = joblib.load('models/top_features_model2.pkl')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model files not found in 'models/' directory")

# Define all features (excluding U1-U5, J, I; including Time (h))
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
def preprocess_data(df, features):
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Handle outliers
    for col in features:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df.loc[:, col] = df[col].clip(lower, upper)
    
    # Select features
    X = df[features].copy()
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, df

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "API is running"}

# File upload endpoint for Utot prediction
@app.post("/predict_utot/file")
async def predict_utot_file(file: UploadFile = File(...)):
    try:
        # Read file
        file_content = await file.read()
        file_like = BytesIO(file_content)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_like)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file_like, engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or XLSX.")
        
        logger.debug(f"Initial df columns: {df.columns.tolist()}")
        
        # Verify features
        missing_features = [f for f in top_features if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        
        # Preprocess
        X_scaled, df = preprocess_data(df, top_features)
        
        # Predict Utot
        df.loc[:, 'Utot_Predicted'] = xgb_model_top.predict(X_scaled)
        
        # Generate summary
        summary = {
            "total_rows": len(df),
            "mean_utot_predicted": float(df['Utot_Predicted'].mean()),
            "utot_range": [float(df['Utot_Predicted'].min()), float(df['Utot_Predicted'].max())]
        }
        
        # Save predictions
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'outputs/utot_predictions_model2_{timestamp}.csv'
        df[['Time (h)', 'Utot_Predicted'] + top_features].to_csv(report_path, index=False)
        
        return JSONResponse(content={
            "message": f"Predicted Utot for {len(df)} rows",
            "predictions": df[['Time (h)', 'Utot_Predicted'] + top_features].to_dict(orient='records'),
            "report_path": report_path,
            "summary": summary
        })
    except Exception as e:
        logger.error(f"Error in predict_utot_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# JSON input endpoint for Utot prediction
@app.post("/predict_utot/json")
async def predict_utot_json(data: FuelCellData):
    try:
        # Convert JSON to DataFrame
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
        
        # Preprocess
        X_scaled, df = preprocess_data(df, top_features)
        
        # Predict Utot
        df.loc[:, 'Utot_Predicted'] = xgb_model_top.predict(X_scaled)
        
        # Generate result
        result = df[['Time (h)', 'Utot_Predicted'] + top_features].iloc[0].to_dict()
        
        return JSONResponse(content={"result": result})
    except Exception as e:
        logger.error(f"Error in predict_utot_json: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))