# Fuel Cell Digital Twin: Model 2 - Voltage Prediction (XGBoost)

## Overview
**This is a voltage prediction model designed to predict the total stack voltage (`Utot (V)`) of a fuel cell using the XGBoost algorithm. It processes data from `FC_Ageing.csv` (143,862 rows) to predict `Utot (V)` based on operational parameters, excluding `J (A/cm²)`, `I (A)`, and `U1 (V)` to `U5 (V)` to avoid data leakage. The model is served via a FastAPI application, deployed using Docker, and tested with Postman. It includes performance metrics (RMSE, MAE, R²) and visualizations (correlation heatmap, feature importance, actual vs. predicted, residual plot) to evaluate prediction accuracy.

## Directory Structure
```
Voltage_Prediction_Digital_Twin
├── app
│   ├── main.py                # FastAPI application
│   ├── requirements.txt       # Python dependencies
├── data
│   ├── FC_Ageing.csv         # Dataset
├── models
│   ├── xgb_utot_model2.pkl   # XGBoost model (all features)
│   ├── xgb_utot_top_model2.pkl # XGBoost model (top features)
│   ├── scaler_model2.pkl      # StandardScaler
│   ├── top_features_model2.pkl # Selected features
├── outputs
│   ├── correlation_matrix.png # Correlation heatmap
│   ├── feature_importance.png # Feature importance plot
│   ├── actual_vs_predicted.png # Actual vs. predicted plot
│   ├── residual_plot.png      # Residual plot
│   ├── metrics.txt            # Model metrics
│   ├── top_metrics.txt        # Top features metrics
│   ├── utot_predictions_model2_<timestamp>.csv # Prediction outputs
├── scripts
│   ├── train_model2.py       # Training script
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
```

## Features
- **Input Features**: `Time (h)`, `TinH2 (°C)`, `ToutH2 (°C)`, `TinAIR (°C)`, `ToutAIR (°C)`, `TinWAT (°C)`, `ToutWAT (°C)`, `PinAIR (mbara)`, `PoutAIR (mbara)`, `PoutH2 (mbara)`, `PinH2 (mbara)`, `DinH2 (l/mn)`, `DoutH2 (l/mn)`, `DinAIR (l/mn)`, `DoutAIR (l/mn)`, `DWAT (l/mn)`, `HrAIRFC (%)`.
- **Target**: `Utot (V)` (continuous, range: [3.093, 3.365] V).
- **Algorithm**: XGBoost Regressor for accurate voltage prediction.
- **Metrics**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R² Score.
- **Visualizations**: Correlation heatmap, feature importance, actual vs. predicted plot, residual plot.

## Setup and Installation
1. **Navigate to Directory**:
   ```powershell
   cd D:\FC-final\Voltage_Prediction_Digital_Twin
   ```
2. **Create and Activate Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install Dependencies**:
   Ensure `app/requirements.txt` contains:
   ```
   fastapi
   uvicorn
   pandas
   numpy
   scikit-learn
   xgboost
   joblib
   matplotlib
   seaborn
   openpyxl
   python-multipart
   ```
   Install:
   ```powershell
   pip install -r app\requirements.txt
   ```

## Training
- **Script**: `scripts/train_model2.py`
- **Steps**:
  1. Ensure `data/FC_Ageing.csv` exists with `Utot (V)` and required features.
  2. Run:
     ```powershell
     python scripts\train_model2.py
     ```
  3. Outputs:
     - Models: `models/xgb_utot_model2.pkl`, `models/xgb_utot_top_model2.pkl`
     - Scaler: `models/scaler_model2.pkl`
     - Features: `models/top_features_model2.pkl`
     - Metrics: `outputs/metrics.txt`, `outputs/top_metrics.txt`
     - Visualizations: `outputs/correlation_matrix.png`, `feature_importance.png`, `actual_vs_predicted.png`, `residual_plot.png`
  4. Expected Metrics:
     - Test RMSE: ~0.02–0.05 V
     - Test MAE: ~0.01–0.03 V
     - Test R²: ~0.7–0.85 (lower than with `J` and `I` due to no leakage)

## Deployment
1. **Build Docker**:
   ```powershell
   docker-compose build
   ```
2. **Run**:
   ```powershell
   docker-compose up -d
   ```
3. **Check Logs**:
   ```powershell
   docker logs fuel_cell_api
   ```
   - Expect: `Application startup complete`
   - Troubleshoot: Check for `KeyError: 'Utot'` or syntax errors.

## API Endpoints
- **GET /health**: Checks API status.
  - Response: `{"status": "API is running"}`
- **POST /predict_utot/file**: Predicts `Utot (V)` from uploaded `FC_Ageing.csv` or `.xlsx`.
  - Input: File upload (`form-data`, key=`file`).
  - Output: JSON with predictions, summary, and report path.
- **POST /predict_utot/json**: Predicts `Utot (V)` for a single JSON data point.
  - Input: JSON with operational features (e.g., `Time_h`, `TinWAT`).
  - Output: JSON with predicted `Utot (V)` and input features.

## Testing in Postman


### 1. GET /health
- **Setup**:
  - Method: GET
  - URL: `http://localhost:8000/health`
  - Click **Send**
- **Expected**:
  - Status: 200 OK
  - Body: `{"status": "API is running"}`
- **Notes**:
  - If connection error, check `docker ps` and logs:
    ```powershell
    docker logs fuel_cell_api
    ```

### 2. POST /predict_utot/file
- **Setup**:
  - Method: POST
  - URL: `http://localhost:8000/predict_utot/file`
  - Body: `form-data`, Key=`file`, Type=`File`, Value=`D:\FC-final\Voltage_Prediction_Digital_Twin\data\FC_Ageing.csv`
  - Click **Send**
- **Expected** (for 143,862 rows):
  ```json
  {
    "message": "Predicted Utot for 143862 rows",
    "predictions": [
      {
        "Time (h)": 0.0,
        "Utot_Predicted": 3.317,
        "TinWAT (°C)": 53.806,
        "ToutWAT (°C)": 55.502,
        "PinH2 (mbara)": 1292.489,
        ...
      },
      ...
    ],
    "report_path": "outputs/utot_predictions_model2_20250701_183000.csv",
    "summary": {
      "total_rows": 143862,
      "mean_utot_predicted": 3.27,
      "utot_range": [3.09, 3.36]
    }
  }
  ```
- **Generated Files** (in `outputs/`):
  - `utot_predictions_model2_<timestamp>.csv`
- **Notes**:
  - Check logs for dropped columns or preprocessing errors.

### 3. POST /predict_utot/json
- **Setup**:
  - Method: POST
  - URL: `http://localhost:8000/predict_utot/json`
  - Body: `raw`, `JSON`:
    ```json
    {
      "Time_h": 0.0,
      "TinH2": 25.93,
      "ToutH2": 38.889,
      "TinAIR": 41.918,
      "ToutAIR": 51.503,
      "TinWAT": 53.806,
      "ToutWAT": 55.502,
      "PinAIR": 1302.204,
      "PoutAIR": 1269.602,
      "PoutH2": 1305.319,
      "PinH2": 1292.489,
      "DinH2": 4.799,
      "DoutH2": 2.115,
      "DinAIR": 23.038,
      "DoutAIR": 21.331,
      "DWAT": 2.02,
      "HrAIRFC": 48.904
    }
    ```
  - Click **Send**
- **Expected**:
  - Status: 200 OK
  - Body:
    ```json
    {
      "result": {
        "Time (h)": 0.0,
        "Utot_Predicted": 3.317,
        "TinWAT (°C)": 53.806,
        "ToutWAT (°C)": 55.502,
        "PinH2 (mbara)": 1292.489,
        ...
      }
    }
