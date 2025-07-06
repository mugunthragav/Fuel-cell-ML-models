# Fuel Cell Digital Twin: Model 3 - Health State Classification (Random Forest)

## Overview
This is a health state classification model designed to classify the health state of a fuel cell stack into **Healthy**, **Degraded**, or **Critical** using a Random Forest Classifier. It processes data from `FC_Ageing.csv` to classify health states based on operational features, excluding `U1 (V)` to `U5 (V)`, `J (A/cm²)`, and `I (A)` to avoid data leakage. The model uses dynamic thresholds based on `Utot (V)` quantiles (5th and 50th percentiles) to handle extreme voltage values (< 3.0 V or > 5.0 V/10.0 V). It is served via a FastAPI application, deployed using Docker, and tested with Postman. Performance metrics (Accuracy, Precision, Recall, F1-score) and visualizations (correlation matrix, feature importance, confusion matrix, health state time-series) are provided to evaluate the model.

## Directory Structure
```
FC-final\Health_State_Classification
├── app
│   ├── main.py                # FastAPI application
│   ├── requirements.txt       # Python dependencies
├── data
│   ├── FC_Ageing.csv         # Dataset
├── models
│   ├── rf_health_model3.pkl   # Random Forest Classifier (all features)
│   ├── rf_health_top_model3.pkl # Random Forest Classifier (top features)
│   ├── scaler_model3.pkl      # StandardScaler
│   ├── top_features_model3.pkl # Selected features
├── outputs
│   ├── correlation_matrix_model3.png # Correlation heatmap
│   ├── feature_importance_model3.png # Feature importance plot
│   ├── confusion_matrix_model3.png   # Confusion matrix
│   ├── health_states_model3.png      # Health state time-series
│   ├── metrics_model3.txt            # Model metrics
│   ├── top_metrics_model3.txt        # Top features metrics
│   ├── health_predictions_model3_<timestamp>.csv # Prediction outputs
├── scripts
│   ├── train_model3.py       # Training script
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
```

## Features
- **Input Features**: `Time (h)`, `TinH2 (°C)`, `ToutH2 (°C)`, `TinAIR (°C)`, `ToutAIR (°C)`, `TinWAT (°C)`, `ToutWAT (°C)`, `PinAIR (mbara)`, `PoutAIR (mbara)`, `PoutH2 (mbara)`, `PinH2 (mbara)`, `DinH2 (l/mn)`, `DoutH2 (l/mn)`, `DinAIR (l/mn)`, `DoutAIR (l/mn)`, `DWAT (l/mn)`, `HrAIRFC (%)`.
- **Target**: Health state (`Critical`: 0, `Degraded`: 1, `Healthy`: 2) based on `Utot (V)` thresholds.
- **Thresholds**: Dynamic, based on `Utot (V)`:
  - Critical: < 5th percentile (~3.1 V)
  - Degraded: 5th percentile to 50th percentile (~3.1–3.2 V)
  - Healthy: ≥ 50th percentile (~3.2 V)
- **Algorithm**: Random Forest Classifier with `class_weight='balanced'` to handle imbalanced classes.
- **Data Augmentation**: Synthetic data for `Utot (V)` < 3.0 V (2.0–3.0 V) and > 5.0 V (5.0–10.0 V) to handle extreme values.
- **Metrics**: Accuracy, Precision, Recall, F1-score (weighted), Confusion Matrix.
- **Visualizations**: Correlation matrix, feature importance, confusion matrix, health state time-series.

## Setup and Installation
1. **Navigate to Directory**:
   ```powershell
   cd D:\FC-final\Health_State_Classification
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
- **Script**: `scripts/train_model3.py`
- **Steps**:
  1. Ensure `data/FC_Ageing.csv` exists with `Utot (V)` and required features.
  2. Run:
     ```powershell
     python scripts\train_model3.py
     ```
  3. Outputs:
     - Models: `models/rf_health_model3.pkl`, `models/rf_health_top_model3.pkl`
     - Scaler: `models/scaler_model3.pkl`
     - Features: `models/top_features_model3.pkl`
     - Metrics: `outputs/metrics_model3.txt`, `outputs/top_metrics_model3.txt`
     - Visualizations: `outputs/correlation_matrix_model3.png`, `outputs/feature_importance_model3.png`, `outputs/confusion_matrix_model3.png`, `outputs/health_states_model3.png`
  4. Expected Metrics:
     - Test Accuracy: > 0.85
     - Test Precision/Recall/F1: > 0.8 (weighted)
     - Health State Distribution: ~5% Critical (including synthetic), ~45% Degraded, ~50% Healthy

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
   - Troubleshoot: Check for `KeyError` or syntax errors.

## API Endpoints
- **GET /health**: Checks API status.
  - Response: `{"status": "API is running"}`
- **POST /predict_health/file**: Classifies health states from uploaded `FC_Ageing.csv` or `.xlsx`.
  - Input: File upload (`form-data`, key=`file`).
  - Output: JSON with predictions, thresholds, summary, and report path.
- **POST /predict_health/json**: Classifies health state for a single JSON data point.
  - Input: JSON with operational features (e.g., `Time_h`, `TinWAT`).
  - Output: JSON with predicted health state and thresholds.

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

### 2. POST /predict_health/file
- **Setup**:
  - Method: POST
  - URL: `http://localhost:8000/predict_health/file`
  - Body: `form-data`, Key=`file`, Type=`File`, Value=`D:\FC-final\Health_State_Classification\data\FC_Ageing.csv`
  - Click **Send**
- **Expected** (for ~151,055 rows):
  ```json
  {
    "message": "Predicted Health State for 151055 rows",
    "predictions": [
      {
        "Time (h)": 0.0,
        "Health_State_Predicted": 2,
        "Health_State_Label": "Healthy",
        "TinWAT (°C)": 53.806,
        "ToutWAT (°C)": 55.502,
        "PinH2 (mbara)": 1292.489,
        ...
      },
      ...
    ],
    "report_path": "outputs/health_predictions_model3_20250701_183000.csv",
    "summary": {
      "total_rows": 151055,
      "health_state_counts": {
        "Healthy": 75000,
        "Degraded": 68000,
        "Critical": 8055
      },
      "thresholds": {
        "Critical": "< 3.100 V",
        "Degraded": "3.100–3.200 V",
        "Healthy": "≥ 3.200 V"
      }
    }
  }
  ```
- **Generated Files** (in `outputs/`):
  - `health_predictions_model3_<timestamp>.csv`
- **Notes**:
  - Check logs for preprocessing errors or missing features.

### 3. POST /predict_health/json
- **Setup**:
  - Method: POST
  - URL: `http://localhost:8000/predict_health/json`
  - Body: `raw`, `JSON`:
    ```json
    {
      "Time_h": 1000.0,
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
- **Expected** (Critical due to high `Time_h`):
  ```json
  {
    "result": {
      "Time (h)": 1000.0,
      "Health_State_Predicted": 0,
      "Health_State_Label": "Critical",
      "TinWAT (°C)": 53.806,
      "ToutWAT (°C)": 55.502,
      "PinH2 (mbara)": 1292.489,
      ...
    },
    "thresholds": {
      "Critical": "< 3.100 V",
      "Degraded": "3.100–3.200 V",
      "Healthy": "≥ 3.200 V"
    }
  }
  ```
- **Notes**:
  - Ensure JSON matches `FuelCellData` schema in `main.py`.
  - If 422 error, check logs for `Error in predict_health_json`.

## How Model 3 Works
1. **Objective**: Classify fuel cell health states (Healthy, Degraded, Critical) based on operational features, using `Utot (V)` to define labels during training.
2. **Dynamic Thresholds**:
   - Uses quantiles of `Utot (V)` from `FC_Ageing.csv` (range: [3.093, 3.365] V):
     - Critical: < 5th percentile (~3.1 V)
     - Degraded: 5th percentile to 50th percentile (~3.1–3.2 V)
     - Healthy: ≥ 50th percentile (~3.2 V)
   - Thresholds stored in `outputs/metrics_model3.txt`.
3. **Data Augmentation**: Adds synthetic data (5% with `Utot (V)` = 2.0–3.0 V, 5% with 5.0–10.0 V) to handle extreme voltages.
4. **Training** (`train_model3.py`):
   - Loads `FC_Ageing.csv`, fills missing values, drops low-variance columns.
   - Defines health states using `pd.cut` with dynamic thresholds.
   - Trains two Random Forest Classifiers: full features (`rf_health_model3.pkl`) and top features (`rf_health_top_model3.pkl`).
   - Saves models, scaler, features, metrics, and visualizations.
5. **Prediction** (`main.py`):
   - Preprocesses input (handles duplicates, missing values, outliers).
   - Uses `rf_health_top_model3.pkl` and `scaler_model3.pkl` to predict health states.
   - Returns predictions, thresholds, and summary.

## Expected Performance
- **Metrics** (in `outputs/metrics_model3.txt`, `outputs/top_metrics_model3.txt`):
  - Test Accuracy: > 0.85
  - Test Precision/Recall/F1: > 0.8 (weighted)
  - Health State Distribution: ~5% Critical, ~45% Degraded, ~50% Healthy
- **Visualizations** (in `outputs/`):
  - `correlation_matrix_model3.png`: Shows feature correlations.
  - `feature_importance_model3.png`: Highlights `Time (h)`, `TinWAT (°C)`, `PinH2 (mbara)`.
  - `confusion_matrix_model3.png`: Strong diagonal, some Critical predictions.
  - `health_states_model3.png`: Health state transitions over `Time (h)`.

