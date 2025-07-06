# Fuel Cell Digital Twin: Model 1 - Anomaly Detection (DBSCAN)

## Overview
This is an anomaly detection model for identifying irregular voltage behavior in a fuel cell stack with 96 cells (`Cell1` to `Cell96`) using the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. The model processes data from `data3040b.xlsx` (or `data3040b.csv`) to detect anomalies based on operational features and cell voltages, enabling early identification of potential issues in fuel cell performance. The model is served via a FastAPI application, deployed using Docker, and tested with Postman.

## Directory Structure
```
FC-final\Anomaly_Detection_Stack_Voltage
├── app
│   ├── main.py                # FastAPI application for Model 1
│   ├── requirements.txt       # Python dependencies
├── data
│   ├── data3040b.xlsx        # Dataset (Excel)
│   ├── data3040b.csv         # Dataset (CSV, optional)
├── models
│   ├── dbscan_model1.pkl     # DBSCAN model
│   ├── scaler_model1.pkl     # StandardScaler
├── outputs
│   ├── realtime_anomalies_dbscan_model1_<timestamp>.csv # Anomaly report
│   ├── realtime_anomalies_cell_Cell1_<timestamp>.csv    # Per-cell anomaly reports
│   ├── ...                                             # Up to Cell96
│   ├── realtime_cluster_anomaly_plot_<timestamp>.png    # Anomaly visualization
│   ├── metrics_model1.txt                              # Model metrics
├── scripts
│   ├── anomaly_detection_dbscan.py # Training script
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose configuration
```

## Features
- **Input Features**: `Load`, `Loaddensity`, `H2Flow`, `TstackIn`, `TstackOut`, `CoolantFlow`, `AnodePressureDiff`, `CathodePressureDiff`, `RH_An`, `RH_Cat`, `Cell1` to `Cell96`.
- **Target**: Binary classification (Normal: 0, Anomaly: 1) for each cell’s voltage.
- **Algorithm**: DBSCAN, with `eps` set dynamically based on the 30th percentile of k-distance (k=4) to identify outliers.
- **Metrics**: Total anomalies, anomaly percentage, most affected cell, min cell voltage range, key operating conditions (e.g., avg `Load`, `H2Flow`).
- **Outputs**: Anomaly reports (`CSV`), per-cell anomaly reports, and visualization (`PNG`).

## Setup and Installation
1. **Navigate to Directory**:
   ```powershell
   cd D:\FC-final\Anomaly_Detection_Stack_Voltage
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
   openpyxl
   python-multipart
   ```
   Install:
   ```powershell
   pip install -r app\requirements.txt
   ```

## Training
- **Script**: `scripts/anomaly_detection_dbscan.py`
- **Steps**:
  1. Ensure `data/data3040b.xlsx` or `data/data3040b.csv` exists.
  2. Run:
     ```powershell
     python scripts\anomaly_detection_dbscan.py
     ```
  3. Outputs:
     - Model: `models/dbscan_model1.pkl`
     - Scaler: `models/scaler_model1.pkl`
     - Metrics: `outputs/metrics_model1.txt`
     - Visualization: `outputs/realtime_cluster_anomaly_plot_<timestamp>.png`
  4. Adjust `eps` (line ~62) if anomaly count is high (~80%):
     ```python
     eps = k_distances[int(len(k_distances) * 0.3)]
     ```

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
   - Troubleshoot: Check for `IndentationError` or missing model files.

## API Endpoints
- **GET /health**: Checks API status.
  - Response: `{"status": "API is running"}`
- **POST /detect_anomalies/file**: Detects anomalies from uploaded `data3040b.xlsx` or `.csv`.
  - Input: File upload (`form-data`, key=`file`).
  - Output: JSON with anomalies, summary, and paths to reports/plots.
- **POST /detect_anomalies/json**: Detects anomalies for a single JSON data point with 96 cell voltages.
  - Input: JSON with `Date_Time`, operational features, and `Cell_Voltages` (array of 96 floats).
  - Output: JSON with anomaly status and details.
- **GET /detect_anomalies/cell/{cell_number}**: Retrieves anomalies for a specific cell (1–96).
  - Input: URL parameter `cell_number` (e.g., 35).
  - Output: JSON with anomalies for the specified cell.

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

### 2. POST /detect_anomalies/file
- **Setup**:
  - Method: POST
  - URL: `http://localhost:8000/detect_anomalies/file`
  - Body: `form-data`, Key=`file`, Type=`File`, Value=`D:\FC-final\model-1\data\data3040b.xlsx`
  - Click **Send**
- **Expected** (for ~9,000 rows, ~5,000 anomalies):
  ```json
  {
    "message": "Detected 5000 anomalies",
    "anomalies": [
      {
        "Date_Time": "2022-10-10 12:18:29",
        "Min_Cell_Voltage": -0.011,
        "Min_Cell_Index": "Cell35",
        "Load": 0.5107061,
        "Loaddensity": 2.55353,
        "H2Flow": 3.712091,
        "TstackIn": 59.0,
        "TstackOut": 56.8,
        "CoolantFlow": 3.744213,
        "AnodePressureDiff": 78.125,
        "CathodePressureDiff": 10.30816,
        "RH_An": 4.007779,
        "RH_Cat": 4.007779
      },
      ...
    ],
    "plot_path": "outputs/realtime_cluster_anomaly_plot_20250701_183000.png",
    "report_path": "outputs/realtime_anomalies_dbscan_model1_20250701_183000.csv",
    "summary": {
      "total_rows": 9000,
      "total_anomalies": 5000,
      "anomaly_percentage": 55.56,
      "most_affected_cell": "Cell35",
      "min_cell_voltage_range": [-0.014, 0.409],
      "key_operating_conditions": {
        "avg_load": 0.51,
        "avg_h2flow": 3.71,
        "avg_tstackin": 59.0
      }
    }
  }
  ```
- **Generated Files** (in `outputs/`):
  - `realtime_anomalies_dbscan_model1_<timestamp>.csv`
  - `realtime_anomalies_cell_Cell1_<timestamp>.csv` to `..._Cell96_<timestamp>.csv`
  - `realtime_cluster_anomaly_plot_<timestamp>.png`
- **Notes**:
  - Check logs for `Anomaly report - Date_Time type before serialization: object`.
  - If anomaly count is high, adjust `eps` and retrain.

### 3. POST /detect_anomalies/json
- **Setup**:
  - Method: POST
  - URL: `http://localhost:8000/detect_anomalies/json`
  - Body: `raw`, `JSON`:
    ```json
    {
      "Date_Time": "2022-10-10 12:18:29",
      "Load": 0.5107061,
      "Loaddensity": 2.55353,
      "H2Flow": 3.712091,
      "TstackIn": 59.0,
      "TstackOut": 56.8,
      "CoolantFlow": 3.744213,
      "AnodePressureDiff": 78.125,
      "CathodePressureDiff": 10.30816,
      "RH_An": 4.007779,
      "RH_Cat": 4.007779,
      "Cell_Voltages": [
        -0.011, 0.006, 0.028, 0.005, 0.01, 0.012, 0.015, 0.02, 0.018, 0.017,
        0.016, 0.014, 0.013, 0.011, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004,
        0.003, 0.002, 0.001, 0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006,
        0.007, 0.008, 0.009, 0.01, -0.013, 0.012, 0.013, 0.014, 0.015, 0.016,
        0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026,
        0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036,
        0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046,
        0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056,
        0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066,
        0.067, 0.068, 0.069, 0.07
      ]
    }
    ```
  - Click **Send**
- **Expected**:
  - Status: 200 OK
  - Body:
    ```json
    {
      "result": {
        "Date_Time": "2022-10-10 12:18:29",
        "Min_Cell_Voltage": -0.011,
        "Min_Cell_Index": "Cell35",
        "Load": 0.5107061,
        "Loaddensity": 2.55353,
        "H2Flow": 3.712091,
        "TstackIn": 59.0,
        "TstackOut": 56.8,
        "CoolantFlow": 3.744213,
        "AnodePressureDiff": 78.125,
        "CathodePressureDiff": 10.30816,
        "RH_An": 4.007779,
        "RH_Cat": 4.007779,
        "Is_Anomaly": true
      }
    }
    ```
- **Notes**:
  - Ensure `Cell_Voltages` has exactly 96 float values.
  - If 422 error, check logs for `JSON input - Cell_Voltages length` or `JSON input - Data`.

### 4. GET /detect_anomalies/cell/{cell_number}
- **Setup**:
  - Method: GET
  - URL: `http://localhost:8000/detect_anomalies/cell/35`
  - Click **Send**
- **Expected**:
  - Status: 200 OK
  - Body:
    ```json
    {
      "message": "Detected 3000 anomalies for Cell35",
      "anomalies": [
        {
          "Date_Time": "2022-10-10 12:18:29",
          "Min_Cell_Voltage": -0.011,
          "Min_Cell_Index": "Cell35"
        },
        ...
      ]
    }
    ```
- **Notes**:
  - Requires `outputs/realtime_anomalies_dbscan_model1.csv`. If missing, expect:
    ```json
    {"detail": "Anomaly report not found. Run training script first."}
    ```

