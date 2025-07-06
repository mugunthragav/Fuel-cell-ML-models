# Fuel Cell Prediction Dashboard

This project provides a FastAPI backend and a Dash UI frontend for interacting with five machine learning models for fuel cell analysis. The models include:

1. **Anomaly Detection Stack Voltage**: Detects anomalies in stack voltage data using DBSCAN clustering.
2. **Health State Classification**: Classifies fuel cell health (Critical, Degraded, Healthy) using a Random Forest model.
3. **Time Series Anomaly Detection**: Identifies anomalies in time-series current and voltage data using a transformer autoencoder and Isolation Forest.
4. **Transient Current Detection**: Predicts if a transient current event is Normal or Faulty using a Random Forest classifier.
5. **Voltage Prediction Digital Twin**: Predicts total stack voltage (Utot) using an XGBoost model.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Operating System**: Windows (tested), Linux, or macOS.
- **Project Directory**: `Fuel-Cell-Predictive-Analytics-ML-Models\combined_api` (adjust paths for your system).
- **Dependencies**: Listed in `requirements.txt` (created below).
- **Ports**: Ensure ports `8000` (FastAPI) and `8050` (Dash) are free.

## Setup Instructions

### 1. Clone or Navigate to Project Directory

If the project is in a repository, clone it. Otherwise, navigate to the project directory:

```bash
Clone the repository 
cd Fuel-Cell-Predictive-Analytics-ML-Models\combined_api
```

### 2. Create and Activate Virtual Environment

Create a virtual environment to isolate dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows**:
  ```bash
  .\venv\Scripts\activate
  ```
- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install required Python packages:

```bash
pip install -r app/requirements.txt
```

```


  ```

### 4. Verify Model Files

Ensure the following model files exist in their respective directories:
- **Anomaly Detection Stack Voltage**: `\Anomaly_Detection_Stack_Voltage\models\model.pkl`
- **Health State Classification**: `\Health_State_Classification\models\model.pkl`
- **Time Series Anomaly Detection**: `\Time_Series_Anomaly_Detection\models\model.pkl`
- **Transient Current Detection**: `\Transient_Current_Detection\models\model.pkl`, `feature_names.pkl`
- **Voltage Prediction Digital Twin**: `\Voltage_Prediction_Digital_Twin\models\model.pkl`, `top_features_model2.pkl`

If any files are missing, contact the project maintainer or retrain the models.

### 5. Run FastAPI Server

Start the FastAPI server to handle API requests:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
```

- **URL**: `http://localhost:8000`
- **Verify**: Open `http://localhost:8000` in a browser; expect `{"message": "Fuel Cell Prediction API"}`.
- **Logs**: Check for `INFO: Application startup complete`.

### 6. Run Dash UI

In a new terminal, navigate to the project directory and activate the virtual environment again:

```bash
cd D:\FC-final\combined_api
.\venv\Scripts\activate  # Windows
# or source venv/bin/activate  # Linux/macOS
```

Run the Dash UI:

```bash
python app.py
```

- **URL**: `http://localhost:8050`
- **Verify**: Open `http://localhost:8050` in a browser; expect a dropdown with five models and input forms.

### 7. Test the Application

#### Model 1: Anomaly Detection Stack Voltage
- **Input**: Upload a CSV/XLSX file with columns: `Date_Time`, `Load`, `Loaddensity`, `H2Flow`, `TstackIn`, `TstackOut`, `CoolantFlow`, `AnodePressureDiff`, `CathodePressureDiff`, `RH_An`, `RH_Cat`, `Cell1` to `Cell96`.
- **Example CSV**:
  ```python
  import pandas as pd
  data = {
      'Date_Time': ['2025-07-03 17:30:00', '2025-07-03 17:31:00'],
      'Load': [100.0, 101.0],
      'Loaddensity': [0.5, 0.51],
      'H2Flow': [1.0, 1.1],
      'TstackIn': [60.0, 60.5],
      'TstackOut': [65.0, 65.5],
      'CoolantFlow': [2.0, 2.1],
      'AnodePressureDiff': [0.1, 0.11],
      'CathodePressureDiff': [0.1, 0.12],
      'RH_An': [50.0, 51.0],
      'RH_Cat': [50.0, 50.5]
  }
  for i in range(1, 97):
      data[f'Cell{i}'] = [3.5, 3.4]
  df = pd.DataFrame(data)
  df.to_csv('test_model1.csv', index=False)
  ```
- **Action**: Select model, upload file, click Submit (rocket icon).
- **Output**: Number of anomalies and PCA plot (or error if PCA data is missing).

#### Model 2: Health State Classification
- **JSON Input**: Enter 17 fields (e.g., `Time (hours): 1.0`, `Hydrogen Inlet Temperature (°C): 60.0`).
- **File Input**: Upload CSV/XLSX with matching columns.
- **Action**: Select model, enter JSON or upload file, click Submit.
- **Output**: Health state (Critical/Degraded/Healthy).

#### Model 3: Time Series Anomaly Detection
- **Input**: Upload CSV with columns: `time`, `current`, `voltage`.
- **Example CSV**:
  ```python
  import pandas as pd
  data = {
      'time': ['2025-07-03 17:30:00', '2025-07-03 17:30:01'],
      'current': [10.0, 10.5],
      'voltage': [65.0, 64.8]
  }
  df = pd.DataFrame(data)
  df.to_csv('test_model3.csv', index=False)
  ```
- **Action**: Select model, upload file, click Submit.
- **Output**: Anomaly percentage and plot.

#### Model 4: Transient Current Detection
- **Input**: Enter 16 JSON fields (e.g., `Current (A): 10.0`, `Air Flow Rate (l/min): 300.0`).
- **Example JSON** (for Postman testing):
  ```json
  {
    "features": {
      "I": 10.0,
      "ARF": 300.0,
      "AIP1": 1.5,
      "AIP2": 1.55,
      "CAIF": 290.0,
      "CIP1": 1.4,
      "CS": 10.5,
      "COT2": 65.0,
      "CIT2": 60.0,
      "COT1": 64.0,
      "CIT1": 59.0,
      "WIP2": 1.2,
      "WIP1": 1.25,
      "WIF2": 2.0,
      "WIF1": 2.1,
      "WIT": 55.0
    }
  }
  ```
- **Action**: Select model, enter values, click Submit.
- **Output**: Prediction (Normal/Faulty).

#### Model 5: Voltage Prediction Digital Twin
- **JSON Input**: Enter 17 fields (e.g., `Time (hours): 1.0`).
- **File Input**: Upload CSV/XLSX with matching columns.
- **Example CSV**:
  ```python
  import pandas as pd
  data = {
      'Time (h)': [1.0, 2.0],
      'Hydrogen Inlet Temperature (°C)': [60.0, 61.0],
      'Air Outlet Temperature (°C)': [30.0, 31.0],
      'Air Inlet Pressure (mbara)': [1500.0, 1505.0],
      'Hydrogen Inlet Flow Rate (l/min)': [0.8, 0.9],
      'Water Flow Rate (l/min)': [2.0, 2.1]
  }
  df = pd.DataFrame(data)
  df.to_csv('test_model5.csv', index=False)
  ```
- **Action**: Select model, enter JSON or upload file, click Submit.
- **Output**: Predicted Utot value.

