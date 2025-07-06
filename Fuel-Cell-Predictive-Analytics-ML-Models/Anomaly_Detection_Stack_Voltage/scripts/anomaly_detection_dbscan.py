import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import joblib
import matplotlib.pyplot as plt
import os
import warnings

# Suppress SettingWithCopyWarning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Set environment variable to silence joblib warning
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Adjust based on your CPU cores

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load dataset
data_path = 'data/data3040b.xlsx'
data = pd.read_excel(data_path)

# Step 1: Preprocessing
# Remove duplicates
data = data.drop_duplicates()

# Select features
features = ['Load', 'Loaddensity', 'H2Flow', 'TstackIn', 'TstackOut', 
            'CoolantFlow', 'AnodePressureDiff', 'CathodePressureDiff', 
            'RH_An', 'RH_Cat'] + [f'Cell{i}' for i in range(1, 97)]

# Add feature: standard deviation of cell voltages
data['Cell_Voltage_Std'] = data[[f'Cell{i}' for i in range(1, 97)]].std(axis=1)
features.append('Cell_Voltage_Std')

# Verify features exist in the dataset
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    raise ValueError(f"Missing features in dataset: {missing_features}")

X = data[features].copy()

# Handle zeros/missing values using .loc
for col in ['Load', 'Loaddensity', 'H2Flow', 'TstackIn', 'TstackOut', 
            'CoolantFlow', 'AnodePressureDiff', 'CathodePressureDiff', 
            'RH_An', 'RH_Cat', 'Cell_Voltage_Std']:
    X.loc[:, col] = X[col].replace(0, X[col].mean())

for col in [f'Cell{i}' for i in range(1, 97)]:
    X.loc[:, col] = X[col].replace(0, X[col].median())

# Handle any remaining NaN values
X = X.fillna(X.mean())

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Select optimal eps using elbow method
min_samples = max(5, int(np.log(len(data))))  # Dynamic min_samples
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, _ = neighbors_fit.kneighbors(X_scaled)
k_distances = np.sort(distances[:, min_samples-1])

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(len(k_distances)), k_distances)
plt.xlabel('Sorted Points')
plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
plt.title('Elbow Plot for DBSCAN eps Selection')
plt.savefig('outputs/elbow_plot.png')
plt.close()

# Choose eps (increased to 20th percentile to reduce anomalies)
eps = k_distances[int(len(k_distances) * 0.2)]

# Step 3: Train DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
cluster_labels = dbscan.fit_predict(X_scaled)
data['Cluster'] = cluster_labels

# Step 4: Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

# Step 5: Enhanced cluster plot
plt.figure(figsize=(12, 8))
normal_points = data[data['Cluster'] != -1]
anomalies = data[data['Cluster'] == -1]
plt.scatter(normal_points['PCA1'], normal_points['PCA2'], c=normal_points['Cluster'], 
            cmap='viridis', s=50, alpha=0.6, label='Normal (Clustered)')
plt.scatter(anomalies['PCA1'], anomalies['PCA2'], c='red', s=100, marker='x', 
            label='Anomalies', linewidths=2)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.title('DBSCAN Clustering: Normal vs. Anomalous Points', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('outputs/cluster_anomaly_plot.png', dpi=300)
plt.close()

# Step 6: Generate anomaly report
data['Min_Cell_Voltage'] = data[[f'Cell{i}' for i in range(1, 97)]].min(axis=1)
data['Min_Cell_Index'] = data[[f'Cell{i}' for i in range(1, 97)]].idxmin(axis=1)

# Overall anomaly report
anomalies = data[data['Cluster'] == -1]
anomaly_report = anomalies[['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index', 
                           'Load', 'Loaddensity', 'H2Flow', 'TstackIn', 
                           'TstackOut', 'CoolantFlow', 'AnodePressureDiff', 
                           'CathodePressureDiff', 'RH_An', 'RH_Cat']]
anomaly_report.to_csv('outputs/anomalies_dbscan_model1.csv', index=False)

# Generate cell-specific anomaly reports
for cell in [f'Cell{i}' for i in range(1, 97)]:
    cell_anomalies = anomalies[anomalies['Min_Cell_Index'] == cell][['Date_Time', cell, 'Min_Cell_Voltage', 'Min_Cell_Index']]
    cell_anomalies.to_csv(f'outputs/anomalies_cell_{cell}.csv', index=False)

# Step 7: Generate descriptive summary
anomaly_summary = {
    "total_rows": len(data),
    "total_anomalies": len(anomalies),
    "anomaly_percentage": (len(anomalies) / len(data)) * 100,
    "most_affected_cell": anomalies['Min_Cell_Index'].value_counts().idxmax() if not anomalies.empty else "None",
    "min_cell_voltage_range": [anomalies['Min_Cell_Voltage'].min(), anomalies['Min_Cell_Voltage'].max()] if not anomalies.empty else [0, 0],
    "key_operating_conditions": {
        "avg_load": anomalies['Load'].mean() if not anomalies.empty else 0,
        "avg_h2flow": anomalies['H2Flow'].mean() if not anomalies.empty else 0,
        "avg_tstackin": anomalies['TstackIn'].mean() if not anomalies.empty else 0
    }
}
with open('outputs/anomaly_summary.txt', 'w') as f:
    f.write(f"Anomaly Detection Summary (DBSCAN Model)\n")
    f.write(f"Total Data Points: {anomaly_summary['total_rows']}\n")
    f.write(f"Total Anomalies Detected: {anomaly_summary['total_anomalies']}\n")
    f.write(f"Anomaly Percentage: {anomaly_summary['anomaly_percentage']:.2f}%\n")
    f.write(f"Most Affected Cell: {anomaly_summary['most_affected_cell']}\n")
    f.write(f"Min Cell Voltage Range: {anomaly_summary['min_cell_voltage_range']}\n")
    f.write(f"Average Load (Anomalies): {anomaly_summary['key_operating_conditions']['avg_load']:.2f}\n")
    f.write(f"Average H2 Flow (Anomalies): {anomaly_summary['key_operating_conditions']['avg_h2flow']:.2f}\n")
    f.write(f"Average Stack Inlet Temp (Anomalies): {anomaly_summary['key_operating_conditions']['avg_tstackin']:.2f}\n")

# Print anomaly summary
print(f"Detected {len(anomalies)} anomalies:")
print(anomaly_report)
print("\nAnomaly Summary:")
for key, value in anomaly_summary.items():
    print(f"{key}: {value}")

# Step 8: Save model, scaler, and PCA transformer
joblib.dump(dbscan, 'models/dbscan_model1.pkl')
joblib.dump(scaler, 'models/scaler_model1.pkl')
joblib.dump(pca, 'models/pca_model1.pkl')

# Step 9: Function to process new uploaded data
def process_uploaded_data(file_path):
    # Load new data
    new_data = pd.read_excel(file_path)
    new_data = new_data.drop_duplicates()
    
    # Preprocess
    new_data['Cell_Voltage_Std'] = new_data[[f'Cell{i}' for i in range(1, 97)]].std(axis=1)
    X_new = new_data[features].copy()
    for col in ['Load', 'Loaddensity', 'H2Flow', 'TstackIn', 'TstackOut', 
                'CoolantFlow', 'AnodePressureDiff', 'CathodePressureDiff', 
                'RH_An', 'RH_Cat', 'Cell_Voltage_Std']:
        X_new.loc[:, col] = X_new[col].replace(0, X_new[col].mean())
    for col in [f'Cell{i}' for i in range(1, 97)]:
        X_new.loc[:, col] = X_new[col].replace(0, X_new[col].median())
    X_new = X_new.fillna(X_new.mean())
    
    # Load saved scaler and model
    scaler = joblib.load('models/scaler_model1.pkl')
    dbscan = joblib.load('models/dbscan_model1.pkl')
    pca = joblib.load('models/pca_model1.pkl')
    
    # Normalize and predict
    X_new_scaled = scaler.transform(X_new)
    cluster_labels = dbscan.fit_predict(X_new_scaled)
    new_data['Cluster'] = cluster_labels
    
    # Visualize
    X_new_pca = pca.transform(X_new_scaled)
    new_data['PCA1'] = X_new_pca[:, 0]
    new_data['PCA2'] = X_new_pca[:, 1]
    plt.figure(figsize=(12, 8))
    normal_points = new_data[new_data['Cluster'] != -1]
    anomalies = new_data[new_data['Cluster'] == -1]
    plt.scatter(normal_points['PCA1'], normal_points['PCA2'], c=normal_points['Cluster'], 
                cmap='viridis', s=50, alpha=0.6, label='Normal (Clustered)')
    plt.scatter(anomalies['PCA1'], anomalies['PCA2'], c='red', s=100, marker='x', 
                label='Anomalies', linewidths=2)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.title('DBSCAN Clustering: New Data Anomalies', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('outputs/new_cluster_anomaly_plot.png', dpi=300)
    plt.close()
    
    # Generate anomaly report
    new_data['Min_Cell_Voltage'] = new_data[[f'Cell{i}' for i in range(1, 97)]].min(axis=1)
    new_data['Min_Cell_Index'] = new_data[[f'Cell{i}' for i in range(1, 97)]].idxmin(axis=1)
    new_anomaly_report = anomalies[['Date_Time', 'Min_Cell_Voltage', 'Min_Cell_Index', 
                                   'Load', 'Loaddensity', 'H2Flow', 'TstackIn', 
                                   'TstackOut', 'CoolantFlow', 'AnodePressureDiff', 
                                   'CathodePressureDiff', 'RH_An', 'RH_Cat']]
    new_anomaly_report.to_csv('outputs/new_anomalies_dbscan_model1.csv', index=False)
    
    # Generate cell-specific anomaly reports
    for cell in [f'Cell{i}' for i in range(1, 97)]:
        cell_anomalies = anomalies[anomalies['Min_Cell_Index'] == cell][['Date_Time', cell, 'Min_Cell_Voltage', 'Min_Cell_Index']]
        cell_anomalies.to_csv(f'outputs/new_anomalies_cell_{cell}.csv', index=False)
    
    print(f"Detected {len(anomalies)} anomalies in new data:")
    print(new_anomaly_report)
    
    return new_anomaly_report