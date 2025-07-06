import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load dataset
try:
    df = pd.read_csv('data/FC_Ageing.csv')
    logger.debug(f"Loaded dataset with shape: {df.shape}")
    logger.debug(f"Columns: {df.columns.tolist()}")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise

# Preprocessing
df = df.fillna(df.mean())
logger.debug("Filled missing values with column means")

low_variance_cols = [col for col in df.columns if df[col].std() < 0.01]
df = df.drop(columns=low_variance_cols)
logger.debug(f"Dropped low-variance columns: {low_variance_cols}")

# Features (exclude U1-U5, J, I; include Time (h))
features = [col for col in df.columns if col not in ['U1 (V)', 'U2 (V)', 'U3 (V)', 'U4 (V)', 'U5 (V)', 'J (A/cm²)', 'I (A)', 'Utot (V)']]
logger.debug(f"Selected features: {features}")

if 'Utot (V)' not in df.columns or 'Time (h)' not in df.columns:
    logger.error("Required columns 'Utot (V)' or 'Time (h)' not found")
    raise KeyError("Required columns 'Utot (V)' or 'Time (h)' not found")

# Dynamic thresholds based on Utot (V) quantiles
utot = df['Utot (V)']
critical_threshold = utot.quantile(0.05)  # 5th percentile
degraded_threshold = utot.quantile(0.50)  # Median
logger.debug(f"Dynamic thresholds - Critical: < {critical_threshold:.3f} V, Degraded: {critical_threshold:.3f}–{degraded_threshold:.3f} V, Healthy: ≥ {degraded_threshold:.3f} V")

# Augment dataset with synthetic extreme values
# Low Utot (< 3.0 V): Simulate severe degradation
synthetic_low = df.sample(frac=0.05, random_state=42).copy()
synthetic_low['Utot (V)'] = np.random.uniform(2.0, 3.0, size=len(synthetic_low))
synthetic_low['Time (h)'] = synthetic_low['Time (h)'] + 100  # Simulate ageing
# High Utot (> 5.0 V): Simulate overperformance
synthetic_high = df.sample(frac=0.05, random_state=43).copy()
synthetic_high['Utot (V)'] = np.random.uniform(5.0, 10.0, size=len(synthetic_high))
synthetic_high['Time (h)'] = synthetic_high['Time (h)'] - 50  # Simulate early operation
df_augmented = pd.concat([df, synthetic_low, synthetic_high], ignore_index=True)
logger.debug(f"Augmented dataset shape: {df_augmented.shape}")

# Define health states
df_augmented['Health_State'] = pd.cut(
    df_augmented['Utot (V)'],
    bins=[-float('inf'), critical_threshold, degraded_threshold, float('inf')],
    labels=['Critical', 'Degraded', 'Healthy'],
    right=False
)
df_augmented['Health_State_Code'] = df_augmented['Health_State'].cat.codes  # 0=Critical, 1=Degraded, 2=Healthy
logger.debug(f"Health state distribution: {df_augmented['Health_State'].value_counts().to_dict()}")

# Split features and target
X = df_augmented[features]
y = df_augmented['Health_State_Code']

for col in features:
    lower, upper = X[col].quantile([0.01, 0.99])
    X.loc[:, col] = X[col].clip(lower, upper)
logger.debug("Capped outliers at 1% and 99% percentiles")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logger.debug(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logger.debug("Scaled features using StandardScaler")

# Train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)
logger.debug("Trained Random Forest Classifier")

# Predictions
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

# Compute metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

logger.debug(f"Train - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
logger.debug(f"Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

# Save metrics
metrics = {
    "train_accuracy": train_accuracy,
    "train_precision": train_precision,
    "train_recall": train_recall,
    "train_f1": train_f1,
    "test_accuracy": test_accuracy,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_f1": test_f1,
    "critical_threshold": critical_threshold,
    "degraded_threshold": degraded_threshold
}
with open('outputs/metrics_model3.txt', 'w') as f:
    f.write(str(metrics))
logger.debug("Saved metrics to outputs/metrics_model3.txt")

# Correlation analysis
corr_matrix = df_augmented[features + ['Health_State_Code']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (Model 3: Health State)')
plt.savefig('outputs/correlation_matrix_model3.png')
plt.close()

# Feature importance
feat_importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feat_importance.plot(kind='bar')
plt.title('Random Forest Feature Importance (Model 3: Health State)')
plt.savefig('outputs/feature_importance_model3.png')
plt.close()
logger.debug(f"Feature importance: {feat_importance.to_dict()}")

# Select top features
top_features = feat_importance[feat_importance > 0.05].index.tolist()
logger.debug(f"Top features: {top_features}")

# Train with top features
if top_features:
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    X_train_top_scaled = scaler.fit_transform(X_train_top)
    X_test_top_scaled = scaler.transform(X_test_top)
    rf_model_top = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model_top.fit(X_train_top_scaled, y_train)
    logger.debug("Trained Random Forest with top features")
else:
    rf_model_top = rf_model
    top_features = features

# Predictions and metrics for top features
y_train_top_pred = rf_model_top.predict(X_train_top_scaled)
y_test_top_pred = rf_model_top.predict(X_test_top_scaled)
top_train_accuracy = accuracy_score(y_train, y_train_top_pred)
top_train_precision = precision_score(y_train, y_train_top_pred, average='weighted')
top_train_recall = recall_score(y_train, y_train_top_pred, average='weighted')
top_train_f1 = f1_score(y_train, y_train_top_pred, average='weighted')
top_test_accuracy = accuracy_score(y_test, y_test_top_pred)
top_test_precision = precision_score(y_test, y_test_top_pred, average='weighted')
top_test_recall = recall_score(y_test, y_test_top_pred, average='weighted')
top_test_f1 = f1_score(y_test, y_test_top_pred, average='weighted')

logger.debug(f"Top features - Train Accuracy: {top_train_accuracy:.4f}, Precision: {top_train_precision:.4f}, Recall: {top_train_recall:.4f}, F1: {top_train_f1:.4f}")
logger.debug(f"Top features - Test Accuracy: {top_test_accuracy:.4f}, Precision: {top_test_precision:.4f}, Recall: {top_test_recall:.4f}, F1: {top_test_f1:.4f}")

# Save top features metrics
top_metrics = {
    "top_train_accuracy": top_train_accuracy,
    "top_train_precision": top_train_precision,
    "top_train_recall": top_train_recall,
    "top_train_f1": top_train_f1,
    "top_test_accuracy": top_test_accuracy,
    "top_test_precision": top_test_precision,
    "top_test_recall": top_test_recall,
    "top_test_f1": top_test_f1
}
with open('outputs/top_metrics_model3.txt', 'w') as f:
    f.write(str(top_metrics))
logger.debug("Saved top features metrics to outputs/top_metrics_model3.txt")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_top_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Critical', 'Degraded', 'Healthy'], yticklabels=['Critical', 'Degraded', 'Healthy'])
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('outputs/confusion_matrix_model3.png')
plt.close()

# Time-series of health states
df_test = X_test.copy()
df_test['Health_State_Predicted'] = y_test_top_pred
df_test['Time (h)'] = X_test['Time (h)']
plt.figure(figsize=(12, 6))
plt.scatter(df_test['Time (h)'], df_test['Health_State_Predicted'], c=df_test['Health_State_Predicted'], cmap='viridis', alpha=0.5)
plt.xlabel('Time (h)')
plt.ylabel('Predicted Health State (0=Critical, 1=Degraded, 2=Healthy)')
plt.title('Predicted Health States over Time (Test Set)')
plt.savefig('outputs/health_states_model3.png')
plt.close()

# Save models and scaler
joblib.dump(rf_model, 'models/rf_health_model3.pkl')
joblib.dump(rf_model_top, 'models/rf_health_top_model3.pkl')
joblib.dump(scaler, 'models/scaler_model3.pkl')
joblib.dump(top_features, 'models/top_features_model3.pkl')
logger.debug("Saved models, scaler, and top features to models/")