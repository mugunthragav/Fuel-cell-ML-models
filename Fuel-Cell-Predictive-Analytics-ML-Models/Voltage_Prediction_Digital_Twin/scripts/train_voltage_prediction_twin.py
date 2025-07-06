import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import xgboost as xgb
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create models and outputs directories
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'
DATA_DIR = 'data' # Assuming data is in 'data' subdirectory of the project folder

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True) # Ensure data dir exists
logger.debug(f"Ensured directories exist: {MODELS_DIR}, {OUTPUTS_DIR}, {DATA_DIR}")

# Load dataset
try:
    df = pd.read_csv(os.path.join(DATA_DIR, 'FC_Ageing.csv'))
    logger.debug(f"Loaded dataset with shape: {df.shape}")
    logger.debug(f"Columns: {df.columns.tolist()}")
except Exception as e:
    logger.error(f"Error loading dataset from {os.path.join(DATA_DIR, 'FC_Ageing.csv')}: {str(e)}")
    raise

# Preprocessing
# Handle missing values
df = df.fillna(df.mean(numeric_only=True))
logger.debug("Filled missing values with column means")

# Drop columns that are constant or almost constant (standard deviation very close to zero)
constant_or_near_constant_cols = [col for col in df.columns if df[col].std() < 1e-6 and col != 'Utot (V)']
if constant_or_near_constant_cols:
    df = df.drop(columns=constant_or_near_constant_cols)
    logger.debug(f"Dropped constant or near-constant columns: {constant_or_near_constant_cols}")
else:
    logger.debug("No constant or near-constant columns found to drop.")

# Define features and target
target = 'Utot (V)'
if target not in df.columns:
    logger.error(f"Target column '{target}' not found in dataset")
    raise KeyError(f"Target column '{target}' not found in dataset")

features_to_exclude_from_input = [target, 'U1 (V)', 'U2 (V)', 'U3 (V)', 'U4 (V)', 'U5 (V)']
features = [col for col in df.columns if col not in features_to_exclude_from_input]

if 'J (A/cm²)' not in features and 'J (A/cmÂ²)' in df.columns:
    features.append('J (A/cmÂ²)')
elif 'J (A/cm²)' not in features and 'J (A/cm²)' in df.columns:
    features.append('J (A/cm²)')

if 'I (A)' not in features and 'I (A)' in df.columns:
    features.append('I (A)')

features = list(set(features))
features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

logger.debug(f"Selected features for model training: {features}")

# Split features and target
X = df[features]
y = df[target]

# Handle outliers using IQR-based capping
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X.loc[:, col] = X[col].clip(lower_bound, upper_bound)
logger.debug("Capped outliers using IQR method")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.debug(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logger.debug("Scaled features using StandardScaler")

# Save scaler and feature names
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_model2.pkl'))
joblib.dump(features, os.path.join(MODELS_DIR, 'top_features_model2.pkl'))
logger.debug("Saved StandardScaler and feature names to models/")

# Define XGBoost model (initial estimator for GridSearchCV)
xgb_regressor = xgb.XGBRegressor(random_state=42, tree_method='hist', enable_categorical=False)

# Define a much smaller parameter grid for GridSearchCV for faster execution
param_grid = {
    'n_estimators': [50, 100], # Reduced options
    'max_depth': [5, 7],       # Reduced options
    'learning_rate': [0.1],    # Single value
    'subsample': [1.0],        # Single value
    'colsample_bytree': [1.0], # Single value
    'gamma': [0]               # Single value
}

# Use KFold for cross-validation with fewer splits
cv_strategy = KFold(n_splits=2, shuffle=True, random_state=42) # Reduced to 2 folds

# Use negative root mean squared error as the scoring metric
scorer = make_scorer(mean_squared_error, greater_is_better=False)

logger.info("Starting GridSearchCV for XGBoost Regressor with reduced parameters...")
grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid,
                           cv=cv_strategy, scoring=scorer, verbose=1, n_jobs=-1) # verbose=1 for less output

grid_search.fit(X_train_scaled, y_train)
logger.info("GridSearchCV completed.")

best_xgb_model = grid_search.best_estimator_
logger.info(f"Best parameters found: {grid_search.best_params_}")
logger.info(f"Best cross-validation RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

# Predictions with the best model
y_train_pred_best = best_xgb_model.predict(X_train_scaled)
y_test_pred_best = best_xgb_model.predict(X_test_scaled)

# Compute metrics for the best model
train_rmse_best = np.sqrt(mean_squared_error(y_train, y_train_pred_best))
train_mae_best = mean_absolute_error(y_train, y_train_pred_best)
train_r2_best = r2_score(y_train, y_train_pred_best)
test_rmse_best = np.sqrt(mean_squared_error(y_test, y_test_pred_best))
test_mae_best = mean_absolute_error(y_test, y_test_pred_best)
test_r2_best = r2_score(y_test, y_test_pred_best)

logger.info(f"Best Model - Train RMSE: {train_rmse_best:.4f}, MAE: {train_mae_best:.4f}, R2: {test_r2_best:.4f}")
logger.info(f"Best Model - Test RMSE: {test_rmse_best:.4f}, MAE: {test_mae_best:.4f}, R2: {test_r2_best:.4f}")

# Save metrics for the best model
metrics_best = {
    "best_params": grid_search.best_params_,
    "cv_rmse": np.sqrt(-grid_search.best_score_),
    "train_rmse": train_rmse_best,
    "train_mae": train_mae_best,
    "train_r2": train_r2_best,
    "test_rmse": test_rmse_best,
    "test_mae": test_mae_best,
    "test_r2": test_r2_best
}
with open(os.path.join(OUTPUTS_DIR, 'metrics_xgb_tuned.txt'), 'w') as f:
    f.write(json.dumps(metrics_best, indent=4))
logger.info("Saved best model metrics to outputs/metrics_xgb_tuned.txt")

# Correlation analysis
corr_matrix_final = df[features + [target]].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_final, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (Selected Features)')
plt.savefig(os.path.join(OUTPUTS_DIR, 'correlation_matrix_selected_features.png'))
plt.close()
logger.info("Saved updated correlation matrix plot.")

# Feature importance
feat_importance_best = pd.Series(best_xgb_model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feat_importance_best.plot(kind='bar')
plt.title('XGBoost Feature Importance (Tuned Model)')
plt.savefig(os.path.join(OUTPUTS_DIR, 'feature_importance_tuned_xgb.png'))
plt.close()
logger.info(f"Feature importance (tuned model): {feat_importance_best.to_dict()}")

# Actual vs. Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Utot (V)')
plt.ylabel('Predicted Utot (V)')
plt.title('Actual vs. Predicted Utot (Test Set - Tuned Model)')
plt.legend()
plt.savefig(os.path.join(OUTPUTS_DIR, 'actual_vs_predicted_tuned.png'))
plt.close()
logger.info("Saved actual vs. predicted plot.")

# Residual plot
residuals_best = y_test - y_test_pred_best
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred_best, residuals_best, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Utot (V)')
plt.ylabel('Residuals')
plt.title('Residual Plot (Test Set - Tuned Model)')
plt.savefig(os.path.join(OUTPUTS_DIR, 'residual_plot_tuned.png'))
plt.close()
logger.info("Saved residual plot.")

# Save the best model
joblib.dump(best_xgb_model, os.path.join(MODELS_DIR, 'xgb_utot_model2.pkl'))
joblib.dump(best_xgb_model, os.path.join(MODELS_DIR, 'xgb_utot_top_model2.pkl'))
logger.info("Saved best XGBoost model and top features to models/")

logger.info("Training script finished successfully.")
