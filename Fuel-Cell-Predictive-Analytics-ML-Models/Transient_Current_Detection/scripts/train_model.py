import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    make_scorer
)
from imblearn.over_sampling import SMOTE
from scipy import stats
import joblib
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define directories
models_dir = "models"
outputs_dir = "outputs"
data_dir = "data"

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
logger.info(f"Ensured directories exist: {models_dir}, {outputs_dir}, {data_dir}")

def evaluate_model(model, X_test, y_test, model_name, outputs_dir, prefix=""):
    """Evaluates a model and saves its metrics, confusion matrix, and PR curves (ROC skipped)."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # Probabilities for all classes

    # Use argmax for multiclass prediction if more than 2 classes
    n_classes = len(model.classes_)
    if n_classes > 2:
        y_pred_opt = np.argmax(y_proba, axis=1)
    else:
        y_pred_opt = (y_proba[:, 1] >= 0.5).astype(int)  # Binary threshold

    accuracy = accuracy_score(y_test, y_pred_opt)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_opt)
    report = classification_report(y_test, y_pred_opt, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_opt)
    f1_normal = f1_score(y_test, y_pred_opt, pos_label=0) if n_classes == 2 else f1_score(y_test, y_pred_opt, average='weighted')

    logger.info(f"{model_name} Results (Threshold: 0.50):")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"F1-Score (Normal): {f1_normal:.4f}")
    logger.info(f"Classification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[f'Pred {i}' for i in range(n_classes)],
                yticklabels=[f'Act {i}' for i in range(n_classes)])
    plt.title(f'Confusion Matrix - {model_name} {prefix}(Threshold: 0.50)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}{prefix}.png'))
    plt.close()
    logger.info(f"Saved confusion matrix plot for {model_name}.")

    # Precision-Recall Curve (for multiclass, average or per class)
    if n_classes > 2:
        precision = dict()
        recall = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test == i, y_proba[:, i])
            plt.figure(figsize=(8, 6))
            plt.plot(recall[i], precision[i], label=f'Class {i}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name} {prefix}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(outputs_dir, f'pr_curve_{model_name.lower().replace(" ", "_")}{prefix}.png'))
        plt.close()
    else:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba[:, 1], pos_label=1)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=f'{model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name} {prefix}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(outputs_dir, f'pr_curve_{model_name.lower().replace(" ", "_")}{prefix}.png'))
        plt.close()
    logger.info(f"Saved Precision-Recall curve plot for {model_name}.")

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score_normal': f1_normal,
        'report': report,
        'confusion_matrix': cm.tolist()
    }

def main():
    # Load the dataset
    excel_path = os.path.join(data_dir, 'combined_all_files.xlsx')
    csv_path = os.path.join(data_dir, 'data3040b.csv')

    logger.info(f"Loading dataset from {excel_path}")
    try:
        data = pd.read_excel(excel_path)
        data.to_csv(csv_path, index=False)
        logger.info(f"Saved dataset as CSV: {csv_path}")
    except FileNotFoundError:
        logger.error(f"Dataset not found at {excel_path}. Please ensure 'combined_all_files.xlsx' is in the '{data_dir}' directory.")
        return
    except Exception as e:
        logger.error(f"Error loading or saving data: {e}")
        return

    # Handle missing values
    data = data.fillna(data.mean(numeric_only=True))
    logger.info("Filled missing values with column means")

    # Drop constant columns
    constant_columns = [col for col in data.columns if data[col].nunique() == 1]
    if constant_columns:
        logger.info(f"Dropping constant columns: {constant_columns}")
        data = data.drop(columns=constant_columns)
    else:
        logger.info("No constant columns found.")

    # Feature selection: Remove highly correlated features
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if 'fault' in numeric_cols:
        numeric_cols.remove('fault')

    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
        if to_drop_corr:
            data = data.drop(columns=to_drop_corr)
            logger.info(f"Dropped highly correlated columns (correlation > 0.9): {to_drop_corr}")
        else:
            logger.info("No highly correlated columns found to drop.")
    else:
        logger.warning("Not enough numeric columns to calculate correlation matrix for feature selection.")

    # Feature selection: Remove low-variance features
    numeric_cols_after_corr = data.select_dtypes(include=np.number).columns.tolist()
    if 'fault' in numeric_cols_after_corr:
        numeric_cols_after_corr.remove('fault')

    if numeric_cols_after_corr:
        variance = data[numeric_cols_after_corr].var()
        to_drop_var = variance[variance < 0.01].index.tolist()
        if to_drop_var:
            data = data.drop(columns=to_drop_var)
            logger.info(f"Dropping low-variance columns (variance < 0.01): {to_drop_var}")
        else:
            logger.info("No low-variance columns found to drop.")
    else:
        logger.warning("No numeric columns remaining after correlation filtering to check for low variance.")

    # Separate features and target
    if 'fault' not in data.columns:
        logger.error("Target column 'fault' not found in the dataset. Please ensure your dataset has a 'fault' column.")
        return
    X = data.drop(columns=['fault'])
    y = data['fault']

    # Preserve original classes (0, 1, 2) if present
    original_unique_y = np.unique(y)
    logger.info(f"Original unique values in target variable 'fault': {original_unique_y}")
    if len(original_unique_y) > 2:
        logger.info("Target variable 'fault' has three or more unique values. Preserving as multiclass (0=Normal, 1=Faulty, 2=Critical).")
    elif len(original_unique_y) == 1:
        logger.error(f"Target variable 'fault' has only one unique value ({original_unique_y}). Cannot perform classification. Exiting.")
        return
    elif not all(val in [0, 1, 2] for val in original_unique_y):
        logger.warning(f"Target variable 'fault' has invalid values ({original_unique_y}). Mapping to 0=Normal, 1/2=Faulty.")
        y = y.apply(lambda x: 0 if x == 0 else 1)
    logger.info(f"Target variable 'fault' unique values: {np.unique(y)}")

    # Handle outliers consistently using Z-score
    if not X.empty and X.select_dtypes(include=np.number).shape[1] > 0:
        z_scores = np.abs(stats.zscore(X.select_dtypes(include=np.number)))
        mask = (z_scores < 3).all(axis=1)
        X = X[mask]
        y = y[mask]
        logger.info(f"Removed outliers using z-score (threshold 3), new sample size: {len(X)}")
    else:
        logger.warning("Skipping outlier removal: X is empty or has no numeric columns.")

    # Verify sample consistency and minimum samples for splitting/SMOTE
    if len(X) < 2 or len(y) < 2:
        logger.error(f"Insufficient samples after preprocessing: X={len(X)}, y={len(y)}. Need at least 2 samples. Exiting.")
        return
    if len(X) != len(y):
        logger.error(f"Inconsistent sample sizes after outlier removal: X={len(X)}, y={len(y)}. Exiting.")
        return
    if len(np.unique(y)) < 2:
        logger.error(f"Only one class ({np.unique(y)}) remains after preprocessing. Cannot perform classification. Exiting.")
        return

    # Save the feature names BEFORE scaling and SMOTE but AFTER initial selection
    initial_feature_names = X.columns.tolist()
    joblib.dump(initial_feature_names, os.path.join(models_dir, 'initial_feature_names.pkl'))
    logger.info(f"Saved initial feature names ({len(initial_feature_names)} features).")

    # Split the data
    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    logger.info("Applying StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler_model1.pkl'))
    logger.info("Saved StandardScaler to 'models/scaler_model1.pkl'")

    # Convert scaled arrays back to DataFrame to retain column names for SelectKBest
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Feature Selection using SelectKBest
    k_features = min(6, X_train_scaled_df.shape[1])  # Select top 6 features for speed
    if X_train_scaled_df.shape[1] > 0 and len(np.unique(y_train)) > 1:
        logger.info(f"Applying SelectKBest to select top {k_features} features.")
        kbest = SelectKBest(score_func=f_classif, k=k_features)
        X_train_selected = kbest.fit_transform(X_train_scaled_df, y_train)
        X_test_selected = kbest.transform(X_test_scaled_df)
        
        selected_feature_indices = kbest.get_support(indices=True)
        selected_feature_names = X_train_scaled_df.columns[selected_feature_indices].tolist()
        joblib.dump(selected_feature_names, os.path.join(models_dir, 'selected_feature_names.pkl'))
        joblib.dump(kbest, os.path.join(models_dir, 'kbest_transformer.pkl'))
        logger.info(f"Selected {len(selected_feature_names)} features: {selected_feature_names}")
    else:
        logger.warning("Skipping SelectKBest: Not enough features or target has only one class. Using all scaled features.")
        X_train_selected = X_train_scaled
        X_test_selected = X_test_scaled
        selected_feature_names = X_train.columns.tolist()  # All original features if no selection
        joblib.dump(selected_feature_names, os.path.join(models_dir, 'selected_feature_names.pkl'))

    # Apply SMOTE to balance classes with a controlled ratio
    if len(np.unique(y_train)) > 1 and np.min(np.bincount(y_train)) >= 2:
        logger.info("Applying SMOTE to handle class imbalance")
        min_samples = np.min(np.bincount(y_train))
        smote_k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        if smote_k_neighbors < 1:
            logger.warning(f"SMOTE cannot be applied as minority class has only {min_samples} sample(s). Skipping SMOTE.")
            X_train_final = X_train_selected
            y_train_final = y_train
        else:
            smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors, sampling_strategy='auto')
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
            logger.info(f"After SMOTE, class distribution: {np.bincount(y_train_resampled)}")
            X_train_final = X_train_resampled
            y_train_final = y_train_resampled
    else:
        logger.warning("Skipping SMOTE: Not enough classes or minority class samples.")
        X_train_final = X_train_selected
        y_train_final = y_train

    # Define the model
    model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Define a much smaller parameter grid for GridSearchCV for faster execution
    param_grid = {
        'n_estimators': [10, 20],  # Reduced number of trees
        'max_depth': [10, 20],  # Reduced depth options
        'min_samples_split': [5],  # Single value
        'min_samples_leaf': [2],  # Single value
        'max_features': ['sqrt', 'log2']  # Still two options
    }

    # Use StratifiedKFold for cross-validation with fewer splits
    cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Reduced to 2 folds

    # Use balanced_accuracy as the scoring metric for GridSearchCV
    scorer = make_scorer(balanced_accuracy_score)

    logger.info("Starting GridSearchCV for RandomForestClassifier with reduced parameters...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                              cv=cv_strategy, scoring=scorer, verbose=1, n_jobs=-1)  # verbose=1 for less output
    
    grid_search.fit(X_train_final, y_train_final)
    logger.info("GridSearchCV completed.")

    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation balanced accuracy: {grid_search.best_score_:.4f}")

    # Evaluate the best model on the test set
    logger.info("Evaluating the best model on the test set.")
    rf_metrics = evaluate_model(best_model, X_test_selected, y_test, "RandomForestClassifier_Tuned", outputs_dir, prefix="_tuned")

    # Save the best model
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    logger.info("Saved best RandomForestClassifier model (after tuning) as 'best_model.pkl'")

    # Plot Feature Importances
    if hasattr(best_model, 'feature_importances_') and selected_feature_names:
        importances = best_model.feature_importances_
        if len(importances) == len(selected_feature_names):
            feature_importance_df = pd.DataFrame({'feature': selected_feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
            plt.title('Top 15 Feature Importances from Tuned RandomForestClassifier')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(outputs_dir, 'feature_importances_tuned_rf.png'))
            plt.close()
            logger.info("Saved feature importances plot.")
        else:
            logger.warning("Could not plot feature importances: Mismatch between importances and selected feature names length.")
    else:
        logger.warning("Could not plot feature importances: Model does not have feature_importances_ attribute or no selected features.")

    # Save metrics to a text file
    metrics_file_path = os.path.join(outputs_dir, 'metrics_randomforest_tuned.txt')
    with open(metrics_file_path, 'w') as f:
        f.write("--- Model Performance Metrics (RandomForestClassifier - Tuned) ---\n\n")
        f.write(f"Best Parameters (from GridSearchCV): {grid_search.best_params_}\n")
        f.write(f"Best Cross-Validation Balanced Accuracy: {grid_search.best_score_:.4f}\n\n")
        f.write(f"Test Set Evaluation:\n")
        f.write(f"  Accuracy: {rf_metrics['accuracy']:.4f}\n")
        f.write(f"  Balanced Accuracy: {rf_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"  F1-Score (Normal): {rf_metrics['f1_score_normal']:.4f}\n")
        f.write(f"  ROC AUC: {rf_metrics['roc_auc']:.4f}\n")
        f.write(f"  Classification Report:\n{rf_metrics['report']}\n")
        f.write(f"  Confusion Matrix:\n{np.array(rf_metrics['confusion_matrix'])}\n")
        if selected_feature_names:
            f.write(f"\nSelected Features: {selected_feature_names}\n")
    logger.info(f"All model metrics saved to {metrics_file_path}")

    logger.info("Training script finished successfully.")

if __name__ == "__main__":
    main()