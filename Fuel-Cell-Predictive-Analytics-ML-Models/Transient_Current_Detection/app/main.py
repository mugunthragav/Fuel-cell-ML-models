from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Transient Current Fault Prediction API")

# Define paths to models and transformers
MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
BEST_TRANSFORMER_PATH = os.path.join(MODELS_DIR, 'best_transformer.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_model1.pkl')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.pkl')

# Load pre-trained models and transformers
try:
    best_model = joblib.load(BEST_MODEL_PATH)
    best_transformer = joblib.load(BEST_TRANSFORMER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    logger.info("Successfully loaded best model, transformer, scaler, and feature names.")
except FileNotFoundError as e:
    logger.error(f"Error loading model files: {e}. Ensure 'scripts/train_model.py' has been run successfully.")
    # Exit or raise an exception to prevent the app from starting without models
    raise RuntimeError(f"Missing model files. Please run 'scripts/train_model.py' first. Error: {e}")
except Exception as e:
    logger.error(f"An unexpected error occurred while loading model files: {e}")
    raise RuntimeError(f"Failed to load model files. Error: {e}")


# Dynamically create a Pydantic model for input features
# This ensures that the API expects all the features the model was trained on
# and provides better documentation in OpenAPI (Swagger UI)
# For simplicity, we'll use a dict and validate features manually in the endpoint
# If you want a strict Pydantic model with all features, you'd generate it here.
class PredictionInput(BaseModel):
    # Features will be a dictionary where keys are feature names and values are floats/ints
    features: dict[str, float] = Field(
        ...,
        example={
            "AOP1": 0.0, "AOP2": 0.0, "COP1": 700.0, "COP2": 700.0, "I": 105.0,
            "ARF": 282.0, "AIP1": 1776.0, "AIP2": 1776.0, "CAIF": 1684.0, "CIP1": 1252.0,
            "CS": 2.5, "COT2": 75.0, "CIT2": 45.0, "COT1": 75.0, "CIT1": 46.0,
            "WIP2": 1932.0, "WIP1": 1912.0, "WIF2": 820.0, "WIF1": 818.0, "WIT": 37.0
        },
        description="Dictionary of feature names and their corresponding numerical values."
    )

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Predict fuel cell fault state (Faulty or Normal) based on input features
    using the best pre-trained model.
    """
    try:
        # Convert input dictionary to a pandas DataFrame
        input_df = pd.DataFrame([input_data.features])

        # Ensure all expected features are present and in the correct order
        missing_cols = [col for col in feature_names if col not in input_df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_cols}. Please provide all {len(feature_names)} features: {feature_names}")

        # Ensure no extra columns are passed that the model wasn't trained on
        extra_cols = [col for col in input_df.columns if col not in feature_names]
        if extra_cols:
            logger.warning(f"Extra features provided in input and will be ignored: {extra_cols}")
            input_df = input_df.drop(columns=extra_cols)

        # Reorder columns to match the training data's feature order
        input_df = input_df[feature_names]

        # Apply the scaler
        input_scaled = scaler.transform(input_df)

        # Apply the best transformer (PCA, SVD, or SelectKBest)
        input_transformed = best_transformer.transform(input_scaled)

        # Make prediction
        prediction = best_model.predict(input_transformed)[0]

        # Determine the fault state
        result = "Faulty" if prediction == 0 else "Normal" if prediction == 1 else "Unknown"

        logger.info(f"Prediction made: {result} for input features: {input_data.features}")

        return {
            "prediction_status": "Success",
            "prediction": result,
            "features_provided": list(input_data.features.keys())
        }
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Transient Current Fault Prediction API. Use /predict for predictions."}
