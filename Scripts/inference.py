"""
Inference module for the Diabetes Prediction Model.
"""

import numpy as np
import pandas as pd
import joblib
import os
from scripts.config import MODEL_PATH, SCALER_PATH, FEATURE_NAMES
from scripts.preprocess import preprocess_data

def load_model_and_scaler():
    """
    Load the trained model and scaler from disk.
    
    Returns:
        tuple: model, scaler
    """
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # Check if scaler exists
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    
    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    return model, scaler

def predict_single(data, model=None, scaler=None):
    """
    Make a prediction for a single data point.
    
    Args:
        data (dict): Dictionary with feature values
        model: Trained model (optional, will load if None)
        scaler: Fitted scaler (optional, will load if None)
        
    Returns:
        tuple: prediction (0 or 1), probability
    """
    # Load model and scaler if not provided
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler()
    
    # Convert input to DataFrame
    if isinstance(data, dict):
        # Ensure all features are present
        for feature in FEATURE_NAMES:
            if feature not in data:
                data[feature] = 0
        
        # Create DataFrame with features in correct order
        df = pd.DataFrame([data])[FEATURE_NAMES]
    else:
        df = pd.DataFrame([data], columns=FEATURE_NAMES)
    
    # Preprocess the data (handle missing values)
    X, _ = preprocess_data(df)
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    
    return int(prediction), float(probability)

def predict_batch(data, model=None, scaler=None):
    """
    Make predictions for a batch of data.
    
    Args:
        data (pandas.DataFrame): DataFrame with feature values
        model: Trained model (optional, will load if None)
        scaler: Fitted scaler (optional, will load if None)
        
    Returns:
        tuple: predictions array, probabilities array
    """
    # Load model and scaler if not provided
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler()
    
    # Ensure data has all required features
    for feature in FEATURE_NAMES:
        if feature not in data.columns:
            data[feature] = 0
    
    # Select only the required features in the correct order
    data = data[FEATURE_NAMES]
    
    # Preprocess the data
    X, _ = preprocess_data(data)
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return predictions, probabilities

def get_prediction_explanation(data, model=None, scaler=None):
    """
    Get an explanation for a prediction based on feature importance.
    
    Args:
        data (dict): Dictionary with feature values
        model: Trained model (optional, will load if None)
        scaler: Fitted scaler (optional, will load if None)
        
    Returns:
        dict: Explanation with feature contributions
    """
    # Load model and scaler if not provided
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler()
    
    # Get feature importance from model
    feature_importance = model.feature_importances_
    
    # Convert input to DataFrame
    if isinstance(data, dict):
        # Create DataFrame with features in correct order
        df = pd.DataFrame([data])[FEATURE_NAMES]
    else:
        df = pd.DataFrame([data], columns=FEATURE_NAMES)
    
    # Preprocess the data
    X, _ = preprocess_data(df)
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Calculate contribution of each feature
    contributions = {}
    for i, feature in enumerate(FEATURE_NAMES):
        # Contribution = feature value * feature importance
        contributions[feature] = float(X_scaled[0, i] * feature_importance[i])
    
    # Sort contributions by absolute value
    sorted_contributions = sorted(
        contributions.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    # Make prediction
    prediction, probability = predict_single(data, model, scaler)
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'contributions': dict(sorted_contributions),
        'feature_importance': dict(zip(FEATURE_NAMES, feature_importance.tolist()))
    }

if __name__ == "__main__":
    # Test the inference module
    test_data = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    prediction, probability = predict_single(test_data)
    print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
    print(f"Probability of diabetes: {probability:.4f}")
    
    explanation = get_prediction_explanation(test_data)
    print("\nFeature contributions:")
    for feature, contribution in explanation['contributions'].items():
        print(f"{feature}: {contribution:.4f}")
