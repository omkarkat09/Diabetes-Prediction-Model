"""
Script to test the inference functionality of the Diabetes Prediction Model.
This script tests the model's ability to make predictions on sample data.
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference import predict_single, predict_batch, get_prediction_explanation

def test_single_prediction():
    """Test prediction on a single sample."""
    print("Testing single prediction...")
    
    # Sample data for a person with high risk of diabetes
    high_risk_sample = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    # Sample data for a person with low risk of diabetes
    low_risk_sample = {
        'Pregnancies': 1,
        'Glucose': 85,
        'BloodPressure': 66,
        'SkinThickness': 29,
        'Insulin': 0,
        'BMI': 26.6,
        'DiabetesPedigreeFunction': 0.351,
        'Age': 31
    }
    
    # Make predictions
    high_risk_pred, high_risk_prob = predict_single(high_risk_sample)
    low_risk_pred, low_risk_prob = predict_single(low_risk_sample)
    
    # Print results
    print(f"\nHigh risk sample prediction: {high_risk_pred} (Probability: {high_risk_prob:.4f})")
    print(f"Low risk sample prediction: {low_risk_pred} (Probability: {low_risk_prob:.4f})")
    
    # Test explanation functionality
    print("\nTesting prediction explanation...")
    high_risk_explanation = get_prediction_explanation(high_risk_sample)
    
    print("\nFeature contributions for high risk sample:")
    for feature, contribution in high_risk_explanation['contributions'].items():
        print(f"  {feature}: {contribution:.4f}")
    
    return high_risk_explanation, low_risk_sample

def test_batch_prediction():
    """Test prediction on a batch of samples."""
    print("\nTesting batch prediction...")
    
    # Create a small batch of test data
    test_data = pd.DataFrame([
        {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        },
        {
            'Pregnancies': 1,
            'Glucose': 85,
            'BloodPressure': 66,
            'SkinThickness': 29,
            'Insulin': 0,
            'BMI': 26.6,
            'DiabetesPedigreeFunction': 0.351,
            'Age': 31
        },
        {
            'Pregnancies': 8,
            'Glucose': 183,
            'BloodPressure': 64,
            'SkinThickness': 0,
            'Insulin': 0,
            'BMI': 23.3,
            'DiabetesPedigreeFunction': 0.672,
            'Age': 32
        }
    ])
    
    # Make batch predictions
    predictions, probabilities = predict_batch(test_data)
    
    # Print results
    print("\nBatch prediction results:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"  Sample {i+1}: Prediction = {pred}, Probability = {prob:.4f}")
    
    return predictions, probabilities

def main():
    """Run all inference tests."""
    print("Starting Diabetes Prediction Model inference tests...\n")
    
    # Test single prediction
    high_risk_explanation, low_risk_sample = test_single_prediction()
    
    # Test batch prediction
    predictions, probabilities = test_batch_prediction()
    
    print("\nInference tests completed successfully!")
    
    return high_risk_explanation, low_risk_sample, predictions, probabilities

if __name__ == "__main__":
    main()
