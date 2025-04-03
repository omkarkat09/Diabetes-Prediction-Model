"""
Script to download the diabetes dataset and train the model.
This script runs the complete pipeline from data preprocessing to model training and evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess import prepare_data
from scripts.train_model import train_and_evaluate

def main():
    """Run the complete model training pipeline."""
    print("Starting Diabetes Prediction Model training pipeline...")
    
    # Create necessary directories
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../static/images", exist_ok=True)
    
    # Download and prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Train and evaluate model
    print("\nTraining and evaluating model...")
    model, metrics = train_and_evaluate()
    
    print("\nModel training and evaluation completed successfully!")
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    print(f"Model ROC AUC: {metrics['roc_auc']:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    main()
