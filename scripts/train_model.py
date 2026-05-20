"""
Model training module for the Diabetes Prediction Model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.config import MODEL_PATH, RF_PARAMS, RANDOM_STATE
from scripts.preprocess import prepare_data

def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        params (dict, optional): Model parameters. Defaults to None.
        
    Returns:
        RandomForestClassifier: Trained model
    """
    if params is None:
        params = RF_PARAMS
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest hyperparameters using GridSearchCV.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        
    Returns:
        RandomForestClassifier: Best model
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='roc_auc',
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance.
    
    Args:
        model: Trained model
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing target
        
    Returns:
        dict: Performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the plot
    os.makedirs('../static/images', exist_ok=True)
    plt.savefig('../static/images/confusion_matrix.png')
    plt.close()
    
    # Plot feature importance
    feature_importance = model.feature_importances_
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Create DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('../static/images/feature_importance.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'report': report,
        'confusion_matrix': cm,
        'feature_importance': dict(zip(feature_names, feature_importance))
    }

def save_model(model, path=None):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        path (str, optional): Path to save the model. Defaults to None.
        
    Returns:
        str: Path where model was saved
    """
    if path is None:
        path = MODEL_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    
    return path

def load_model(path=None):
    """
    Load a trained model from disk.
    
    Args:
        path (str, optional): Path to the model. Defaults to None.
        
    Returns:
        The loaded model
    """
    if path is None:
        path = MODEL_PATH
    
    # Check if model exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    
    # Load model
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    
    return model

def train_and_evaluate():
    """
    Complete training pipeline: prepare data, train model, evaluate, and save.
    
    Returns:
        tuple: Trained model and evaluation metrics
    """
    # Prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data()
    
    # Train model with hyperparameter tuning
    print("Tuning Random Forest hyperparameters...")
    model = tune_random_forest(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model)
    
    return model, metrics

if __name__ == "__main__":
    # Run the training pipeline
    model, metrics = train_and_evaluate()
    print("\nTraining completed successfully!")
