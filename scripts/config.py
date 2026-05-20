"""
Configuration settings for the Diabetes Prediction Model.
"""

import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_PATH = os.path.join(BASE_DIR, "Data", "diabetes.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Models", "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Models", "scaler.pkl")

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE
}

# Feature names
FEATURE_NAMES = [
    'Pregnancies', 
    'Glucose', 
    'BloodPressure', 
    'SkinThickness', 
    'Insulin', 
    'BMI', 
    'DiabetesPedigreeFunction', 
    'Age'
]

# Target column name
TARGET_COLUMN = 'Outcome'

import os

# Flask app settings
HOST = '0.0.0.0'
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
