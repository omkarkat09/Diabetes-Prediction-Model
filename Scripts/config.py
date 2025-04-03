"""
Configuration settings for the Diabetes Prediction Model.
"""

# Data paths
DATA_PATH = "../data/diabetes.csv"
MODEL_PATH = "../models/random_forest_model.pkl"
SCALER_PATH = "../models/scaler.pkl"

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

# Flask app settings
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True
