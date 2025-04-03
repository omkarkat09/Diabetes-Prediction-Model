"""
Data preprocessing module for the Diabetes Prediction Model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from scripts.config import DATA_PATH, SCALER_PATH, TEST_SIZE, RANDOM_STATE, TARGET_COLUMN, FEATURE_NAMES

def load_data(data_path=None):
    """
    Load the diabetes dataset from the specified path.
    
    Args:
        data_path (str, optional): Path to the dataset. Defaults to None.
        
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    if data_path is None:
        data_path = DATA_PATH
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Check if file exists, if not download from UCI repository
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Downloading from UCI repository...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        column_names = FEATURE_NAMES + [TARGET_COLUMN]
        df = pd.read_csv(url, names=column_names)
        
        # Save to specified path
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Data saved to {data_path}")
    else:
        # Load from existing file
        df = pd.read_csv(data_path)
        
    return df

def preprocess_data(df):
    """
    Preprocess the diabetes dataset.
    
    Args:
        df (pandas.DataFrame): The dataset to preprocess
        
    Returns:
        tuple: X (features), y (target)
    """
    # Handle missing values (replace 0 values with NaN for certain columns)
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_columns:
        if column in df.columns:
            df[column] = df[column].replace(0, np.nan)
    
    # Fill missing values with median
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].median())
    
    # Extract features and target
    if TARGET_COLUMN in df.columns:
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
    else:
        X = df
        y = None
    
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def scale_features(X_train, X_test=None, fit_scaler=True):
    """
    Scale the features using StandardScaler.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame, optional): Testing features. Defaults to None.
        fit_scaler (bool, optional): Whether to fit a new scaler or load existing. Defaults to True.
        
    Returns:
        tuple: Scaled X_train, X_test (if provided), and the scaler object
    """
    if fit_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Save the scaler
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    else:
        # Load existing scaler
        scaler = joblib.load(SCALER_PATH)
        X_train_scaled = scaler.transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler

def prepare_data(data_path=None, split=True, scale=True):
    """
    Complete data preparation pipeline.
    
    Args:
        data_path (str, optional): Path to the dataset. Defaults to None.
        split (bool, optional): Whether to split data. Defaults to True.
        scale (bool, optional): Whether to scale features. Defaults to True.
        
    Returns:
        tuple: Prepared data components depending on parameters
    """
    # Load data
    df = load_data(data_path)
    
    # Preprocess
    X, y = preprocess_data(df)
    
    if split:
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        if scale:
            # Scale features
            X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        
        return X_train, X_test, y_train, y_test
    
    if scale:
        # Scale features without splitting
        X_scaled, scaler = scale_features(X)
        return X_scaled, y, scaler
    
    return X, y

if __name__ == "__main__":
    # Test the preprocessing pipeline
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data()
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Testing data shape: {X_test_scaled.shape}")
    print(f"Scaler saved to: {SCALER_PATH}")
