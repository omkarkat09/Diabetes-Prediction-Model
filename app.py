"""
Flask web application for the Diabetes Prediction Model.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.inference import predict_single, get_prediction_explanation
from scripts.config import FEATURE_NAMES, HOST, PORT, DEBUG

# Create Flask app with correct template and static folder paths
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if request.method == 'POST':
        try:
            # Get form data
            data = {}
            for feature in FEATURE_NAMES:
                value = request.form.get(feature, '')
                if value == '':
                    # Handle empty inputs
                    return render_template('index.html', error=f"Please provide a value for {feature}")
                try:
                    data[feature] = float(value)
                except ValueError:
                    return render_template('index.html', error=f"Invalid value for {feature}. Please enter a number.")
            
            # Get prediction and explanation
            explanation = get_prediction_explanation(data)
            prediction = explanation['prediction']
            probability = explanation['probability']
            
            # Prepare feature contributions for display
            contributions = []
            for feature, value in explanation['contributions'].items():
                contributions.append({
                    'feature': feature,
                    'value': data[feature],
                    'contribution': value,
                    'importance': explanation['feature_importance'][feature]
                })
            
            # Render result
            return render_template(
                'result.html',
                prediction=prediction,
                probability=probability * 100,  # Convert to percentage
                contributions=contributions,
                data=data
            )
        
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction."""
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if all features are present
        missing_features = [f for f in FEATURE_NAMES if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400
        
        # Get prediction and explanation
        explanation = get_prediction_explanation(data)
        
        # Return result
        return jsonify({
            'prediction': explanation['prediction'],
            'probability': explanation['probability'],
            'contributions': explanation['contributions'],
            'feature_importance': explanation['feature_importance']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

if __name__ == '__main__':
    # Create static/images directory if it doesn't exist
    os.makedirs(os.path.join(app.static_folder, 'images'), exist_ok=True)
    
    # Run the app
    app.run(host=HOST, port=PORT, debug=DEBUG)
