# Diabetes Prediction Model

A machine learning application that predicts diabetes risk based on health parameters, developed for a Novo Nordisk internship application.

## Project Overview

This project implements a complete machine learning pipeline for diabetes prediction:

- **Data Preprocessing**: Handling missing values, feature scaling, and data preparation
- **Model Training**: Implementation of Random Forest classifier with hyperparameter tuning
- **Model Evaluation**: Performance assessment using accuracy, ROC AUC, and classification reports
- **Feature Importance**: Analysis of which health parameters contribute most to diabetes risk
- **Web Application**: Flask-based interface for easy interaction with the model
- **Explainable AI**: Visualization of how each feature contributes to individual predictions

## Project Structure

```
DiabetesPredictionModel/
├── data/                  # Data directory
│   └── diabetes.csv       # Diabetes dataset (downloaded automatically)
├── models/                # Saved models directory
│   ├── random_forest_model.pkl  # Trained Random Forest model
│   └── scaler.pkl         # Fitted StandardScaler
├── scripts/               # Python modules
│   ├── config.py          # Configuration settings
│   ├── preprocess.py      # Data preprocessing functions
│   ├── train_model.py     # Model training and evaluation
│   └── inference.py       # Prediction functions
├── static/                # Static files for web app
│   ├── css/               # CSS stylesheets
│   │   └── style.css      # Custom styles
│   └── images/            # Generated images
│       ├── confusion_matrix.png    # Confusion matrix visualization
│       └── feature_importance.png  # Feature importance visualization
├── templates/             # HTML templates
│   ├── index.html         # Home page with input form
│   ├── result.html        # Prediction results page
│   └── about.html         # Project information page
├── app.py                 # Flask web application
├── run_training.py        # Script to run the training pipeline
├── test_inference.py      # Script to test inference functionality
└── requirements.txt       # Project dependencies
```

## Dataset

The model uses the Pima Indians Diabetes Database, which includes the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (μU/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic influence)
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1) indicating whether the patient has diabetes

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DiabetesPredictionModel.git
cd DiabetesPredictionModel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python run_training.py
```

4. Run the web application:
```bash
python app.py
```

5. Access the application in your browser at `http://localhost:5000`

## Model Performance

The Random Forest model achieves:
- **Accuracy**: ~76%
- **ROC AUC**: ~84%
- **Precision**: ~83% for class 0 (no diabetes), ~65% for class 1 (diabetes)
- **Recall**: ~79% for class 0 (no diabetes), ~71% for class 1 (diabetes)

## Feature Importance

The most important features for predicting diabetes risk are:
1. Glucose level
2. Age
3. Insulin level
4. Diabetes Pedigree Function
5. Number of Pregnancies

## API Usage

The application provides a REST API endpoint for programmatic access:

```python
import requests
import json

# Sample data
data = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}

# Make prediction request
response = requests.post('http://localhost:5000/api/predict', 
                         json=data,
                         headers={'Content-Type': 'application/json'})

# Parse response
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
print("Feature contributions:")
for feature, contribution in result['contributions'].items():
    print(f"  {feature}: {contribution}")
```

## Deployment

The application can be deployed to various platforms:

### Local Deployment
Run the Flask application locally:
```bash
python app.py
```

### Docker Deployment
1. Build the Docker image:
```bash
docker build -t diabetes-prediction-model .
```

2. Run the container:
```bash
docker run -p 5000:5000 diabetes-prediction-model
```

### Cloud Deployment
The application can be deployed to cloud platforms like Heroku, AWS, or Google Cloud Platform.

## Limitations and Future Improvements

- **Limitations**:
  - Trained on a specific population (Pima Indian heritage females)
  - Limited sample size (~768 records)
  - Some features may have measurement errors or missing values

- **Future Improvements**:
  - Incorporate more diverse training data
  - Implement ensemble methods combining multiple models
  - Add more advanced feature engineering
  - Develop a mobile application interface

## Disclaimer

This application is for educational and demonstration purposes only. It is not intended to provide medical advice, diagnosis, or treatment. The predictions made by this model should not replace consultation with healthcare professionals.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Pima Indians Diabetes Database from the UCI Machine Learning Repository
- Scikit-learn, Pandas, NumPy, and other open-source libraries used in this project
