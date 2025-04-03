# Diabetes Prediction Model - User Guide

This guide provides instructions on how to use the Diabetes Prediction Model web application.

## Introduction

The Diabetes Prediction Model is a machine learning application that predicts the risk of diabetes based on various health parameters. It uses a Random Forest algorithm trained on the Pima Indians Diabetes Database to make predictions.

## Using the Web Interface

### Home Page

1. Navigate to the home page at `http://localhost:5000`
2. You'll see a form where you can enter your health information:
   - **Number of Pregnancies**: How many times the person has been pregnant
   - **Glucose Level**: Plasma glucose concentration (mg/dL)
   - **Blood Pressure**: Diastolic blood pressure (mm Hg)
   - **Skin Thickness**: Triceps skin fold thickness (mm)
   - **Insulin Level**: 2-Hour serum insulin (μU/ml)
   - **BMI**: Body mass index (kg/m²)
   - **Diabetes Pedigree Function**: A function that represents genetic influence
   - **Age**: Age in years

3. Enter your values in each field
4. Click the "Predict Diabetes Risk" button

### Results Page

After submitting the form, you'll be taken to the results page which shows:

1. **Prediction Result**: Whether you have a high or low risk of diabetes
2. **Probability**: The probability of having diabetes (as a percentage)
3. **Feature Contributions**: A chart showing how each feature contributed to the prediction
4. **Health Information Table**: A table showing your input values and their contributions

The feature contributions help you understand which factors are most influential in your prediction. Positive contributions (red) increase the likelihood of diabetes, while negative contributions (green) decrease it.

## Understanding the Results

### Prediction Interpretation

- **High Risk of Diabetes**: The model predicts that you have a higher likelihood of diabetes based on your health parameters
- **Low Risk of Diabetes**: The model predicts that you have a lower likelihood of diabetes based on your health parameters

### Feature Contributions

The feature contributions chart shows which health parameters had the most impact on your prediction:

- **Glucose Level**: Often the most important factor; higher levels typically increase diabetes risk
- **BMI**: Higher BMI values generally increase risk
- **Age**: Age can be a significant factor in diabetes risk
- **Diabetes Pedigree Function**: Higher values indicate stronger family history of diabetes
- **Pregnancies**: More pregnancies may be associated with increased risk

## Important Notes

1. **Medical Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice
2. **Data Privacy**: Your health information is processed locally and is not stored or shared
3. **Model Limitations**: The model was trained on a specific population (Pima Indian heritage females) and may not be as accurate for other demographics

## Troubleshooting

If you encounter any issues:

1. **Invalid Input**: Ensure all fields contain valid numerical values
2. **Server Error**: Try refreshing the page or restarting the application
3. **Unexpected Results**: Remember that the model provides a probability, not a definitive diagnosis

## Additional Resources

- **About Page**: Click the "About" link in the navigation bar to learn more about the project
- **GitHub Repository**: Visit the project repository for technical details and source code
- **Medical Resources**: For concerns about diabetes, consult healthcare professionals or visit diabetes.org
