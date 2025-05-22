import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def train_logistic_regression_model(data_path='../Datasets/synthetic_dataset.csv'):
    """
    Train a Logistic Regression model on the synthetic dataset for soil fertility classification
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
        
    Returns:
    --------
    LogisticRegression
        The trained logistic regression model
    """
    # Load and prepare the data
    print("Loading data from", data_path)
    df = pd.read_csv(data_path)
    
    # Display basic information about the dataset
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print("\nFeature Distribution:")
    print(df.describe())
    
    # Extract features and target
    X = df.drop('Output', axis=1)
    y = df['Output']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    scaler_path = 'lr_feature_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Initialize and train the model
    lr_classifier = LogisticRegression(
        C=1.0,                # Inverse of regularization strength
        solver='lbfgs',       # Algorithm to use in the optimization problem
        max_iter=1000,        # Maximum number of iterations
        penalty='l2',         # L2 regularization
        random_state=42       # For reproducibility
    )
    
    print("\nTraining logistic regression model...")
    lr_classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr_classifier.predict(X_test_scaled)
    
    # Print performance metrics
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(lr_classifier.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save feature importance to CSV
    importance_path = 'lr_feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")
    
    # Save the model
    model_path = 'soil_fertility_lr_classifier.joblib'
    joblib.dump(lr_classifier, model_path)
    print(f"Model saved to {model_path}")
    
    # Convert to ONNX format
    convert_to_onnx(lr_classifier, X.shape[1])
    
    return lr_classifier

def convert_to_onnx(model, input_dim):
    """
    Convert a scikit-learn model to ONNX format
    
    Parameters:
    -----------
    model : sklearn model
        The trained model to convert
    input_dim : int
        Number of input features
    """
    # Ensure the trained_models directory exists
    os.makedirs('trained_models', exist_ok=True)
    
    # Define the ONNX model name
    onnx_model_path = os.path.join('trained_models', 'soil_fertility_lr_classifier.onnx')
    
    # Define the input type for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, input_dim]))]
    
    # Convert the model to ONNX
    print("\nConverting model to ONNX format...")
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Save the ONNX model
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"ONNX model saved to {onnx_model_path}")

def predict_soil_fertility(model, input_data, scaler_path='lr_feature_scaler.joblib'):
    """
    Make predictions using the trained logistic regression model
    
    Parameters:
    -----------
    model : LogisticRegression
        The trained model
    input_data : dict or DataFrame
        Input data containing features: 'N', 'P', 'K', 'EC', 'Fe'
    scaler_path : str
        Path to the saved scaler
        
    Returns:
    --------
    tuple
        (prediction, probabilities)
    """
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Load the scaler
    scaler = joblib.load(scaler_path)
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    return prediction[0], probabilities[0]

if __name__ == "__main__":
    # Train and save the model
    model = train_logistic_regression_model()
    
    # Example prediction
    sample_input = {
        'N': 250,
        'P': 10,
        'K': 500,
        'EC': 0.5,
        'Fe': 3.0
    }
    
    prediction, probabilities = predict_soil_fertility(model, sample_input)
    
    print("\nExample Prediction:")
    print(f"Input: {sample_input}")
    print(f"Prediction: {'Fertile' if prediction == 1 else 'Not Fertile'}")
    print(f"Probability: {probabilities[prediction]:.4f}")
