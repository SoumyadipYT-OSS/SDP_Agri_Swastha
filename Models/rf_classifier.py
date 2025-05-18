import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_rf_classifier(data_path='../Datasets/dataset_1.csv'):
    # Load and prepare the data
    df = pd.read_csv(data_path)
    X = df.drop('Output', axis=1)
    y = df['Output']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Print performance metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_path = 'soil_fertility_classifier.joblib'
    joblib.dump(rf_classifier, model_path)
    print(f"\nModel saved to {model_path}")
    
    return rf_classifier

def predict_soil_fertility(model, input_data):
    """
    Make predictions using the trained model
    input_data should be a dictionary or pandas DataFrame with keys/columns:
    'N', 'P', 'K', 'EC', 'Fe'
    """
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    return prediction[0], probabilities[0]

if __name__ == "__main__":
    # Train the model
    model = train_rf_classifier()
    
    # Example prediction
    sample_input = {
        'N': 200,
        'P': 8.0,
        'K': 500,
        'EC': 0.5,
        'Fe': 0.6
    }
    
    prediction, probabilities = predict_soil_fertility(model, sample_input)
    print(f"\nSample Prediction:")
    print(f"Input: {sample_input}")
    print(f"Predicted Class: {prediction}")
    print(f"Class Probabilities: {probabilities}")
