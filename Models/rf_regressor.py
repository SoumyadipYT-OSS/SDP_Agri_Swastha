import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_rf_regressor(data_path='../Datasets/dataset_1.csv'):
    # Load and prepare the data
    df = pd.read_csv(data_path)
    X = df.drop('Output', axis=1)
    y = df['Output']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    rf_regressor.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_regressor.predict(X_test)
    
    # Print performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save the model
    model_path = 'soil_fertility_regressor.joblib'
    joblib.dump(rf_regressor, model_path)
    print(f"\nModel saved to {model_path}")
    
    return rf_regressor

def predict_soil_fertility_score(model, input_data):
    """
    Make predictions using the trained model
    input_data should be a dictionary or pandas DataFrame with keys/columns:
    'N', 'P', 'K', 'EC', 'Fe'
    Returns a continuous value representing soil fertility
    """
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    prediction = model.predict(input_data)
    
    return prediction[0]

if __name__ == "__main__":
    # Train the model
    model = train_rf_regressor()
    
    # Example prediction
    sample_input = {
        'N': 200,
        'P': 8.0,
        'K': 500,
        'EC': 0.5,
        'Fe': 0.6
    }
    
    prediction = predict_soil_fertility_score(model, sample_input)
    print(f"\nSample Prediction:")
    print(f"Input: {sample_input}")
    print(f"Predicted Fertility Score: {prediction:.4f}")
