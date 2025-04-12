#!/usr/bin/env python3
# Soil Fertility Classification using KNN
# Created on: April 12, 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data(file_path=None):
    """
    Load and prepare soil fertility data for classification.
    If no file_path is provided, generate synthetic data for demonstration.
    
    Parameters:
    file_path (str): Path to the CSV file containing soil data
    
    Returns:
    X (dataframe): Feature matrix
    y (series): Target labels
    feature_names (list): Names of features
    """
    if file_path is None:
        print("No data file provided. Generating synthetic soil fertility data...")
        # Generate synthetic data for demonstration
        n_samples = 300
        
        # Define soil parameters as features
        nitrogen = np.random.uniform(10, 150, n_samples)  # ppm
        phosphorus = np.random.uniform(5, 100, n_samples)  # ppm
        potassium = np.random.uniform(50, 300, n_samples)  # ppm
        organic_matter = np.random.uniform(0.5, 10, n_samples)  # %
        ph = np.random.uniform(4.0, 8.5, n_samples)
        cation_exchange_capacity = np.random.uniform(5, 25, n_samples)  # meq/100g
        soil_moisture = np.random.uniform(10, 45, n_samples)  # %
        
        # Create feature matrix
        X = pd.DataFrame({
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'organic_matter': organic_matter,
            'ph': ph,
            'cation_exchange_capacity': cation_exchange_capacity,
            'soil_moisture': soil_moisture
        })
        
        # Generate fertility classes (0: Low, 1: Medium, 2: High)
        # Simplified model: higher values generally indicate better fertility
        y_score = (
            0.3 * (nitrogen - 10) / 140 + 
            0.2 * (phosphorus - 5) / 95 + 
            0.15 * (potassium - 50) / 250 + 
            0.15 * (organic_matter - 0.5) / 9.5 +
            0.1 * (1 - abs(ph - 6.5) / 2.25) +  # pH around 6.5 is optimal
            0.05 * (cation_exchange_capacity - 5) / 20 +
            0.05 * (soil_moisture - 10) / 35
        )
        
        # Convert to classes
        y = pd.cut(y_score, bins=3, labels=[0, 1, 2]).astype(int)
        
        feature_names = X.columns.tolist()
        
    else:
        # Load real data from CSV
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}")
            
            # Assume last column is the target
            feature_names = df.columns[:-1].tolist()
            X = df[feature_names]
            y = df[df.columns[-1]]
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None, None
    
    print(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    return X, y, feature_names

def explore_data(X, y, feature_names):
    """
    Perform exploratory data analysis on soil fertility data
    
    Parameters:
    X (dataframe): Feature matrix
    y (series): Target labels
    feature_names (list): Names of features
    """
    print("\n===== Data Exploration =====")
    
    # Basic statistics
    print("\nFeature statistics:")
    print(X.describe())
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Distribution of features by fertility class
    fertility_classes = ['Low', 'Medium', 'High']
    X_with_target = X.copy()
    X_with_target['fertility'] = y
    
    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='fertility', y=feature, data=X_with_target)
        plt.xticks([0, 1, 2], fertility_classes)
        plt.title(f'{feature} by Fertility Class')
        plt.tight_layout()
        plt.savefig(f'{feature}_by_class.png')
    
    # Pair plots for selected features
    if len(feature_names) > 6:
        selected_features = feature_names[:6]  # Limit to first 6 features
    else:
        selected_features = feature_names
    
    plt.figure(figsize=(15, 12))
    sns.pairplot(X_with_target, vars=selected_features, hue='fertility', palette='viridis')
    plt.suptitle('Pair Plots of Soil Features by Fertility Class', y=1.02)
    plt.savefig('pair_plots.png')
    
    print("Data exploration completed. Visualizations saved as PNG files.")

def train_knn_model(X, y):
    """
    Train a KNN model for soil fertility classification
    
    Parameters:
    X (dataframe): Feature matrix
    y (series): Target labels
    
    Returns:
    best_model: Trained KNN model
    X_train, X_test, y_train, y_test: Split data
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print("\n===== Model Training =====")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create a pipeline with scaling and KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    
    # Define hyperparameters grid
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Grid search for best hyperparameters
    print("Performing grid search to find optimal hyperparameters...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    return best_model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the KNN model on test data
    
    Parameters:
    model: Trained KNN model
    X_test (dataframe): Test feature matrix
    y_test (series): Test target labels
    """
    print("\n===== Model Evaluation =====")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Plot decision boundary visualization for two selected features
    # This is a simplified visualization using only two features
    if X_test.shape[1] >= 2:
        visualize_decision_boundary(model, X_test, y_test, 0, 1)
    
def visualize_decision_boundary(model, X, y, feature1_idx=0, feature2_idx=1):
    """
    Visualize decision boundary of the model using two selected features
    
    Parameters:
    model: Trained KNN model
    X (dataframe): Feature matrix
    y (series): Target labels
    feature1_idx (int): Index of first feature for visualization
    feature2_idx (int): Index of second feature for visualization
    """
    # Extract the scaler from the pipeline
    scaler = model.named_steps['scaler']
    knn = model.named_steps['knn']
    
    # Scale the data
    X_scaled = scaler.transform(X)
    
    # Create mesh grid
    x_min, x_max = X_scaled[:, feature1_idx].min() - 1, X_scaled[:, feature1_idx].max() + 1
    y_min, y_max = X_scaled[:, feature2_idx].min() - 1, X_scaled[:, feature2_idx].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Create feature vectors for prediction
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    # Add zeros for other features
    X_mesh_complete = np.zeros((X_mesh.shape[0], X.shape[1]))
    X_mesh_complete[:, feature1_idx] = X_mesh[:, 0]
    X_mesh_complete[:, feature2_idx] = X_mesh[:, 1]
    
    # Predict and reshape
    Z = knn.predict(X_mesh_complete)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    
    # Plot training points
    scatter = plt.scatter(X_scaled[:, feature1_idx], X_scaled[:, feature2_idx], 
                          c=y, edgecolors='k', cmap='viridis')
    
    plt.xlabel(f'Feature {feature1_idx}: {X.columns[feature1_idx]} (scaled)')
    plt.ylabel(f'Feature {feature2_idx}: {X.columns[feature2_idx]} (scaled)')
    plt.title(f'KNN Decision Boundary using {X.columns[feature1_idx]} and {X.columns[feature2_idx]}')
    plt.colorbar(scatter, label='Fertility Class')
    plt.tight_layout()
    plt.savefig('decision_boundary.png')

def save_model(model, feature_names):
    """
    Save the trained model to disk
    
    Parameters:
    model: Trained model to save
    feature_names (list): Names of features
    """
    print("\n===== Saving Model =====")
    try:
        joblib.dump(model, 'soil_fertility_knn_model.pkl')
        print("Model saved as 'soil_fertility_knn_model.pkl'")
        
        # Save feature names
        with open('model_features.txt', 'w') as f:
            f.write('\n'.join(feature_names))
        print("Feature names saved as 'model_features.txt'")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def make_prediction(model, feature_names, input_data=None):
    """
    Make a prediction on new soil data
    
    Parameters:
    model: Trained KNN model
    feature_names (list): Names of features
    input_data (dict): Dictionary of feature values
    
    Returns:
    prediction (int): Predicted fertility class
    """
    fertility_classes = ['Low', 'Medium', 'High']
    
    if input_data is None:
        # Example input
        input_data = {
            'nitrogen': 75.0,
            'phosphorus': 45.0,
            'potassium': 180.0,
            'organic_matter': 4.5,
            'ph': 6.8,
            'cation_exchange_capacity': 15.0,
            'soil_moisture': 25.0
        }
    
    # Create a dataframe from input
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    predicted_class = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    print("\n===== Soil Fertility Prediction =====")
    print(f"Input soil data: {input_data}")
    print(f"Predicted fertility class: {fertility_classes[predicted_class]} (class {predicted_class})")
    print(f"Class probabilities: Low: {probabilities[0]:.2f}, Medium: {probabilities[1]:.2f}, High: {probabilities[2]:.2f}")
    
    return predicted_class

def main():
    """Main function to run the soil fertility classification"""
    print("===== Soil Fertility Classification using KNN =====")
    
    # Ask for data file or use synthetic data
    data_file = input("Enter path to soil data CSV file (press Enter to use synthetic data): ").strip()
    if not data_file:
        data_file = None
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data(data_file)
    if X is None:
        print("Failed to load data. Exiting.")
        return
    
    # Explore data
    explore_data(X, y, feature_names)
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_knn_model(X, y)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, feature_names)
    
    # Make a prediction
    make_prediction(model, feature_names)
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()