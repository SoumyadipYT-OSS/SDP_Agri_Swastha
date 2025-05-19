#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logistic Regression Model for Soil Fertility Prediction
SDP_Agri_Swastha Project

This script trains a logistic regression model to predict soil fertility 
based on various soil parameters like Nitrogen (N), Phosphorus (P), 
Potassium (K), Electrical Conductivity (EC), and Iron (Fe).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filepath):
    """Load the dataset and perform initial exploration"""
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nData information:")
    print(data.info())
    print("\nSummary statistics:")
    print(data.describe())
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing values:")
    print(missing_values)
    
    return data

def preprocess_data(data):
    """Perform data preprocessing"""
    print("\n--- Preprocessing Data ---")
    
    # Check class distribution
    print("\nClass distribution:")
    print(data['Output'].value_counts())
    print(f"Class distribution percentage: \n{data['Output'].value_counts(normalize=True) * 100}")
    
    # Separate features and target
    X = data.drop('Output', axis=1)
    y = data['Output']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a logistic regression model with hyperparameter tuning"""
    print("\n--- Training Logistic Regression Model ---")
    
    # Create a pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Define hyperparameters to tune
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
        'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    
    # Create parameter combinations that are compatible
    valid_params = []
    for p in param_grid['classifier__penalty']:
        for s in param_grid['classifier__solver']:
            for c in param_grid['classifier__C']:
                # Skip invalid combinations
                if p == 'l1' and s in ['newton-cg', 'sag']:
                    continue
                if p == 'elasticnet' and s != 'saga':
                    continue
                if p is None and s in ['liblinear']:
                    continue
                valid_params.append({'classifier__C': c, 'classifier__penalty': p, 'classifier__solver': s})
    
    # Run grid search with cross-validation
    print("Performing grid search with 5-fold cross-validation...")
    grid_search = GridSearchCV(pipeline, valid_params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """Evaluate the trained model"""
    print("\n--- Model Evaluation ---")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Training accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # ROC curve and AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Feature importance
    # For logistic regression, feature importance can be derived from the coefficients
    if hasattr(model[-1], 'coef_'):
        coefficients = model[-1].coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': coefficients,
            'Absolute Importance': np.abs(coefficients)
        }).sort_values(by='Absolute Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\n5-fold Cross-validation Scores: {cv_scores}")
    print(f"Mean CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    
    return {
        'accuracy': accuracy,
        'train_accuracy': train_accuracy,
        'conf_matrix': conf_matrix,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'y_prob': y_prob,
        'feature_importance': feature_importance if hasattr(model[-1], 'coef_') else None
    }

def save_model(model, model_path):
    """Save the trained model to disk"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

def main():
    """Main function to run the entire pipeline"""
    print("=== Soil Fertility Prediction Model ===")
    
    # Define file paths
    data_path = "../Datasets/dataset_1.csv"
    model_path = "trained_models/logistic_regression_model.pkl"
    
    # Load and explore the data
    data = load_data(data_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluation_results = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Save the model
    save_model(model, model_path)
    
    print("\n=== Model Training Complete ===")
    
    return model, evaluation_results, (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
