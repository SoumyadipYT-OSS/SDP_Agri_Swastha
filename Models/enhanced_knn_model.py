"""
Enhanced KNN Model for Soil Fertility Classification
This module provides an enhanced KNN-based soil fertility classifier that uses feature engineering,
selection, and ensemble methods to improve prediction accuracy for soil fertility levels based on
soil parameters (N, P, K, EC, Fe).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

class EnhancedSoilFertilityKNN:
    def __init__(self, model_path=None, poly_degree=2, n_features=20):
        """
        Initialize the enhanced KNN-based soil fertility classifier.
        
        Args:
            model_path (str, optional): Path to a saved model file
            poly_degree (int): Degree of polynomial features (default: 2)
            n_features (int): Number of top features to select (default: 20)
        """
        if model_path:
            self.pipeline = joblib.load(model_path)
        else:
            # Initialize preprocessing components
            self.scaler = StandardScaler()
            self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            self.selector = SelectKBest(score_func=f_classif, k=n_features)
            self.smote = SMOTE(random_state=42)
            
            # Create an ensemble of KNN classifiers with different configurations
            self.estimators = [
                ('knn1', KNeighborsClassifier(
                    n_neighbors=5, weights='distance', metric='minkowski', p=2)),
                ('knn2', KNeighborsClassifier(
                    n_neighbors=7, weights='distance', metric='manhattan')),
                ('knn3', KNeighborsClassifier(
                    n_neighbors=9, weights='uniform', metric='chebyshev'))
            ]
            self.ensemble = VotingClassifier(
                estimators=self.estimators,
                voting='soft'  # Use probability estimates for voting
            )
            
            # Create the full pipeline
            self.pipeline = ImbPipeline([
                ('scaler', self.scaler),
                ('poly', self.poly),
                ('selector', self.selector),
                ('smote', self.smote),
                ('ensemble', self.ensemble)
            ])
    
    def train(self, X, y):
        """
        Train the enhanced KNN model with feature engineering, selection, and class balancing.
        
        Args:
            X (array-like): Feature matrix containing soil parameters
            y (array-like): Target variable (fertility classes)
        
        Returns:
            self: The trained model instance
        """
        # Fit the entire pipeline
        self.pipeline.fit(X, y)
        
        # Store feature names for interpretability
        self.feature_names = []
        if hasattr(self.poly, 'get_feature_names_out'):
            poly_features = self.poly.get_feature_names_out(input_features=['N', 'P', 'K', 'EC', 'Fe'])
            selected_features_mask = self.selector.get_support()
            self.feature_names = poly_features[selected_features_mask].tolist()
        
        return self
    
    def predict(self, X):
        """
        Make predictions for the given soil parameters.
        
        Args:
            X (array-like): Feature matrix containing soil parameters
            
        Returns:
            array: Predicted fertility classes
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for each fertility class.
        
        Args:
            X (array-like): Feature matrix containing soil parameters
            
        Returns:
            array: Probability estimates for each class
        """
        return self.pipeline.predict_proba(X)
    
    def save_model(self, model_path):
        """
        Save the complete pipeline to a file.
        
        Args:
            model_path (str): Path to save the model pipeline
        """
        joblib.dump(self.pipeline, model_path)
    
    def get_feature_importance(self, X, y):
        """
        Calculate feature importance scores.
        
        Args:
            X (array-like): Feature matrix containing soil parameters
            y (array-like): Target variable (fertility classes)
            
        Returns:
            DataFrame: Feature importance scores
        """
        from sklearn.inspection import permutation_importance
        
        # Get importance scores
        result = permutation_importance(
            self.pipeline, X, y,
            n_repeats=10,
            random_state=42
        )
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': ['N', 'P', 'K', 'EC', 'Fe'],
            'importance': result.importances_mean
        }).sort_values('importance', ascending=False)
        
        return importance_df

def main():
    """Example usage of the EnhancedSoilFertilityKNN class."""
    # Load sample data
    print("Loading dataset...")
    data = pd.read_csv('Datasets/dataset_1.csv')
    X = data.drop('Output', axis=1)
    y = data['Output']
    
    # Create and train model
    print("\nTraining enhanced KNN model...")
    model = EnhancedSoilFertilityKNN()
    model.train(X, y)
    
    # Calculate and display feature importance
    print("\nCalculating feature importance...")
    importance_df = model.get_feature_importance(X, y)
    print("\nFeature Importance Ranking:")
    print(importance_df)
    
    # Example prediction
    print("\nMaking example prediction...")
    sample = pd.DataFrame({
        'N': [200],
        'P': [8.0],
        'K': [500],
        'EC': [0.5],
        'Fe': [0.6]
    })
    
    prediction = model.predict(sample)
    probabilities = model.predict_proba(sample)
    
    print("\nExample Prediction:")
    print("Input Features:")
    print(sample)
    print(f"\nPredicted class: {prediction[0]}")
    print("\nClass probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"Class {i}: {prob:.4f}")
    
    # Save the model
    print("\nSaving model...")
    model.save_model('Models/enhanced_soil_fertility_knn.joblib')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
