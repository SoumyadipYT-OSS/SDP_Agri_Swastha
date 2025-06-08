import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

def create_and_save_models():
    print("Loading dataset...")
    df = pd.read_csv('../Datasets/dataset_1.csv')
    X = df.drop('Output', axis=1)
    y = df['Output']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # KNN Model
    print("\nTraining KNN model...")
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ])
    knn_pipeline.fit(X_train, y_train)
    joblib.dump(knn_pipeline, 'new_knn_classifier.joblib')
    
    # SVM Model
    print("\nTraining SVM model...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)
    joblib.dump(svm_pipeline, 'new_svm_classifier.joblib')
    
    # Random Forest Model
    print("\nTraining Random Forest model...")
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    joblib.dump(rf_pipeline, 'new_rf_classifier.joblib')
    
    print("\nAll models have been trained and saved as .joblib files.")

if __name__ == "__main__":
    create_and_save_models()
