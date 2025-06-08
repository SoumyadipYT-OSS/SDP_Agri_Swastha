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
from sklearn.metrics import classification_report

def create_and_save_models():
    print("Loading datasets...")
    # Load both original and synthetic data
    df_original = pd.read_csv('../Datasets/dataset_1.csv')
    df_synthetic = pd.read_csv('../Datasets/synthetic_dataset.csv')
    
    # Combine datasets with a flag to track source
    df_original['is_synthetic'] = 0
    df_synthetic['is_synthetic'] = 1
    df = pd.concat([df_original, df_synthetic], ignore_index=True)
    
    # Print class distribution
    print("\nClass distribution in combined dataset:")
    class_dist = df['Output'].value_counts(normalize=True)
    for cls in sorted(class_dist.index):
        print(f"Class {cls}: {len(df[df['Output'] == cls])} samples ({class_dist[cls]*100:.2f}%)")
    
    # Prepare features and target
    X = df.drop(['Output', 'is_synthetic'], axis=1)
    y = df['Output']
    
    # Split data ensuring both original and synthetic samples are represented
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate balanced class weights
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = {
        i: total_samples / (len(class_counts) * count)
        for i, count in enumerate(class_counts)
    }
    print("\nClass weights:", class_weights)
    
    # KNN Model with weighted voting
    print("\nTraining KNN model...")
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',  # Use distance weighting
            metric='manhattan'  # Try manhattan distance
        ))
    ])
    knn_pipeline.fit(X_train, y_train)
    joblib.dump(knn_pipeline, 'new_knn_classifier.joblib')
    
    # SVM Model with enhanced class weights
    print("\nTraining SVM model...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight=class_weights,
            C=10.0,  # Increase C to allow more complex decision boundaries
            gamma='scale'
        ))
    ])
    svm_pipeline.fit(X_train, y_train)
    joblib.dump(svm_pipeline, 'new_svm_classifier.joblib')
    
    # Random Forest with enhanced class weights and tuned parameters
    print("\nTraining Random Forest model...")
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=500,  # More trees
            max_depth=None,    # Allow full depth
            min_samples_split=10,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight=class_weights,
            random_state=42,
            bootstrap=True,
            oob_score=True    # Use out-of-bag score
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    
    # Print model performance metrics
    y_pred = rf_pipeline.predict(X_test)
    print("\nRandom Forest Performance:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(rf_pipeline, 'new_rf_classifier.joblib')
    print("\nAll models have been trained and saved as .joblib files.")

if __name__ == "__main__":
    create_and_save_models()
