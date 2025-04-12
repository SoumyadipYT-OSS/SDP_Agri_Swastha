#!/usr/bin/env python3
# Soil Fertility Classification using Neural Network
# Created on: April 12, 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import keras
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras._tf_keras.keras.utils import to_categorical
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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
        n_samples = 500
        
        # Define soil parameters as features - focusing on N, P, K, EC, Fe
        nitrogen = np.random.uniform(10, 150, n_samples)  # ppm
        phosphorus = np.random.uniform(5, 100, n_samples)  # ppm
        potassium = np.random.uniform(50, 300, n_samples)  # ppm
        electrical_conductivity = np.random.uniform(0.2, 4.0, n_samples)  # dS/m
        iron = np.random.uniform(2.0, 25.0, n_samples)  # ppm
        
        # Create feature matrix
        X = pd.DataFrame({
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'electrical_conductivity': electrical_conductivity,
            'iron': iron
        })
        
        # Generate fertility classes (0: Low, 1: Medium, 2: High)
        # Simplified model: combination of features determines fertility
        y_score = (
            0.3 * (nitrogen - 10) / 140 + 
            0.25 * (phosphorus - 5) / 95 + 
            0.25 * (potassium - 50) / 250 + 
            0.1 * (1 - (electrical_conductivity - 1.5)**2 / 3.0) +  # EC around 1.5 is optimal
            0.1 * (iron - 2) / 23  # Higher iron generally better
        )
        
        # Convert to classes with slightly imbalanced distribution to simulate real-world scenarios
        y = pd.cut(y_score, bins=[-np.inf, 0.35, 0.65, np.inf], labels=[0, 1, 2]).astype(int)
        
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
    
    # Check for missing values
    missing_values = X.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
    
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
    
    # Histogram for each feature
    X.hist(figsize=(15, 10))
    plt.suptitle('Feature Distributions', y=1.02)
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    
    # Pair plots for all features
    plt.figure(figsize=(15, 12))
    sns.pairplot(X_with_target, vars=feature_names, hue='fertility', palette='viridis')
    plt.suptitle('Pair Plots of Soil Features by Fertility Class', y=1.02)
    plt.savefig('pair_plots.png')
    
    print("Data exploration completed. Visualizations saved as PNG files.")

def preprocess_data(X, y):
    """
    Preprocess data for neural network
    
    Parameters:
    X (dataframe): Feature matrix
    y (series): Target labels
    
    Returns:
    X_train, X_test, y_train, y_test: Processed and split data
    scaler: Fitted scaler for future transformations
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # One-hot encode targets for neural network
    y_train_encoded = to_categorical(y_train, num_classes=3)
    y_test_encoded = to_categorical(y_test, num_classes=3)
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler

def build_neural_network(input_dim):
    """
    Build and compile a neural network model
    
    Parameters:
    input_dim (int): Number of input features
    
    Returns:
    model: Compiled neural network model
    """
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(3, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    return model

def train_neural_network(model, X_train, y_train, X_test, y_test):
    """
    Train the neural network model
    
    Parameters:
    model: Neural network model
    X_train, y_train: Training data
    X_test, y_test: Validation data
    
    Returns:
    model: Trained model
    history: Training history
    """
    print("\n===== Model Training =====")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Callbacks for training
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint('best_model.keras', save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    model = load_model('best_model.keras')
    
    return model, history

def plot_training_history(history):
    """
    Plot training history
    
    Parameters:
    history: Training history
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')

def evaluate_model(model, X_test, y_test, y_test_encoded, feature_names):
    """
    Evaluate the neural network model
    
    Parameters:
    model: Trained neural network model
    X_test: Test features
    y_test: Original test labels
    y_test_encoded: One-hot encoded test labels
    feature_names: Names of features
    """
    print("\n===== Model Evaluation =====")
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
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
    
    # Feature importance analysis - using a permutation-based approach
    print("\nFeature importance (approximation based on sensitivity):")
    
    # Get base accuracy
    y_base_pred = model.predict(X_test)
    base_acc = accuracy_score(np.argmax(y_test_encoded, axis=1), np.argmax(y_base_pred, axis=1))
    
    importance_scores = []
    # For each feature, shuffle its values and measure the drop in performance
    for i, feature in enumerate(feature_names):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        
        y_permuted_pred = model.predict(X_test_permuted)
        permuted_acc = accuracy_score(np.argmax(y_test_encoded, axis=1), np.argmax(y_permuted_pred, axis=1))
        
        # Importance = drop in performance
        importance = base_acc - permuted_acc
        importance_scores.append(importance)
        print(f"- {feature}: {importance:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(importance_scores)
    plt.barh([feature_names[i] for i in sorted_idx], [importance_scores[i] for i in sorted_idx])
    plt.xlabel('Drop in Accuracy (Importance)')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

def save_model(model, scaler, feature_names):
    """
    Save the trained model and preprocessing components
    
    Parameters:
    model: Trained neural network model
    scaler: Fitted scaler
    feature_names (list): Names of features
    """
    print("\n===== Saving Model =====")
    
    # Create models directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    try:
        # Save the model
        model.save('model/soil_fertility_nn_model.keras')
        print("Neural network model saved as 'model/soil_fertility_nn_model.keras'")
        
        # Save scaler
        np.save('model/scaler.npy', [scaler.mean_, scaler.scale_])
        print("Scaler parameters saved")
        
        # Save feature names
        with open('model/features.txt', 'w') as f:
            f.write('\n'.join(feature_names))
        print("Feature names saved as 'model/features.txt'")
        
        print("Model and associated components saved successfully.")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def load_saved_model():
    """
    Load the saved model and preprocessing components
    
    Returns:
    model: Loaded model
    scaler: Reconstructed scaler
    feature_names: List of feature names
    """
    try:
        # Load model
        model = load_model('model/soil_fertility_nn_model.keras')
        
        # Load scaler parameters
        scaler_params = np.load('model/scaler.npy', allow_pickle=True)
        scaler = StandardScaler()
        scaler.mean_ = scaler_params[0]
        scaler.scale_ = scaler_params[1]
        
        # Load feature names
        with open('model/features.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
            
        return model, scaler, feature_names
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def make_prediction(model, scaler, feature_names, input_data=None):
    """
    Make a prediction on new soil data
    
    Parameters:
    model: Trained neural network model
    scaler: Fitted scaler
    feature_names (list): Names of features
    input_data (dict): Dictionary of feature values
    
    Returns:
    prediction (int): Predicted fertility class
    """
    fertility_classes = ['Low', 'Medium', 'High']
    
    if input_data is None:
        # Example input with N, P, K, EC, Fe
        input_data = {
            'nitrogen': 85.0,
            'phosphorus': 55.0,
            'potassium': 190.0,
            'electrical_conductivity': 1.5,
            'iron': 15.0
        }
    
    # Check if all expected features are in the input
    for feature in feature_names:
        if feature not in input_data:
            print(f"Warning: Feature '{feature}' missing from input data. Using default value of 0.")
            input_data[feature] = 0.0
    
    # Create a dataframe from input and ensure correct feature order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]  # Ensure correct order
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_proba = model.predict(input_scaled)[0]
    predicted_class = np.argmax(prediction_proba)
    
    print("\n===== Soil Fertility Prediction =====")
    print(f"Input soil data: {input_data}")
    print(f"Predicted fertility class: {fertility_classes[predicted_class]} (class {predicted_class})")
    
    for i, fertility_class in enumerate(fertility_classes):
        print(f"Probability of {fertility_class} fertility: {prediction_proba[i]:.4f}")
    
    return predicted_class

def main():
    """Main function to run the soil fertility classification"""
    print("===== Soil Fertility Classification using Neural Network =====")
    print("This application predicts soil fertility based on key nutrients and properties:")
    print("- N (Nitrogen)")
    print("- P (Phosphorus)")
    print("- K (Potassium)")
    print("- EC (Electrical Conductivity)")
    print("- Fe (Iron)")
    
    # Check if model exists
    if os.path.exists('model/soil_fertility_nn_model.keras'):
        print("\nFound existing model. Would you like to:")
        choice = input("1. Use existing model for prediction\n2. Train a new model\nEnter choice (1/2): ").strip()
        
        if choice == '1':
            # Load model and make prediction
            model, scaler, feature_names = load_saved_model()
            if model is not None:
                make_prediction(model, scaler, feature_names)
            return
    
    # Ask for data file or use synthetic data
    data_file = input("\nEnter path to soil data CSV file (press Enter to use synthetic data): ").strip()
    if not data_file:
        data_file = None
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data(data_file)
    if X is None:
        print("Failed to load data. Exiting.")
        return
    
    # Explore data
    explore_data(X, y, feature_names)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Build neural network
    model = build_neural_network(X_train.shape[1])
    
    # Train model
    model, history = train_neural_network(model, X_train, y_train, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Get original y_test (not encoded)
    y_test_original = np.argmax(y_test, axis=1)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test_original, y_test, feature_names)
    
    # Save model
    save_model(model, scaler, feature_names)
    
    # Make a prediction
    make_prediction(model, scaler, feature_names)
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()