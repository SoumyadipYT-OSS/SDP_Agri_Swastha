# Soil Fertility Classification using Neural Networks

This project implements a deep learning neural network model to classify soil fertility levels based on key soil nutrients and properties.

## Overview

The model classifies soil fertility into three categories:
- **Low**: Soil with poor fertility characteristics
- **Medium**: Soil with moderate fertility characteristics
- **High**: Soil with good fertility characteristics

## Features Used

The model uses the following key soil parameters as features:
- **N (Nitrogen content)** (ppm): Essential macronutrient for plant growth, protein synthesis
- **P (Phosphorus content)** (ppm): Critical for energy transfer, root development, flowering
- **K (Potassium content)** (ppm): Regulates water usage, disease resistance, and overall plant health
- **EC (Electrical Conductivity)** (dS/m): Indicates salt concentration, optimal around 1.5 dS/m
- **Fe (Iron content)** (ppm): Important micronutrient for chlorophyll production and enzyme function

## Model Architecture

The project uses a deep neural network built with TensorFlow/Keras with the following architecture:
- Input layer with 5 neurons (one for each soil parameter)
- Multiple hidden layers with batch normalization and dropout for regularization
- Output layer with 3 neurons (one for each fertility class)
- ReLU activation for hidden layers, Softmax activation for output layer
- Adam optimizer with categorical cross-entropy loss

## Requirements

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the script:
```
python soil_fertility_classification.py
```

### Using your own data

Prepare your data CSV file with the following columns:
- nitrogen
- phosphorus
- potassium
- electrical_conductivity
- iron
- target (0: Low, 1: Medium, 2: High)

The target column should be the last column.

### Using synthetic data

If you don't have your own dataset, the application will generate synthetic data for demonstration purposes.

## Output

The script will:
1. Load/generate soil data
2. Perform exploratory data analysis with visualizations
3. Train a neural network model with early stopping and learning rate reduction
4. Evaluate the model's performance
5. Calculate feature importance using permutation technique
6. Save the trained model
7. Make a sample prediction

## Files generated

- `correlation_matrix.png`: Correlation between features
- `feature_by_class.png`: Distribution of features by fertility class
- `feature_distributions.png`: Histograms of feature distributions
- `pair_plots.png`: Pair plots of features
- `training_history.png`: Neural network training history (accuracy and loss)
- `confusion_matrix.png`: Model evaluation confusion matrix
- `feature_importance.png`: Feature importance visualization
- `model/soil_fertility_nn_model.keras`: Serialized trained model
- `model/scaler.npy`: Feature scaling parameters
- `model/features.txt`: Feature names list

## Implementation Notes

- The model uses batch normalization and dropout to prevent overfitting
- Early stopping is implemented to optimize training time
- Learning rate reduction on plateau helps with convergence
- Permutation-based feature importance analysis helps identify key soil parameters
- Data preprocessing includes standardization and one-hot encoding

## About Neural Network Approach

Neural networks offer several advantages for soil fertility classification:
1. Ability to capture complex, non-linear relationships between soil parameters
2. Robust performance with properly tuned hyperparameters
3. Can handle interactions between features automatically
4. Provides probability distribution across all classes