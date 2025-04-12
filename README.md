# Soil Fertility Classification using KNN

This project implements a K-Nearest Neighbors (KNN) machine learning model to classify soil fertility levels based on various soil parameters.

## Overview

The model classifies soil fertility into three categories:
- **Low**: Soil with poor fertility characteristics
- **Medium**: Soil with moderate fertility characteristics
- **High**: Soil with good fertility characteristics

## Features Used

The model uses the following soil parameters as features:
- Nitrogen content (ppm)
- Phosphorus content (ppm)
- Potassium content (ppm)
- Organic matter (%)
- pH level
- Cation Exchange Capacity (meq/100g)
- Soil moisture (%)

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

Prepare your data CSV file with soil parameters and a target column. The target should be the last column with values 0 (Low fertility), 1 (Medium fertility), or 2 (High fertility).

When prompted, provide the path to your CSV file.

### Using synthetic data

If you don't have your own dataset, the application will generate synthetic data for demonstration purposes.

## Output

The script will:
1. Load/generate soil data
2. Perform exploratory data analysis with visualizations
3. Train a KNN model with hyperparameter tuning
4. Evaluate the model's performance
5. Save the trained model
6. Make a sample prediction

## Files generated

- `correlation_matrix.png`: Correlation between features
- `feature_by_class.png`: Distribution of features by fertility class
- `pair_plots.png`: Pair plots of features
- `confusion_matrix.png`: Model evaluation confusion matrix
- `decision_boundary.png`: Decision boundary visualization (for two selected features)
- `soil_fertility_knn_model.pkl`: Serialized trained model
- `model_features.txt`: Feature names list

## Implementation Notes

- The model uses a pipeline with data scaling for better performance
- Grid search cross-validation is used to find optimal hyperparameters
- Standard ML evaluation metrics are provided (accuracy, precision, recall, F1-score)

## About KNN Algorithm

The K-Nearest Neighbors algorithm classifies samples based on the majority class of their k nearest neighbors in feature space. It's a simple but powerful non-parametric method that can work well for multi-class classification problems like this one.