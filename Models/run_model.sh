#!/bin/bash

echo "Installing required packages..."
pip install -r ../requirements.txt

echo "Running logistic regression model..."
python logistic_regression_model.py

echo "Done! The model has been trained and saved."
echo "You can now open the Jupyter notebook for detailed analysis:"
echo "logistic_regression_soil_fertility.ipynb"
