import pandas as pd

# Load your dataset
data = pd.read_csv("Datasets\synthetic_dataset.csv")

# Take a peek at the first few rows
print(data.head())

# Optionally, check the summary
print(data.describe())




# Data Preprocessing
from sklearn.model_selection import train_test_split

# Features: all columns except the target, and target column ('Output')
X = data[['N', 'P', 'K', 'EC', 'Fe']]
y = data['Output']

# Split into training and test sets (80% training; 20% test here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize and train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model performance on the test set
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Convert the Trained Model to ONNX
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define the input type.
# There are 5 features (N, P, K, EC, Fe). The name 'float_input' is arbitrary but must be consistent.
initial_type = [('float_input', FloatTensorType([None, 5]))]

# Convert the model
onnx_model = convert_sklearn(rf, initial_types=initial_type)

# Save the ONNX model to a file
with open("rfc.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model conversion completed and model saved as 'rfc.onnx'.")



