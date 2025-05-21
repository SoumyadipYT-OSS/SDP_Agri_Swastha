import joblib
import os
import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define paths
model_dir = "Models/trained_models"
model_path = os.path.join(model_dir, "rf_classifier_synthetic.pkl")
onnx_output_path = os.path.join(model_dir, "rf.onnx")

# Load the scikit-learn model from the .pkl file
loaded_model = joblib.load(model_path)

# Determine the correct input shape dynamically
input_features = loaded_model.n_features_in_
print(f"Detected model input features: {input_features}")

# Define input type dynamically based on the actual model
initial_type = [("float_input", FloatTensorType([None, input_features]))]

# Convert the model to ONNX format
onnx_model = convert_sklearn(loaded_model, initial_types=initial_type)

# Save the converted ONNX model
with open(onnx_output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"ONNX model successfully saved at: {onnx_output_path}")

# Load ONNX model for inference
sess = rt.InferenceSession(onnx_output_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# Example test data for prediction (Ensure it matches model's expected input format)
X_test = np.random.rand(1, input_features).astype(np.float32)  # Adjusted dynamically

# Run inference
pred_onx = sess.run([label_name], {input_name: X_test})[0]
print("ONNX Model Prediction:", pred_onx)