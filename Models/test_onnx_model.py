import numpy as np
import pandas as pd
import onnxruntime as rt
from sklearn.preprocessing import StandardScaler
import joblib

def test_onnx_model(input_data=None):
    """
    Test the ONNX model with sample input data
    
    Parameters:
    -----------
    input_data : dict, optional
        Sample input data containing features: 'N', 'P', 'K', 'EC', 'Fe'
    """
    # Use default data if none is provided
    if input_data is None:
        input_data = {
            'N': 250,
            'P': 10,
            'K': 500,
            'EC': 0.5,
            'Fe': 3.0
        }
    
    # Convert to DataFrame for easier handling
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = pd.DataFrame(input_data)
    
    # Load the scaler
    scaler = joblib.load('lr_feature_scaler.joblib')
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Load ONNX model
    print(f"Loading ONNX model from trained_models/soil_fertility_lr_classifier.onnx...")
    sess = rt.InferenceSession('trained_models/soil_fertility_lr_classifier.onnx')
    
    # Get input name
    input_name = sess.get_inputs()[0].name
    
    # Run prediction
    pred_onx = sess.run(None, {input_name: input_scaled.astype(np.float32)})
    
    # Process results
    predicted_class = pred_onx[0][0]
    class_probabilities = pred_onx[1][0]
    
    print(f"\nONNX Model Prediction Results:")
    print(f"Input data: {input_data}")
    print(f"Predicted class: {predicted_class}")
    print(f"Class label: {'Fertile' if predicted_class == 1 else 'Not Fertile'}")
    print(f"Class probabilities: {class_probabilities}")
    
    return predicted_class, class_probabilities

if __name__ == "__main__":
    # Test with sample input
    test_onnx_model()
    
    # Test another sample
    test_onnx_model({
        'N': 100,
        'P': 6.5,
        'K': 350,
        'EC': 0.7,
        'Fe': 1.8
    })
