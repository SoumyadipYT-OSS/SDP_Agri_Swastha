import os
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

def convert_model_to_onnx(model_path, model_name, output_path):
    """Convert a scikit-learn model to ONNX format.
    
    Args:
        model_path: Path to the .joblib model file
        model_name: Name of the model (used for ONNX metadata)
        output_path: Path where to save the ONNX model
    """
    print(f"\nConverting {model_name} to ONNX format...")
    print(f"Model path: {model_path}")
    print(f"ONNX path: {output_path}")
    
    try:
        # Load the model
        model = joblib.load(model_path)
        print("Model loaded successfully")
        print(f"Model type: {type(model).__name__}")
        
        # Create initial type for ONNX conversion
        initial_type = [('float_input', FloatTensorType([None, 5]))]  # 5 features: N, P, K, EC, Fe
        
        # Convert to ONNX
        print("Starting ONNX conversion...")
        onx = convert_sklearn(
            model, 
            initial_types=initial_type,
            target_opset=12,  # Compatible opset version
            options={id(model): {'zipmap': False}},  # Disable zipmap to get raw probabilities
            name=model_name
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the model
        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())
        
        print(f"Model converted and saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

def main():
    # Base paths
    models_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(models_dir, "trained_models")
    
    # List of models to convert
    models_to_convert = [
        {
            'input_path': os.path.join(models_dir, 'new_knn_classifier.joblib'),
            'output_path': os.path.join(output_dir, 'new_knn_classifier.onnx'),
            'name': 'new_knn_classifier'
        },
        {
            'input_path': os.path.join(models_dir, 'new_svm_classifier.joblib'),
            'output_path': os.path.join(output_dir, 'new_svm_classifier.onnx'),
            'name': 'new_svm_classifier'
        },
        {
            'input_path': os.path.join(models_dir, 'new_rf_classifier.joblib'),
            'output_path': os.path.join(output_dir, 'new_rf_classifier.onnx'),
            'name': 'new_rf_classifier'
        },
        {
            'input_path': os.path.join(models_dir, 'enhanced_soil_fertility_knn.joblib'),
            'output_path': os.path.join(output_dir, 'soil_fertility_knn_classifier.onnx'),
            'name': 'soil_fertility_knn'
        },
        {
            'input_path': os.path.join(models_dir, 'soil_fertility_classifier.joblib'),
            'output_path': os.path.join(output_dir, 'soil_fertility_classifier.onnx'),
            'name': 'soil_fertility_classifier'
        },
        {
            'input_path': os.path.join(models_dir, 'soil_fertility_lr_classifier.joblib'),
            'output_path': os.path.join(output_dir, 'soil_fertility_lr_classifier.onnx'),
            'name': 'soil_fertility_lr'
        }
    ]
    
    # Convert each model
    for model_info in models_to_convert:
        if os.path.exists(model_info['input_path']):
            convert_model_to_onnx(
                model_info['input_path'],
                model_info['name'],
                model_info['output_path']
            )
        else:
            print(f"\nWarning: Model file not found at {model_info['input_path']}")

if __name__ == "__main__":
    main()
