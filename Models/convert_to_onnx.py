import os
import sys
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from onnxruntime import InferenceSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

def debug_print(*args, **kwargs):
    """Print debug information to stderr"""
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

class SklearnPipelineExtractor:
    """Helper class to extract supported parts from a sklearn pipeline."""
    
    @staticmethod
    def extract_supported_steps(pipeline):
        """Extract steps that are supported by skl2onnx."""
        if not hasattr(pipeline, 'steps'):
            return pipeline
            
        # Known supported steps and their alternatives
        supported_transforms = {
            'StandardScaler': 'StandardScaler',
            'PolynomialFeatures': 'PolynomialFeatures',
            'SelectKBest': None,  # Skip feature selection in ONNX
            'SMOTE': None,  # Skip SMOTE in ONNX
            'VotingClassifier': 'VotingClassifier'
        }
        
        new_steps = []
        for name, transformer in pipeline.steps:
            transformer_type = type(transformer).__name__
            debug_print(f"Processing step {name} of type {transformer_type}")
            if transformer_type in supported_transforms:
                if supported_transforms[transformer_type] is not None:
                    new_steps.append((name, transformer))
            else:
                new_steps.append((name, transformer))
        
        if not new_steps:
            return pipeline.steps[-1][1]  # Return just the final estimator
            
        return Pipeline(new_steps)

def get_model_type(model):
    """Determine the type of the model."""
    model_type = type(model).__name__.lower()
    if hasattr(model, 'steps'):  # Handle Pipeline objects
        model_type = type(model.steps[-1][1]).__name__.lower()
    return model_type

def create_initial_type(has_pipeline=False):
    """Create initial type definition for ONNX conversion."""
    if has_pipeline:
        # For pipelines, use float as features will be scaled
        return [('float_input', FloatTensorType([None, 5]))]
    else:
        # For direct models, maintain original types
        # N, K are integers; P, EC, Fe are floats
        return [('input', FloatTensorType([None, 5]))]

def convert_model_to_onnx(model_path, onnx_path, model_name):
    """
    Convert a scikit-learn model to ONNX format.
    
    Args:
        model_path: Path to the .joblib model file
        onnx_path: Path where to save the ONNX model
        model_name: Name of the model (used for ONNX metadata)
    """
    debug_print(f"\nConverting {model_name} to ONNX format...")
    debug_print(f"Model path: {model_path}")
    debug_print(f"ONNX path: {onnx_path}")
    
    # Load the model
    try:
        model = joblib.load(model_path)
        debug_print("Model loaded successfully")
        debug_print(f"Model type: {type(model).__name__}")
    except Exception as e:
        debug_print(f"Error loading model: {str(e)}")
        return False
    
    # Extract supported parts from pipeline if needed
    if hasattr(model, 'steps'):
        debug_print("Model is a pipeline, extracting supported steps...")
        model = SklearnPipelineExtractor.extract_supported_steps(model)
        debug_print(f"Pipeline processed, final model type: {type(model).__name__}")
    
    # Determine if result is still a pipeline
    has_pipeline = hasattr(model, 'steps')
    
    # Create initial types for ONNX conversion
    initial_types = create_initial_type(has_pipeline)
    
    # Convert to ONNX
    try:
        debug_print("Starting ONNX conversion...")
        onx = convert_sklearn(
            model, 
            initial_types=initial_types,
            target_opset=12,  # Compatible opset version
            options={id(model): {'zipmap': False}},  # Disable zipmap to get raw probabilities
            name=model_name
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        # Save the model
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        
        debug_print(f"Model converted and saved to {onnx_path}")
        
        # Verify the model
        verify_onnx_model(onnx_path)
        
        return True
    
    except Exception as e:
        debug_print(f"Error during conversion: {str(e)}")
        return False

def verify_onnx_model(model_path):
    """
    Verify that the converted ONNX model can make predictions.
    """
    try:
        debug_print("\nVerifying ONNX model...")
        # Create an inference session
        session = InferenceSession(model_path)
        
        # Create a sample input
        sample_input = np.array([[200, 8.0, 500, 0.5, 0.6]], dtype=np.float32)
        input_name = session.get_inputs()[0].name
        
        # Run inference
        prediction = session.run(None, {input_name: sample_input})
        
        debug_print("Model verification successful!")
        debug_print(f"Sample prediction shape: {prediction[0].shape}")
        debug_print(f"Sample prediction: {prediction[0]}")
        debug_print(f"Model input name: {input_name}")
        
    except Exception as e:
        debug_print(f"Error during verification: {str(e)}")

def main():
    debug_print("Starting ONNX conversion process...")
    
    # First try to convert the RF model from rf_classifier_analysis.ipynb
    debug_print("\nAttempting to convert Random Forest model from analysis notebook...")
    rf_model_path = os.path.join(os.path.dirname(__file__), 'trained_models/rf_classifier.pkl')
    if os.path.exists(rf_model_path):
        rf_onnx_path = os.path.join(os.path.dirname(__file__), 'trained_models/soil_fertility_rf_classifier.onnx')
        convert_model_to_onnx(rf_model_path, rf_onnx_path, 'random_forest_classifier')
    else:
        debug_print(f"RF model not found at {rf_model_path}")

    # Try to convert the KNN model
    debug_print("\nAttempting to convert KNN model...")
    knn_model_path = os.path.join(os.path.dirname(__file__), 'enhanced_soil_fertility_knn.joblib')
    if os.path.exists(knn_model_path):
        knn_onnx_path = os.path.join(os.path.dirname(__file__), 'trained_models/soil_fertility_knn_classifier.onnx')
        convert_model_to_onnx(knn_model_path, knn_onnx_path, 'knn_classifier')
    else:
        debug_print(f"KNN model not found at {knn_model_path}")

    # Try to convert the logistic regression model
    debug_print("\nAttempting to convert Logistic Regression model...")
    lr_model_path = os.path.join(os.path.dirname(__file__), 'soil_fertility_lr_classifier.joblib')
    if os.path.exists(lr_model_path):
        lr_onnx_path = os.path.join(os.path.dirname(__file__), 'trained_models/soil_fertility_lr_classifier.onnx')
        convert_model_to_onnx(lr_model_path, lr_onnx_path, 'logistic_regression')
    else:
        debug_print(f"LR model not found at {lr_model_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        debug_print(f"Unhandled error in main: {str(e)}")
        import traceback
        debug_print(traceback.format_exc())
