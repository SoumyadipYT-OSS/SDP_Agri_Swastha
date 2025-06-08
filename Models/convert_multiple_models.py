import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

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

def convert_models_to_onnx():
    print("Loading datasets...")
    # Load both original and synthetic data
    df_original = pd.read_csv('../Datasets/dataset_1.csv')
    df_synthetic = pd.read_csv('../Datasets/synthetic_dataset.csv')
    
    # Combine datasets
    df_original['is_synthetic'] = 0
    df_synthetic['is_synthetic'] = 1
    df = pd.concat([df_original, df_synthetic], ignore_index=True)
    
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
    
    # Define initial shape for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, 5]))]
    
    # Convert each model
    model_files = [
        'new_knn_classifier.joblib',
        'new_svm_classifier.joblib',
        'new_rf_classifier.joblib'
    ]
    
    for model_file in model_files:
        try:
            print(f"\nConverting {model_file} to ONNX...")
            
            # Load the model
            model = joblib.load(model_file)
            
            # Convert to ONNX
            onnx_model = convert_sklearn(
                model, 
                initial_types=initial_type,
                target_opset=12,
                options={
                    'zipmap': False,  # Disable zipmap to get raw probabilities
                    'return_probabilities': True
                }
            )
            
            # Save ONNX model
            onnx_file = f'trained_models/{model_file.replace(".joblib", ".onnx")}'
            with open(onnx_file, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"Saved ONNX model to {onnx_file}")
            
            # Also copy to website models directory
            website_model_path = f'../website/models/{model_file.replace(".joblib", ".onnx")}'
            with open(website_model_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"Copied ONNX model to {website_model_path}")
            
        except Exception as e:
            print(f"Error converting {model_file}: {str(e)}")

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
    convert_models_to_onnx()
