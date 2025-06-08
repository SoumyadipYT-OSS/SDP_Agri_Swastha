import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def test_optimal_fertility():
    """Test model predictions with optimal fertility values"""
    # Load models
    models = {
        'KNN': joblib.load('new_knn_classifier.joblib'),
        'SVM': joblib.load('new_svm_classifier.joblib'),
        'RF': joblib.load('new_rf_classifier.joblib')
    }
    
    # Test cases with optimal ranges
    test_cases = [
        {
            'name': 'Optimal High Fertility',
            'values': {
                'N': 365,  # Optimal range: 350-380
                'P': 12,   # Optimal range: 11-13
                'K': 825,  # Optimal range: 800-850
                'EC': 0.9, # Optimal range: 0.85-0.95
                'Fe': 9    # Optimal range: 8-10
            }
        },
        {
            'name': 'Medium Fertility',
            'values': {
                'N': 290,
                'P': 8.5,
                'K': 500,
                'EC': 0.6,
                'Fe': 4.5
            }
        },
        {
            'name': 'Low Fertility',
            'values': {
                'N': 150,
                'P': 5.0,
                'K': 300,
                'EC': 0.3,
                'Fe': 2.0
            }
        }
    ]
    
    print("Testing models with different fertility levels:\n")
    
    # Test each case
    for case in test_cases:
        print(f"\n{case['name']}:")
        print("Input values:", case['values'])
        
        # Convert input to array
        X = np.array([[
            case['values']['N'],
            case['values']['P'],
            case['values']['K'],
            case['values']['EC'],
            case['values']['Fe']
        ]])
        
        # Get predictions from each model
        for model_name, model in models.items():
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            
            print(f"\n{model_name} Model:")
            print(f"Predicted class: {y_pred[0]}")
            print("Class probabilities:")
            for i, prob in enumerate(y_proba[0]):
                print(f"Class {i}: {prob:.4f}")

if __name__ == "__main__":
    test_optimal_fertility()