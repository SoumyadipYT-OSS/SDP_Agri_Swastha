import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import pickle
# ONNX conversion libraries
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import warnings
warnings.filterwarnings('ignore')

class SoilFertilityRandomForestClassifier:
    """
    Random Forest Classifier for Soil Fertility Prediction
    Features: N, P, K, EC, Fe
    Target Classes: 0 (Low Fertility), 1 (Medium Fertility), 2 (High Fertility)
    Designed for integration with .NET MAUI application through ONNX
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['N', 'P', 'K', 'EC', 'Fe']
        self.target_classes = {0: 'Low Fertility', 1: 'Medium Fertility', 2: 'High Fertility'}
        self.is_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
    
    def load_data(self, filepath=None):
        """Load the synthetic dataset"""
        try:
            if filepath is None:
                # Get the absolute path to the dataset
                script_dir = os.path.dirname(os.path.abspath(__file__))
                filepath = os.path.join(script_dir, "..", "Datasets", "synthetic_dataset.csv")
                filepath = os.path.normpath(filepath)
            
            print(f"Loading dataset from: {filepath}")
            
            if not os.path.exists(filepath):
                print(f"Error: Dataset file not found at {filepath}")
                return None
                
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully!")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Display dataset information
            print(f"\nDataset Overview:")
            print(df.head())
            print(f"\nTarget Distribution:")
            print(df['Output'].value_counts().sort_index())
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\nMissing values found:")
                print(missing_values[missing_values > 0])
            else:
                print(f"\nNo missing values found.")
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        try:
            # Extract features and target
            X = df[self.feature_names].copy()
            y = df['Output'].copy()
            
            print(f"\nPreprocessing data...")
            print(f"Features shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            print(f"Feature columns: {list(X.columns)}")
            print(f"Target classes: {sorted(y.unique())}")
            
            # Check feature statistics
            print(f"\nFeature Statistics:")
            print(X.describe())
            
            return X, y
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return None, None
    
    def train_model(self, X, y, test_size=0.2, optimize_hyperparameters=True):
        """Train the Random Forest model"""
        try:
            print(f"\nSplitting data into train/test sets...")
            # Split the data with stratification to maintain class balance
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, 
                stratify=y, shuffle=True
            )
            
            print(f"Training set size: {self.X_train.shape}")
            print(f"Test set size: {self.X_test.shape}")
            print(f"Training set class distribution:")
            print(self.y_train.value_counts().sort_index())
            print(f"Test set class distribution:")
            print(self.y_test.value_counts().sort_index())
            
            # Scale the features
            print(f"\nScaling features...")
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            
            if optimize_hyperparameters:
                print(f"\nOptimizing hyperparameters with GridSearchCV...")
                # Define parameter grid
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
                
                # Create base model
                rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
                
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(
                    rf, param_grid, cv=5, scoring='accuracy', 
                    n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train_scaled, self.y_train)
                
                # Use the best model
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
                
            else:
                print(f"\nTraining with default optimized parameters...")
                # Use pre-optimized parameters for faster training
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                self.model.fit(X_train_scaled, self.y_train)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_scaled)
            self.y_pred = self.model.predict(X_test_scaled)
            
            # Calculate performance metrics
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred, average='weighted')
            recall = recall_score(self.y_test, self.y_pred, average='weighted')
            f1 = f1_score(self.y_test, self.y_pred, average='weighted')
            
            print(f"\n=== Model Performance ===")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Precision (weighted): {precision:.4f}")
            print(f"Recall (weighted): {recall:.4f}")
            print(f"F1-Score (weighted): {f1:.4f}")
            
            # Detailed classification report
            print(f"\n=== Classification Report ===")
            print(classification_report(self.y_test, self.y_pred, 
                                      target_names=[self.target_classes[i] for i in sorted(self.target_classes.keys())]))
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                print(f"\n=== Feature Importance ===")
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                print(feature_importance)
            
            self.is_trained = True
            
            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'model': self.model,
                'scaler': self.scaler
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def save_model(self, model_path=None):
        """Save the trained model and scaler"""
        try:
            if not self.is_trained:
                print("Error: Model is not trained yet")
                return False
            
            if model_path is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, "rfc_model.pkl")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'target_classes': self.target_classes,
                'random_state': self.random_state
            }
            
            joblib.dump(model_data, model_path)
            print(f"Model saved successfully to: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path=None):
        """Load a previously trained model"""
        try:
            if model_path is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, "rfc_model.pkl")
            
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return False
            
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.target_classes = model_data['target_classes']
            self.random_state = model_data.get('random_state', 42)
            self.is_trained = True
            
            print(f"Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def convert_to_onnx(self, onnx_path=None):
        """Convert the trained model to ONNX format for .NET MAUI integration"""
        try:
            if not self.is_trained:
                print("Error: Model is not trained yet")
                return False
            
            if onnx_path is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                onnx_path = os.path.join(script_dir, "rfc_model.onnx")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            
            print(f"Converting model to ONNX format...")
            
            # Define input type for ONNX (5 features: N, P, K, EC, Fe)
            initial_type = [('float_input', FloatTensorType([None, len(self.feature_names)]))]
            
            # Create a pipeline that includes scaling and classification
            from sklearn.pipeline import Pipeline
            
            pipeline = Pipeline([
                ('scaler', self.scaler),
                ('classifier', self.model)
            ])
            
            # Convert to ONNX
            onnx_model = convert_sklearn(
                pipeline,
                initial_types=initial_type,
                target_opset=11
            )
            
            # Save ONNX model
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"ONNX model saved successfully to: {onnx_path}")
            
            # Test ONNX model
            self.test_onnx_model(onnx_path)
            
            return True
            
        except Exception as e:
            print(f"Error converting to ONNX: {e}")
            return False
    
    def test_onnx_model(self, onnx_path=None):
        """Test the ONNX model to ensure it works correctly"""
        try:
            if onnx_path is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                onnx_path = os.path.join(script_dir, "rfc_model.onnx")
            
            print(f"Testing ONNX model...")
            
            # Load ONNX model
            sess = rt.InferenceSession(onnx_path)
            
            # Get input and output names
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            
            print(f"ONNX Input name: {input_name}")
            print(f"ONNX Output name: {output_name}")
            
            # Test with sample data
            if hasattr(self, 'X_test') and len(self.X_test) > 0:
                # Use first few test samples
                test_samples = self.X_test.iloc[:3].values.astype(np.float32)
                
                # Make prediction with ONNX
                onnx_pred = sess.run([output_name], {input_name: test_samples})[0]
                
                # Make prediction with scikit-learn (for comparison)
                test_samples_scaled = self.scaler.transform(test_samples)
                sklearn_pred = self.model.predict(test_samples_scaled)
                
                print(f"\nONNX Model Test Results:")
                for i, (sample, onnx_p, sklearn_p) in enumerate(zip(test_samples, onnx_pred, sklearn_pred)):
                    print(f"Sample {i+1}: {sample}")
                    print(f"  ONNX prediction: {onnx_p} ({self.target_classes[onnx_p]})")
                    print(f"  Scikit-learn prediction: {sklearn_p} ({self.target_classes[sklearn_p]})")
                    print(f"  Match: {'✓' if onnx_p == sklearn_p else '✗'}")
                
                print("ONNX model test completed successfully!")
                return True
            else:
                print("No test data available for ONNX testing")
                return False
                
        except Exception as e:
            print(f"Error testing ONNX model: {e}")
            return False
    
    def predict(self, features):
        """Make predictions on new data"""
        try:
            if not self.is_trained:
                print("Error: Model is not trained yet")
                return None
            
            # Convert to numpy array if needed
            if isinstance(features, list):
                features = np.array(features)
            elif isinstance(features, dict):
                # If features provided as dictionary with feature names
                features = np.array([features[name] for name in self.feature_names])
            
            # Ensure features is a 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Validate feature count
            if features.shape[1] != len(self.feature_names):
                print(f"Error: Expected {len(self.feature_names)} features, got {features.shape[1]}")
                return None
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            prediction_proba = self.model.predict_proba(features_scaled)
            
            result = []
            for i in range(len(prediction)):
                result.append({
                    'prediction': int(prediction[i]),
                    'prediction_class': self.target_classes[prediction[i]],
                    'probabilities': {
                        self.target_classes[j]: float(prediction_proba[i][j]) 
                        for j in range(len(self.target_classes))
                    },
                    'confidence': float(np.max(prediction_proba[i]))
                })
            
            return result if len(result) > 1 else result[0]
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def visualize_performance(self, save_path=None):
        """Create comprehensive performance visualizations"""
        try:
            if not self.is_trained or self.y_test is None or self.y_pred is None:
                print("Error: Model not trained or test data not available")
                return False
            
            if save_path is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(script_dir, "rfc_performance_analysis.png")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Random Forest Classifier - Soil Fertility Prediction Performance', fontsize=16)
            
            # 1. Confusion Matrix
            cm = confusion_matrix(self.y_test, self.y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                       xticklabels=[self.target_classes[i] for i in sorted(self.target_classes.keys())],
                       yticklabels=[self.target_classes[i] for i in sorted(self.target_classes.keys())])
            axes[0,0].set_title('Confusion Matrix')
            axes[0,0].set_xlabel('Predicted')
            axes[0,0].set_ylabel('Actual')
            
            # 2. Feature Importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                bars = axes[0,1].barh(feature_importance['feature'], feature_importance['importance'])
                axes[0,1].set_title('Feature Importance')
                axes[0,1].set_xlabel('Importance')
                
                # Add value labels on bars
                for bar, value in zip(bars, feature_importance['importance']):
                    axes[0,1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                                  f'{value:.3f}', ha='left', va='center')
            
            # 3. Class Distribution Comparison
            test_counts = pd.Series(self.y_test).value_counts().sort_index()
            pred_counts = pd.Series(self.y_pred).value_counts().sort_index()
            
            x = np.arange(len(self.target_classes))
            width = 0.35
            
            axes[1,0].bar(x - width/2, test_counts.values, width, label='Actual', alpha=0.8)
            axes[1,0].bar(x + width/2, pred_counts.values, width, label='Predicted', alpha=0.8)
            axes[1,0].set_title('Class Distribution: Actual vs Predicted')
            axes[1,0].set_xlabel('Fertility Class')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels([self.target_classes[i] for i in sorted(self.target_classes.keys())], rotation=45)
            axes[1,0].legend()
            
            # 4. Performance Metrics
            accuracy = accuracy_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred, average='weighted')
            recall = recall_score(self.y_test, self.y_pred, average='weighted')
            f1 = f1_score(self.y_test, self.y_pred, average='weighted')
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [accuracy, precision, recall, f1]
            colors = ['skyblue', 'lightgreen', 'orange', 'pink']
            
            bars = axes[1,1].bar(metrics, values, color=colors)
            axes[1,1].set_title('Performance Metrics')
            axes[1,1].set_ylabel('Score')
            axes[1,1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance visualization saved to: {save_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return False
    
    def get_model_info(self):
        """Get comprehensive information about the trained model"""
        if not self.is_trained:
            return {"status": "Model is not trained yet"}
        
        info = {
            "model_type": "Random Forest Classifier",
            "purpose": "Soil Fertility Prediction",
            "features": {
                "names": self.feature_names,
                "count": len(self.feature_names),
                "description": {
                    "N": "Nitrogen content",
                    "P": "Phosphorus content", 
                    "K": "Potassium content",
                    "EC": "Electrical Conductivity",
                    "Fe": "Iron content"
                }
            },
            "target_classes": self.target_classes,
            "model_parameters": {
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "min_samples_split": self.model.min_samples_split,
                "min_samples_leaf": self.model.min_samples_leaf,
                "max_features": self.model.max_features,
                "random_state": self.random_state
            },
            "performance": {
                "test_accuracy": accuracy_score(self.y_test, self.y_pred) if hasattr(self, 'y_test') and hasattr(self, 'y_pred') else None,
                "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_)) if hasattr(self.model, 'feature_importances_') else None
            }
        }
        
        return info

# Main execution function
def main():
    """Main function to train and save the model"""
    print("=== Soil Fertility Random Forest Classifier ===")
    print("Initializing classifier...")
    
    # Create classifier instance
    classifier = SoilFertilityRandomForestClassifier(random_state=42)
    
    # Load data
    print("\n1. Loading dataset...")
    df = classifier.load_data()
    
    if df is not None:
        # Preprocess data
        print("\n2. Preprocessing data...")
        X, y = classifier.preprocess_data(df)
        
        if X is not None and y is not None:
            # Train model
            print("\n3. Training model...")
            results = classifier.train_model(X, y, optimize_hyperparameters=False)
            
            if results is not None:
                # Save model
                print("\n4. Saving model...")
                classifier.save_model()
                
                # Convert to ONNX
                print("\n5. Converting to ONNX...")
                classifier.convert_to_onnx()
                
                # Create visualizations
                print("\n6. Creating performance visualizations...")
                classifier.visualize_performance()
                
                # Display model info
                print("\n7. Model Information:")
                model_info = classifier.get_model_info()
                print(f"Model Type: {model_info['model_type']}")
                print(f"Features: {model_info['features']['names']}")
                print(f"Target Classes: {model_info['target_classes']}")
                print(f"Test Accuracy: {model_info['performance']['test_accuracy']:.4f}")
                
                # Test prediction with sample data
                print("\n8. Testing prediction with sample data...")
                sample_data = {
                    'N': 250.0,
                    'P': 12.0,
                    'K': 450.0,
                    'EC': 0.5,
                    'Fe': 3.5
                }
                prediction = classifier.predict(sample_data)
                if prediction:
                    print(f"Sample input: {sample_data}")
                    print(f"Prediction: {prediction['prediction_class']} (Class {prediction['prediction']})")
                    print(f"Confidence: {prediction['confidence']:.4f}")
                    print(f"Probabilities: {prediction['probabilities']}")
                
                print("\n=== Model Training Complete ===")
                print("Files generated:")
                print("- rfc_model.pkl (Scikit-learn model)")
                print("- rfc_model.onnx (ONNX model for .NET MAUI)")
                print("- rfc_performance_analysis.png (Performance visualizations)")
                
            else:
                print("Failed to train model")
        else:
            print("Failed to preprocess data")
    else:
        print("Failed to load dataset")

if __name__ == "__main__":
    main()