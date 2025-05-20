#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Synthetic Data Generator for Soil Fertility Dataset

This script generates high-quality synthetic soil fertility data based on the statistical 
properties of the original dataset (dataset_1.csv). The synthetic data maintains the 
same relationships between features and target variables while providing new samples
for model testing and validation.

Features:
- N: Nitrogen level
- P: Phosphorus level
- K: Potassium level
- EC: Electrical Conductivity
- Fe: Iron level
- Output: Soil fertility class (0: Not Fertile, 1: Fertile, 2: Highly Fertile)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed for reproducibility
np.random.seed(42)

# Set display options for better output readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_data(data_path):
    """
    Load the original dataset and display basic information.
    
    Args:
        data_path (str): Path to the source CSV file
        
    Returns:
        pd.DataFrame: The loaded dataset
    """
    print(f"Loading original dataset from {data_path}...")
    data = pd.read_csv(data_path)
    
    print(f"Dataset shape: {data.shape}")
    print("\nSample of the original data:")
    print(data.head())
    
    print("\nBasic dataset information:")
    print(f"Features: {', '.join(data.columns[:-1])}")
    print(f"Target variable: {data.columns[-1]}")
    
    # Display class distribution
    class_counts = data['Output'].value_counts().sort_index()
    print("\nClass distribution in original dataset:")
    for cls, count in class_counts.items():
        percentage = count / len(data) * 100
        print(f"Class {cls}: {count} samples ({percentage:.2f}%)")
    
    return data

def analyze_feature_distributions(data):
    """
    Analyze the statistical properties and distributions of features.
    
    Args:
        data (pd.DataFrame): The dataset
        
    Returns:
        dict: Feature statistics by class
    """
    print("\n" + "="*50)
    print("Analyzing feature distributions by class...")
    
    class_labels = sorted(data['Output'].unique())
    feature_names = data.columns[:-1]
    
    # Dictionary to store class-specific statistics
    class_stats = {}
    
    # Get statistics for each feature by class
    for cls in class_labels:
        # Filter data for this class
        class_data = data[data['Output'] == cls]
        print(f"\nClass {cls} Statistics (n={len(class_data)}):")
        
        # Calculate and display statistics
        stats_df = class_data[feature_names].describe()
        print(stats_df)
        
        # Calculate correlation for this class
        corr = class_data[feature_names].corr()
        
        # Store statistics for this class
        class_stats[cls] = {
            'mean': class_data[feature_names].mean().values,
            'std': class_data[feature_names].std().values,
            'median': class_data[feature_names].median().values,
            'min': class_data[feature_names].min().values,
            'max': class_data[feature_names].max().values,
            'corr': corr,
            'count': len(class_data),
            'data': class_data[feature_names].values
        }
    
    # Calculate overall feature correlations
    print("\nOverall feature correlation matrix:")
    overall_corr = data[feature_names].corr()
    print(overall_corr)
    
    return class_stats, feature_names, class_labels

def generate_synthetic_data_gmm(data, class_stats, feature_names, class_labels, n_samples_per_class):
    """
    Generate synthetic data using Gaussian Mixture Models.
    
    This method creates a separate GMM for each class and generates synthetic 
    samples that preserve the feature distributions and relationships.
    
    Args:
        data (pd.DataFrame): Original dataset
        class_stats (dict): Statistics for each class
        feature_names (list): List of feature names
        class_labels (list): List of class labels
        n_samples_per_class (dict): Number of samples to generate for each class
        
    Returns:
        pd.DataFrame: Synthetic dataset
    """
    print("\n" + "="*50)
    print("Generating synthetic data using Gaussian Mixture Models...")
    
    # Generate synthetic samples for each class
    synthetic_data_list = []
    
    for cls in class_labels:
        print(f"Generating {n_samples_per_class[cls]} samples for Class {cls}...")
        
        # Get training data for this class
        X_cls = class_stats[cls]['data']
        n_features = X_cls.shape[1]
        
        # Determine number of components based on class size
        n_components = min(max(1, len(X_cls) // 10), 10)
        
        # Create and train GMM model
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42
        )
        gmm.fit(X_cls)
        
        # Generate synthetic samples
        X_synthetic, _ = gmm.sample(n_samples_per_class[cls])
        
        # Create a DataFrame with the synthetic samples
        df_synthetic = pd.DataFrame(X_synthetic, columns=feature_names)
        
        # Add class label
        df_synthetic['Output'] = cls
        
        # Apply post-processing to keep values in valid range
        for i, feature in enumerate(feature_names):
            min_val = max(0, class_stats[cls]['min'][i])
            max_val = class_stats[cls]['max'][i]
            
            # Clip values to valid range
            df_synthetic[feature] = df_synthetic[feature].clip(min_val, max_val)
            
            # Round values appropriately
            if feature in ['N', 'K']:
                df_synthetic[feature] = df_synthetic[feature].round().astype(int)
            else:
                df_synthetic[feature] = df_synthetic[feature].round(1)
        
        synthetic_data_list.append(df_synthetic)
    
    # Combine data from all classes
    synthetic_data = pd.concat(synthetic_data_list, ignore_index=True)
    
    # Shuffle the data
    synthetic_data = synthetic_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nSynthetic dataset shape: {synthetic_data.shape}")
    print("Synthetic class distribution:")
    syn_class_counts = synthetic_data['Output'].value_counts().sort_index()
    for cls, count in syn_class_counts.items():
        percentage = count / len(synthetic_data) * 100
        print(f"Class {cls}: {count} samples ({percentage:.2f}%)")
    
    return synthetic_data

def validate_synthetic_data(original_data, synthetic_data, feature_names, class_labels):
    """
    Validate the statistical properties of the synthetic data against the original data.
    
    Args:
        original_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        feature_names (list): List of feature names
        class_labels (list): List of class labels
    """
    print("\n" + "="*50)
    print("Validating synthetic data quality...")
    
    # Compare feature distributions
    for feature in feature_names:
        # Calculate statistics
        orig_mean = original_data[feature].mean()
        orig_std = original_data[feature].std()
        syn_mean = synthetic_data[feature].mean()
        syn_std = synthetic_data[feature].std()
        
        mean_diff_pct = abs(orig_mean - syn_mean) / orig_mean * 100 if orig_mean != 0 else 0
        std_diff_pct = abs(orig_std - syn_std) / orig_std * 100 if orig_std != 0 else 0
        
        print(f"\n{feature}:")
        print(f"  Original mean: {orig_mean:.2f}, std: {orig_std:.2f}")
        print(f"  Synthetic mean: {syn_mean:.2f}, std: {syn_std:.2f}")
        print(f"  Mean diff: {mean_diff_pct:.2f}%, Std diff: {std_diff_pct:.2f}%")
    
    # Compare correlation matrices
    orig_corr = original_data[feature_names].corr()
    syn_corr = synthetic_data[feature_names].corr()
    
    print("\nCorrelation matrix difference:")
    corr_diff = abs(orig_corr - syn_corr)
    print(corr_diff)
    
    print(f"\nAverage correlation difference: {np.mean(corr_diff.values):.4f}")

def generate_comparison_visualizations(original_data, synthetic_data, feature_names, output_dir):
    """
    Generate visualizations comparing original and synthetic data.
    
    Args:
        original_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        feature_names (list): List of feature names
        output_dir (str): Directory to save visualizations
    """
    print("\n" + "="*50)
    print("Generating comparison visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # 1. Feature Distributions
    for feature in feature_names:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original data
        sns.histplot(data=original_data, x=feature, hue='Output', kde=True, ax=axes[0])
        axes[0].set_title(f'Original Data: {feature} Distribution', fontsize=14)
        
        # Synthetic data
        sns.histplot(data=synthetic_data, x=feature, hue='Output', kde=True, ax=axes[1])
        axes[1].set_title(f'Synthetic Data: {feature} Distribution', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature}_distribution_comparison.png", dpi=300)
        plt.close()
    
    # 2. Correlation Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Original data correlation
    orig_corr = original_data[feature_names].corr()
    sns.heatmap(orig_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title('Original Data Correlation', fontsize=14)
    
    # Synthetic data correlation
    syn_corr = synthetic_data[feature_names].corr()
    sns.heatmap(syn_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title('Synthetic Data Correlation', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_comparison.png", dpi=300)
    plt.close()
    
    # 3. Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original data
    orig_class_counts = original_data['Output'].value_counts(normalize=True).sort_index() * 100
    sns.barplot(x=orig_class_counts.index, y=orig_class_counts.values, ax=axes[0])
    axes[0].set_title('Original Data Class Distribution', fontsize=14)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Percentage')
    
    # Add percentage labels
    for i, v in enumerate(orig_class_counts.values):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Synthetic data
    syn_class_counts = synthetic_data['Output'].value_counts(normalize=True).sort_index() * 100
    sns.barplot(x=syn_class_counts.index, y=syn_class_counts.values, ax=axes[1])
    axes[1].set_title('Synthetic Data Class Distribution', fontsize=14)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Percentage')
    
    # Add percentage labels
    for i, v in enumerate(syn_class_counts.values):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_distribution_comparison.png", dpi=300)
    plt.close()
    
    # 4. Pair plots (sample of features to avoid excessive plotting)
    sample_features = feature_names[:3]  # Take first 3 features
    
    # Original data
    plt.figure(figsize=(10, 8))
    orig_pair = sns.pairplot(original_data, vars=sample_features, hue='Output', height=2)
    plt.suptitle('Original Data Pairplot', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/original_pairplot.png", dpi=300)
    plt.close()
    
    # Synthetic data
    plt.figure(figsize=(10, 8))
    syn_pair = sns.pairplot(synthetic_data, vars=sample_features, hue='Output', height=2)
    plt.suptitle('Synthetic Data Pairplot', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/synthetic_pairplot.png", dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function to execute the synthetic data generation process."""
    # Paths
    input_data_path = "../Datasets/dataset_1.csv"
    output_data_path = "../Datasets/synthetic_dataset.csv"
    vis_output_dir = "../Datasets/synthetic_data_visualizations"
    
    # Load and analyze original data
    original_data = load_data(input_data_path)
    class_stats, feature_names, class_labels = analyze_feature_distributions(original_data)
    
    # Determine number of samples to generate for each class
    # Generate more samples than the original dataset
    total_samples = 1000
    class_distribution = {cls: count for cls, count in original_data['Output'].value_counts().items()}
    total_original = sum(class_distribution.values())
    
    n_samples_per_class = {}
    for cls, count in class_distribution.items():
        # Maintain similar class proportions but ensure minimum samples for small classes
        proportion = count / total_original
        n_samples_per_class[cls] = max(50, int(total_samples * proportion))
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data_gmm(
        original_data, 
        class_stats, 
        feature_names, 
        class_labels, 
        n_samples_per_class
    )
    
    # Validate synthetic data quality
    validate_synthetic_data(original_data, synthetic_data, feature_names, class_labels)
    
    # Generate comparison visualizations
    generate_comparison_visualizations(original_data, synthetic_data, feature_names, vis_output_dir)
    
    # Save synthetic data
    synthetic_data.to_csv(output_data_path, index=False)
    print(f"\nSynthetic data saved to {output_data_path}")

if __name__ == "__main__":
    main()
