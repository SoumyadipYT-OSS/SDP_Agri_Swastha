# Synthetic Data Generation for Soil Fertility Analysis

This document explains the process of generating high-quality synthetic data for the SDP_Agri_Swastha project's soil fertility analysis models.

## Overview

The synthetic data generator creates a new dataset that mimics the statistical properties of the original soil fertility dataset (`dataset_1.csv`) while producing entirely new data points. This is useful for:

1. Expanding the training dataset size
2. Testing model robustness
3. Creating diverse test scenarios
4. Protecting data privacy by not sharing the original data

## Features

The synthetic data maintains the same features as the original dataset:

- **N**: Nitrogen level (integer)
- **P**: Phosphorus level (float, 1 decimal place)
- **K**: Potassium level (integer)
- **EC**: Electrical Conductivity (float, 1 decimal place)
- **Fe**: Iron level (float, 1 decimal place)
- **Output**: Soil fertility class (0: Not Fertile, 1: Fertile, 2: Highly Fertile)

## Generation Method

The synthetic data is generated using **Gaussian Mixture Models (GMM)**, which:

1. Creates a separate GMM model for each class in the original data
2. Learns the distribution and relationships between features within each class
3. Generates new samples that maintain the same statistical properties
4. Post-processes the values to ensure they stay within realistic ranges and have appropriate formats

## Validation

The quality of synthetic data is validated by comparing:

- Feature distributions between original and synthetic data
- Feature correlations between original and synthetic data
- Class distributions to ensure proper representation
- Visual comparisons through distributions, correlations, and pairplots

## Usage

To generate synthetic data:

```bash
cd Models
python generate_synthetic_data.py
```

This will:
1. Load the original dataset from `../Datasets/dataset_1.csv`
2. Generate synthetic data
3. Save the synthetic dataset to `../Datasets/synthetic_dataset.csv`
4. Create visualization comparisons in `../Datasets/synthetic_data_visualizations/`


## Visualizations

The script generates several visualizations to compare the original and synthetic data:

1. **Feature Distributions**: Histograms showing the distribution of each feature in both datasets
2. **Correlation Heatmaps**: Comparing feature correlations in both datasets
3. **Class Distributions**: Bar charts showing the distribution of soil fertility classes
4. **Pairplots**: Visualizing relationships between pairs of features

These visualizations are saved in the `synthetic_data_visualizations` directory.



## Quality Metrics

The synthetic data quality is measured by:

- Mean and standard deviation differences for each feature
- Correlation matrix difference between original and synthetic data
- Visual inspection of distributions and relationships



## Customization

You can customize the synthetic data generation by modifying:

- `total_samples`: Total number of synthetic samples to generate
- `n_samples_per_class`: Dictionary specifying how many samples to generate for each class
- The post-processing logic to adjust how values are rounded and clipped


## References

The realistic data generator incorporates standards and patterns from:

1. FAO Soil Fertility Standards
2. USDA Soil Survey
3. European Soil Data Centre (ESDAC)
4. Global Soil Partnership