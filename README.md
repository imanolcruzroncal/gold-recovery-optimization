# Gold Recovery Process Optimization Using Machine Learning
## Overview
This project develops a machine learning pipeline to predict gold recovery rates across multiple purification stages in an industrial metallurgical process.
The objective is to model nonlinear process dynamics and ensure reliable performance evaluation using a custom weighted sMAPE metric.
## Problem Context
Industrial recovery systems involve multistage purification, process variability, and nonlinear relationships between operational parameters and recovery outputs.
Accurate prediction of recovery rates enables improved operational control, process optimization, and performance monitoring.
## Key Challenges
Missing values in target variables
Structural mismatch between training and test datasets
Time-series dependency
Feature alignment to prevent data leakage
Outlier detection in concentration levels
Custom evaluation metric (weighted sMAPE)
## Methodology
### Data validation and integrity checks
Verified official recovery formula (MAE ≈ 0)
Ensured consistency between recorded and theoretical values
### Feature alignment
Identified structural differences between train and test sets
Retained only common features to ensure deployment consistency
### Data preprocessing
Chronological sorting
Forward-fill imputation for continuous process data
Outlier removal based on concentration thresholds
### Time-based validation
80/20 chronological split
No shuffling to preserve temporal structure
### Model development
Random Forest Regressor
Linear Regression (baseline)
Cross-validation
Custom sMAPE metric implementation
## Results
Verified recovery calculation accuracy (MAE ≈ 9e-15)
No significant distribution shift detected between train and test
Final weighted sMAPE < 10%
Random Forest selected as best-performing model
These results demonstrate strong predictive reliability in nonlinear industrial recovery systems.
## Tech Stack
Python
Pandas
NumPy
Scikit-learn
Matplotlib
## How to Run
pip install -r requirements.txt
jupyter notebook
