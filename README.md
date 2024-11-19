# Rossmann Store Sales Prediction
This repository provides a comprehensive solution to analyze and predict sales for Rossmann stores, divided into three main tasks:

# Task 1: Customer Purchasing Behavior Analysis
* Goals: Understand customer behavior, impact of promotions, holidays, and competitor activities.
* Key Steps:
  * Data cleaning: Handle missing values and outliers.
  * Exploratory analysis: Visualize trends, correlations, and seasonal patterns.
  * Logging: Track steps using the logging library.
 
# Task 2: Sales Prediction
* Goals: Predict daily sales for up to 6 weeks.
* Key Steps:
  * Preprocessing: Feature engineering and scaling.
  * Modeling: Build sklearn pipelines with tree-based algorithms and LSTMs.
  * Post-analysis: Feature importance, confidence intervals, and model serialization.
  * Deployment: Use MLFlow for serving predictions.
 
# Task 3: Web Interface for Predictions
 * Frontend: Collect inputs like Store_id, Date, and other features.
 * Backend: Predict sales and customer numbers.
 * Visualization: Display predictions as plots and provide downloadable results.

# Technologies Used
 * Python (Pandas, Scikit-learn, TensorFlow, Streamlit, MLFlow)
 * Visualization (Matplotlib, Seaborn)
 * Deployment (Streamlit/Flask)

## Note
Due to GitHub's file size limits, some large files are not included in this repository.
To run the code, ensure you have the required datasets and models as described in the code comments.
