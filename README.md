# Telecom Customer Churn Prediction

## Objective
This project predicts which customers are likely to cancel their subscription with a telecommunications provider. The goal is to identify at-risk customers so the business can intervene with targeted retention offers.

## Data Preparation
The project uses the Telco Customer Churn dataset. The initial data pipeline handles missing values and formats the information for our ML models.
* Converted text-based billing information into numerical formats.
* Handled missing data points by replacing blank entries with zeros.
* Mapped the binary target variable (Churn) to 1s and 0s.
* Applied one-hot encoding to all remaining categorical features to prevent multicollinearity.

## Methodology
The pipeline trains and evaluates two different classification algorithms to find the most effective predictive model.
1. **Logistic Regression:** A linear baseline model to establish initial performance metrics.
2. **Random Forest Classifier:** A complex ensemble method to test for non-linear patterns in the data.

## Results and The Overfitting Problem
The results demonstrate a clear case of algorithmic overfitting.

The Logistic Regression baseline outperformed the complex Random Forest model on the unseen test data across all key business metrics.

**Predicting Churn (Class 1) Performance:**
* **Logistic Regression:** Recall: 0.60 | Precision: 0.69 | Accuracy: 0.82
* **Random Forest:** Recall: 0.44 | Precision: 0.64 | Accuracy: 0.79

The Random Forest algorithm memorised the training data too closely, so it struggled to generalise when presented with new customer records. It missed more than half of the departing customers, while the Logistic Regression model adapted much better to unseen data.

## Business Recommendation
The company should deploy the Logistic Regression model. 

This model successfully identifies 60% of departing customers who can be targeted with retention discounts. The model also maintains a precision score of 0.69, which ensures the company rarely wastes money offering discounts to customers who plan to stay anyway. 

## How to Run the Project
Ensure you have Python installed on your system.

1. Install the required libraries:
   `pip install -r requirements.txt`
2. Run the data preparation script to clean the raw data:
   `python src/data_preparation.py`
3. Train the models and view the comparative evaluation reports:
   `python src/train_model.py`