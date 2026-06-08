import pandas as pd
import numpy as np

def load_and_clean_data (input_path):
    # Load the dataset
    df = pd.read_csv(input_path)
    # Drop customerID column
    df = df.drop(columns=['customerID'])

    # Convert TotalCharges to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    df['churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded

if __name__ == "__main__":
    input_path  = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path = "data/processed/cleaned_data.csv"

    clean_data = load_and_clean_data(input_path)
    clean_data.to_csv(output_path, index=False)
    print("Data cleaned")