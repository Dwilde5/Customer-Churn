import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_logistic_regression(data_path):
    # Load the cleaned dataset
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Separate features and target variable
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    data_path = "data/processed/cleaned_data.csv"
    train_logistic_regression(data_path)