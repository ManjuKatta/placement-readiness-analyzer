import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# Load engineered dataset
# -----------------------------------

df = pd.read_csv(
    "data/processed/featured_data.csv"
)

print("Dataset Shape:")
print(df.shape)

# -----------------------------------
# Features and Target
# -----------------------------------

X = df.drop("Placement", axis=1)

y = df["Placement"]

print("\nFeature Columns:")
print(X.columns.tolist())

# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Size:", X_train.shape)
print("Testing Size:", X_test.shape)

# -----------------------------------
# Logistic Regression
# -----------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

# -----------------------------------
# Predictions
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# Evaluation
# -----------------------------------

print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)