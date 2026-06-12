import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Load Data
# -----------------------------

df = pd.read_csv(
    "data/processed/featured_data.csv"
)

# -----------------------------
# Features & Target
# -----------------------------

X = df.drop(
    [
        "Placement",
        "Technical_Score",
        "Readiness_Score",
        "Profile_Strength"
    ],
    axis=1
)

y = df["Placement"]






# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Random Forest
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)




feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importances = feature_importances.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importances)

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

print("\nAccuracy")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred
    )
)
