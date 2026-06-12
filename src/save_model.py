import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(
    "data/processed/featured_data.csv"
)

X = df.drop("Placement", axis=1)
y = df["Placement"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("Model saved successfully")