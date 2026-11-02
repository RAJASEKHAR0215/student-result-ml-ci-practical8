import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

def main():
    np.random.seed(42)

    rows = 2000

    data = pd.DataFrame({
        "attendance": np.random.randint(30, 101, rows),
        "internal_marks": np.random.randint(10, 101, rows),
        "assignment_marks": np.random.randint(10, 101, rows),
        "previous_score": np.random.randint(10, 101, rows)
    })

    weighted_score = (
        0.30 * data["attendance"]
        + 0.30 * data["internal_marks"]
        + 0.20 * data["assignment_marks"]
        + 0.20 * data["previous_score"]
    )

    data["result"] = (weighted_score >= 60).astype(int)

    X = data[FEATURES]
    y = data["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, "student_result_model.pkl")

    data.to_csv("student_results.csv", index=False)

    metrics = {
        "accuracy": float(accuracy),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test))
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model training completed.")
    print("Accuracy:", accuracy)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
