import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Load dataset
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

# 4. Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

# 5. Print results
print("\n📊 Model Comparison:\n")

for name, res in results.items():
    print(f"{name}:")
    print(f"  accuracy: {res['accuracy']:.3f}")
    print(f"  precision: {res['precision']:.3f}")
    print(f"  recall: {res['recall']:.3f}")
    print(f"  f1_score: {res['f1_score']:.3f}\n")

# 6. Select best model by F1-score
best_model_name = max(results, key=lambda x: results[x]["f1_score"])
best_model = results[best_model_name]["model"]

best_metrics = {
    "accuracy": results[best_model_name]["accuracy"],
    "precision": results[best_model_name]["precision"],
    "recall": results[best_model_name]["recall"],
    "f1_score": results[best_model_name]["f1_score"]
}

print(f"🏆 Best Model Selected: {best_model_name}")

# 7. Save best model & metrics
joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(best_metrics, "metrics.pkl")

print("✅ Best model & metrics saved successfully")
