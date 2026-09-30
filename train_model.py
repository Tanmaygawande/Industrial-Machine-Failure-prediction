"""
Industrial Machine Failure Prediction
Train/evaluate the baseline models and save the selected pipeline.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "machine_failure_data.csv"
MODEL_PATH = ROOT / "models" / "machine_failure_model.joblib"

df = pd.read_csv(DATA)
X = df.drop(columns=["failure"])
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42
        ))
    ])
}

scores = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    scores.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba)
    })

scores_df = pd.DataFrame(scores)
chosen = scores_df.sort_values(["F1", "ROC_AUC"], ascending=False).iloc[0]["Model"]
final_model = models[chosen]
final_model.fit(X_train, y_train)
joblib.dump(final_model, MODEL_PATH)

print(scores_df.to_string(index=False))
print(f"\nSelected model: {chosen}")
print(f"Saved to: {MODEL_PATH}")
