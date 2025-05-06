import mlflow
import mlflow.sklearn
from pathlib import Path
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ścieżki
data_path = Path("data/heart_2020_cleaned.csv")
models_dir = Path("models")
results_dir = Path("results")
models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("HeartDisease_Pipeline")

# 1. Wczytanie danych
df = pd.read_csv(data_path)
with mlflow.start_run(run_name="Eksploracja_danych"):
    mlflow.log_artifact(str(data_path))

# 2. Przygotowanie danych
with mlflow.start_run(run_name="Przygotowanie_danych", nested=True):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"].apply(lambda x: 1 if x == "Yes" else 0)

    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("encoding", "OneHotEncoder")

    # Zapisz dane tymczasowe
    with open(results_dir / "X_train.pkl", "wb") as f: pickle.dump(X_train, f)
    with open(results_dir / "X_test.pkl", "wb") as f: pickle.dump(X_test, f)
    with open(results_dir / "y_train.pkl", "wb") as f: pickle.dump(y_train, f)
    with open(results_dir / "y_test.pkl", "wb") as f: pickle.dump(y_test, f)

    mlflow.log_artifact(str(results_dir / "X_train.pkl"))
    mlflow.log_artifact(str(results_dir / "X_test.pkl"))
    mlflow.log_artifact(str(results_dir / "y_train.pkl"))
    mlflow.log_artifact(str(results_dir / "y_test.pkl"))

# 3. Trening modelu
with mlflow.start_run(run_name="Trening_LR", nested=True):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("solver", model.solver)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)

    mlflow.sklearn.log_model(model, "model")

    # Zapisz lokalnie
    with open(models_dir / "logreg_best.pkl", "wb") as f:
        pickle.dump(model, f)