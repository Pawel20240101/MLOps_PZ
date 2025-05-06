import pandas as pd
import joblib
import optuna
import plotly.graph_objects as go
import mlflow
from mlflow.client import MlflowClient
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import nannyml as nml
from config import PROCESSED_DATA_DIR, MODELS_DIR, categorical, target, MODEL_NAME, PROJ_ROOT

# Upewnij się, że katalogi istnieją
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Ustawienie poprawnej ścieżki rejestru MLflow
mlflow_tracking_dir = PROJ_ROOT / 'mlruns'
os.makedirs(mlflow_tracking_dir, exist_ok=True)
mlflow.set_tracking_uri(mlflow_tracking_dir.as_uri())

# Ustaw nazwę eksperymentu
mlflow.set_experiment("HeartDisease_Pipeline")

# Wczytaj dane
train_df = pd.read_csv(PROCESSED_DATA_DIR / "heart_train.csv")
y = train_df.pop(target)
X = train_df

# Obsługa zmiennych kategorycznych
categorical_features = [col for col in categorical if col in X.columns]

# Optuna - strojenie hiperparametrów
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0),
        "loss_function": "Logloss",
        "verbose": False
    }
    model = CatBoostClassifier(**params)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, cat_features=categorical_features)
    return log_loss(y_val, model.predict_proba(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # zwiększona liczba iteracji
best_params = study.best_params
best_params.update({"loss_function": "Logloss", "verbose": False, "eval_metric": "F1"})

# Cross-validation
cv_data = Pool(X, y, cat_features=categorical_features)
cv_result = cv(params=best_params, pool=cv_data, fold_count=5, shuffle=True, partition_random_seed=42, plot=False)
cv_result.to_csv(MODELS_DIR / "cv_results.csv", index=False)

# Wykres
fig = go.Figure()
fig.add_trace(go.Scatter(y=cv_result["test-Logloss-mean"], mode="lines", name="Logloss"))
if "test-F1-mean" in cv_result.columns:
    fig.add_trace(go.Scatter(y=cv_result["test-F1-mean"], mode="lines", name="F1"))
fig.update_layout(title="Wyniki Cross-Validation", xaxis_title="Iteracje", yaxis_title="Wartość")
fig.write_image(str(MODELS_DIR / "cv_plot.png"))

# MLflow
with mlflow.start_run() as run:
    model = CatBoostClassifier(**best_params)
    model.fit(X, y, cat_features=categorical_features)

    # Zapisz model lokalnie
    model.save_model(MODELS_DIR / "heart_model.cbm")
    joblib.dump(best_params, MODELS_DIR / "heart_model_params.pkl")

    preds = model.predict(X)
    proba_preds = model.predict_proba(X)
    f1 = f1_score(y, preds)
    logloss = log_loss(y, proba_preds)

    # Logowanie
    mlflow.log_params(best_params)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("log_loss", logloss)

    mlflow.log_artifact(str(MODELS_DIR / "cv_results.csv"))
    mlflow.log_artifact(str(MODELS_DIR / "cv_plot.png"))
    mlflow.log_artifact(str(MODELS_DIR / "heart_model.cbm"))
    mlflow.log_artifact(str(MODELS_DIR / "heart_model_params.pkl"))

    mlflow.catboost.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

    client = MlflowClient()
    latest_versions = client.get_latest_versions(MODEL_NAME)
    if latest_versions:
        latest_version = latest_versions[0].version
        client.set_registered_model_alias(MODEL_NAME, "challenger", latest_version)

    # NannyML
    X_monitor = X.copy()
    X_monitor["prediction"] = preds
    X_monitor["predicted_probability"] = [p[1] for p in proba_preds]
    X_monitor[target] = y

    udc = nml.UnivariateDriftCalculator(column_names=X.columns, chunk_size=100)
    udc.fit(X_monitor.drop(columns=["prediction", "predicted_probability"]))

    estimator = nml.CBPE(y_true=target, y_pred="prediction", y_pred_proba="predicted_probability",
                         problem_type="classification_binary", metrics=["roc_auc"], chunk_size=100)
    estimator.fit(X_monitor)

    store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
    store.store(udc, filename="udc.pkl")
    store.store(estimator, filename="estimator.pkl")

    mlflow.log_artifact(str(MODELS_DIR / "udc.pkl"))
    mlflow.log_artifact(str(MODELS_DIR / "estimator.pkl"))

print("Trening zakończony pomyślnie!")