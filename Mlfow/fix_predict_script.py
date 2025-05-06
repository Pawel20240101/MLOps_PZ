import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
from mlflow.client import MlflowClient
from catboost import CatBoostClassifier
import json
import os
from pathlib import Path
from config import PROCESSED_DATA_DIR, MODELS_DIR, target, categorical, MODEL_NAME
import nannyml as nml

# Upewnij się, że katalogi istnieją
os.makedirs(MODELS_DIR, exist_ok=True)

# Ustaw ścieżkę MLflow
mlflow_tracking_dir = Path.cwd().parent / 'mlruns'
mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
mlflow.set_experiment("heart_predictions")

# Wczytaj dane testowe
df_test = pd.read_csv(PROCESSED_DATA_DIR / "heart_test.csv")

# Inicjalizacja klienta MLflow
client = MlflowClient()

# Zmodyfikowana logika pobrania modelu - najpierw sprawdza czy model istnieje
try:
    # Sprawdź czy model w ogóle istnieje
    client.get_registered_model(MODEL_NAME)
    
    try:
        # Pobierz model za pomocą aliasu
        model_info = client.get_model_version_by_alias(MODEL_NAME, alias="champion")
        print(f"Znaleziono model z aliasem 'champion': {model_info.version}")
    except Exception as e:
        print(f"Nie znaleziono modelu z aliasem 'champion', pobieram najnowszy: {str(e)}")
        # Jeśli nie ma aliasu, pobierz najnowszą wersję
        latest_versions = client.get_latest_versions(MODEL_NAME)
        if not latest_versions:
            raise Exception(f"Nie znaleziono żadnego modelu {MODEL_NAME} w rejestrze")
        model_info = latest_versions[0]
        print(f"Używam najnowszej wersji modelu: {model_info.version}")
        
except Exception as e:
    print(f"Model {MODEL_NAME} nie istnieje w rejestrze. Błąd: {str(e)}")
    print("Musisz najpierw wytrenować i zarejestrować model przed użyciem tego skryptu.")
    print("Sprawdź ścieżkę MLflow i upewnij się, że model został poprawnie zarejestrowany.")
    exit(1)

# Pobierz informacje o modelu
run = client.get_run(model_info.run_id)
print(f"Run ID: {model_info.run_id}")

# Uzyskaj URI modelu
model_uri = f"models:/{MODEL_NAME}/{model_info.version}"
print(f"Ładowanie modelu z URI: {model_uri}")

# Załaduj model
model = mlflow.catboost.load_model(model_uri)

# Kategorialne cechy
categorical_features = [col for col in categorical if col in df_test.columns]

# Predykcje
X_test = df_test.drop(columns=[target], errors='ignore')
y_test = df_test[target] if target in df_test.columns else None

# Wykonaj predykcje
print("Wykonywanie predykcji...")
df_test["prediction"] = model.predict(X_test)
df_test["predicted_probability"] = [p[1] for p in model.predict_proba(X_test)]
df_test.to_csv(MODELS_DIR / "heart_predictions.csv", index=False)
print("Predykcje zapisane do pliku CSV")

# SHAP
print("Generowanie wartości SHAP...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(MODELS_DIR / "heart_shap_summary.png")
print("Zapisano wykres SHAP")

# NannyML - analiza driftu i wydajności
print("Analiza driftu danych...")
store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))

try:
    udc = store.load(filename="udc.pkl", as_type=nml.UnivariateDriftCalculator)
    estimator = store.load(filename="estimator.pkl", as_type=nml.CBPE)

    analysis_df = df_test.copy()
    
    # Szacowanie wydajności
    estimated_performance = estimator.estimate(analysis_df)
    fig1 = estimated_performance.plot()
    fig1.write_image(str(MODELS_DIR / "performance_estimation.png"))
    
    # Analiza driftu
    drift_columns = [col for col in X_test.columns if col not in ["PassengerId", "prediction", "predicted_probability"]]
    univariate_drift = udc.calculate(X_test[drift_columns])
    
    for col in drift_columns:
        try:
            fig = univariate_drift.filter(column_names=[col]).plot()
            fig.write_image(str(MODELS_DIR / f"drift_{col}.png"))
        except Exception as e:
            print(f"Nie udało się utworzyć wykresu driftu dla {col}: {e}")
    
    print("Zapisano wykresy driftu danych")
except Exception as e:
    print(f"Błąd podczas analizy driftu: {e}")

# Logowanie do MLflow
print("Logowanie wyników do MLflow...")
with mlflow.start_run() as run:
    # Loguj metryki jeśli mamy etykiety
    if y_test is not None:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        accuracy = accuracy_score(y_test, df_test["prediction"])
        f1 = f1_score(y_test, df_test["prediction"])
        precision = precision_score(y_test, df_test["prediction"])
        recall = recall_score(y_test, df_test["prediction"])
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        print(f"Metryki: Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    # Loguj artefakty
    mlflow.log_artifact(str(MODELS_DIR / "heart_predictions.csv"))
    mlflow.log_artifact(str(MODELS_DIR / "heart_shap_summary.png"))
    
    try:
        mlflow.log_artifact(str(MODELS_DIR / "performance_estimation.png"))
        print("Zalogowano artefakty do MLflow")
    except:
        print("Nie znaleziono niektórych artefaktów")
    
    # Loguj parametry
    mlflow.log_param("model_uri", model_uri)
    mlflow.log_param("model_version", model_info.version)

print("Przewidywanie zakończone pomyślnie!")
