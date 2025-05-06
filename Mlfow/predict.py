import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
from mlflow.client import MlflowClient
from catboost import CatBoostClassifier
import json
import os
from config import PROCESSED_DATA_DIR, MODELS_DIR, target, categorical, MODEL_NAME
import nannyml as nml

# Wczytaj dane testowe
df_test = pd.read_csv(PROCESSED_DATA_DIR / "heart_test.csv")

# Pobierz model
client = MlflowClient()
model_info = client.get_model_version_by_alias(MODEL_NAME, alias="champion")
if model_info is None:
    model_info = client.get_latest_versions(MODEL_NAME)[0]

run = client.get_run(model_info.run_id)
log_model_meta = json.loads(run.data.tags['mlflow.log-model.history'])
feature_columns = [inp["name"] for inp in json.loads(log_model_meta[0]['signature']['inputs'])]

model_uri = f"runs:/{model_info.run_id}/{model_info.source.split('/')[-1]}"
model = mlflow.catboost.load_model(model_uri)

# Predykcje
df_test["prediction"] = model.predict(df_test[feature_columns])
df_test["predicted_probability"] = [p[1] for p in model.predict_proba(df_test[feature_columns])]
df_test.to_csv(MODELS_DIR / "heart_predictions.csv", index=False)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_test[feature_columns])
shap.summary_plot(shap_values, df_test[feature_columns], show=False)
plt.savefig(MODELS_DIR / "heart_shap_summary.png")

# NannyML - analiza driftu i wydajności
store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
udc = store.load(filename="udc.pkl", as_type=nml.UnivariateDriftCalculator)
estimator = store.load(filename="estimator.pkl", as_type=nml.CBPE)

analysis_df = df_test.copy()
analysis_df[target] = df_test[target] if target in df_test else 0  # fallback jeśli brak etykiety

estimated_performance = estimator.estimate(analysis_df)
fig1 = estimated_performance.plot()
fig1.write_image(MODELS_DIR / "performance_estimation.png")

univariate_drift = udc.calculate(analysis_df.drop(columns=["PassengerId", "prediction", "predicted_probability"], errors="ignore"))
for col in feature_columns:
    try:
        fig = univariate_drift.filter(column_names=[col]).plot()
        fig.write_image(MODELS_DIR / f"drift_{col}.png")
    except:
        continue

# Logowanie do MLflow
mlflow.set_experiment("heart_predictions")
with mlflow.start_run():
    mlflow.log_artifact(MODELS_DIR / "heart_predictions.csv")
    mlflow.log_artifact(MODELS_DIR / "heart_shap_summary.png")
    mlflow.log_artifact(MODELS_DIR / "performance_estimation.png")
    mlflow.log_params({"model_uri": model_uri})
