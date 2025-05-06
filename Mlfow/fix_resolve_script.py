
from config import MODEL_NAME
import mlflow
from mlflow.client import MlflowClient
from loguru import logger
from pathlib import Path
import os

# Set MLflow tracking path correctly - this was missing in the original
mlflow_tracking_dir = Path.cwd().parent / 'mlruns'
mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")  # Fixed - proper file URI format

def get_model_by_alias(client, model_name: str = MODEL_NAME, alias: str = "champion"):
    """Pobiera wersję modelu na podstawie aliasu."""
    try:
        logger.info(f"Próba pobrania modelu {model_name} z aliasem {alias}")
        alias_mv = client.get_model_version_by_alias(model_name, alias)
        logger.info(f"Znaleziono model {model_name} w wersji {alias_mv.version} z aliasem {alias}")
        return alias_mv
    except Exception as e:
        logger.warning(f"Nie znaleziono modelu {model_name} z aliasem {alias}: {str(e)}")
        return None


if __name__ == "__main__":
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    client = MlflowClient(mlflow.get_tracking_uri())
    
    # Check if the model registry itself exists first
    try:
        # First, check if the model exists in the registry
        try:
            latest_versions = client.get_latest_versions(MODEL_NAME)
            if not latest_versions:
                logger.error(f"Nie znaleziono modelu {MODEL_NAME} w rejestrze!")
                logger.info("Spróbuj najpierw zarejestrować model używając mlflow.register_model()")
                exit(1)
            logger.info(f"Znaleziono {len(latest_versions)} wersji modelu {MODEL_NAME}")
        except Exception as e:
            if "Registered Model with name" in str(e) and "not found" in str(e):
                logger.error(f"Model {MODEL_NAME} nie istnieje w rejestrze. Najpierw musisz go zarejestrować.")
                logger.info("Przykład: mlflow.register_model(\"runs:/run_id/model_path\", \"heart-disease-prediction\")")
                exit(1)
            else:
                logger.error(f"Błąd podczas sprawdzania modelu {MODEL_NAME}: {str(e)}")
                exit(1)

        # Sprawdź i obsłuż model champion
        champ_mv = get_model_by_alias(client)
        if champ_mv is None:
            chall_mv = get_model_by_alias(client, alias="challenger")
            if chall_mv is None:
                logger.info("Nie znaleziono champion ani challenger, ustawiam najnowszy model jako champion")
                try:
                    latest_version = latest_versions[0].version
                    client.set_registered_model_alias(MODEL_NAME, "champion", latest_version)
                    logger.info(f"Ustawiono najnowszy model (wersja {latest_version}) jako champion")
                except Exception as e:
                    logger.error(f"Błąd podczas ustawiania aliasu champion: {str(e)}")
            else:
                logger.info("Znaleziono challenger bez championa, awansuję challenger na champion")
                try:
                    client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)
                    client.delete_registered_model_alias(MODEL_NAME, "challenger")
                    logger.info(f"Ustawiono model w wersji {chall_mv.version} jako champion i usunięto alias challenger")
                except Exception as e:
                    logger.error(f"Błąd podczas awansowania challenger na champion: {str(e)}")

        # Sprawdź czy istnieje model challenger
        chall_mv = get_model_by_alias(client, alias="challenger")

        # Jeśli istnieje zarówno champion jak i challenger, porównaj ich metryki
        if champ_mv and chall_mv:
            try:
                champ_run = client.get_run(champ_mv.run_id)
                f1_champ = champ_run.data.metrics.get("f1_score", 0)
                logger.info(f"Champion (wersja {champ_mv.version}) ma f1_score: {f1_champ}")

                chall_run = client.get_run(chall_mv.run_id)
                f1_chall = chall_run.data.metrics.get("f1_score", 0)
                logger.info(f"Challenger (wersja {chall_mv.version}) ma f1_score: {f1_chall}")

                if f1_chall >= f1_champ:
                    logger.info("Challenger przewyższa obecnego championa, awansuję challenger na champion")
                    client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)
                    client.delete_registered_model_alias(MODEL_NAME, "challenger")
                    logger.info(f"Ustawiono model w wersji {chall_mv.version} jako champion i usunięto alias challenger")
                else:
                    logger.warning("Challenger nie przewyższa championa, nie dokonuję zmian")
            except Exception as e:
                logger.error(f"Błąd podczas porównywania modeli: {str(e)}")
        elif champ_mv and chall_mv is None:
            logger.info("Nie znaleziono modelu challenger. Kontynuuję z championem.")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd: {str(e)}")