# Zmodyfikowany config.py
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Załaduj zmienne środowiskowe
load_dotenv()

# Ścieżki
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATASET = "heart"
DATASET_TEST = "heart_test"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_NAME = "heart-disease-prediction"

target = "HeartDisease"

categorical = [
    "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
    "Sex", "Race", "Diabetic", "PhysicalActivity", "GenHealth",
    "Asthma", "KidneyDisease", "SkinCancer", "AgeCategory"
]

