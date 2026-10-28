"""Shared definitions for the youth NEET classification prototype.

Target: youngs == 4 among records where young == 1, as documented in the
supplied LFS 2025 variable labels. Predictors intentionally exclude direct
employment status, current education/training status, and derived NEET flags.
"""
from pathlib import Path

FEATURES = ["A01", "A04", "A05", "Code_UR", "code_dis", "attained", "disable", "hhsize"]
CATEGORICAL_FEATURES = ["A01", "A05", "Code_UR", "code_dis", "attained", "disable"]
NUMERIC_FEATURES = ["A04", "hhsize"]
TARGET_SOURCE = "youngs == 4 among records with young == 1"
TARGET_NAME = "neet"
MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "neet_model.joblib"
METRICS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "neet_model_metrics.json"

DISTRICT_CODES = {
    "Nyarugenge": 11, "Gasabo": 12, "Kicukiro": 13,
    "Nyanza": 21, "Gisagara": 22, "Nyaruguru": 23, "Huye": 24,
    "Nyamagabe": 25, "Ruhango": 26, "Muhanga": 27, "Kamonyi": 28,
    "Karongi": 31, "Rutsiro": 32, "Rubavu": 33, "Nyabihu": 34,
    "Ngororero": 35, "Rusizi": 36, "Nyamasheke": 37,
    "Rulindo": 41, "Gakenke": 42, "Musanze": 43, "Burera": 44,
    "Gicumbi": 45, "Rwamagana": 51, "Nyagatare": 52, "Gatsibo": 53,
    "Kayonza": 54, "Kirehe": 55, "Ngoma": 56, "Bugesera": 57,
    "Nyagatare": 52,
}

SEX_CODES = {"Male": 1, "Female": 2}
RESIDENCE_CODES = {"Urban": 1, "Rural": 2}
EDUCATION_CODES = {
    "No level completed": 1, "Primary": 2, "Lower secondary": 3,
    "Upper secondary": 4, "University": 5,
}
MARITAL_CODES = {
    "Married monogamously": 1, "Married polygamous": 2,
    "Living together": 3, "Divorced": 4, "Separated": 5,
    "Single": 6, "Widow/Widower": 7,
}
