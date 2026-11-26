"""Shared paths, constants and helpers used by every script in this project.

Author : Yatharth Chopra (Roll No. 2301420022)
Course : AI in Cyber Security Lab (ENSP355)
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Project root = parent of src/
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"
REPORT_DIR = ROOT / "report"

# --- source files -----------------------------------------------------------
ARFF_A = RAW_DIR / "Training_Dataset.arff"  # UCI "Phishing Websites"  (id 327)
ARFF_B = RAW_DIR / "PhishingData.arff"      # UCI "Website Phishing"   (id 379)
CSV_A = RAW_DIR / "phishing_websites_uci327.csv"
CSV_B = RAW_DIR / "website_phishing_uci379.csv"
MERGED_FILE = DATA_DIR / "merged_phishing.csv"

SOURCE_A = "UCI327_PhishingWebsites"
SOURCE_B = "UCI379_WebsitePhishing"
SOURCE_SHORT = {SOURCE_A: "Dataset A (UCI 327)", SOURCE_B: "Dataset B (UCI 379)"}

RANDOM_STATE = 42  # fixed seed so every run gives the same results

# The 9 attributes present (by name) in BOTH ARFF files. Original spellings are
# kept on purpose (e.g. "popUpWidnow" is how the UCI files spell it) so every
# column can be traced back to the raw data.
FEATURES = [
    "SFH",
    "popUpWidnow",
    "SSLfinal_State",
    "Request_URL",
    "URL_of_Anchor",
    "web_traffic",
    "URL_Length",
    "age_of_domain",
    "having_IP_Address",
]
RAW_TARGET = "Result"  # 1 = legitimate, 0 = suspicious (only in B), -1 = phishing
TARGET = "label"       # 0 = legitimate, 1 = threat (phishing OR suspicious)

CODE_NAMES = {-1: "Phishy", 0: "Suspicious", 1: "Legitimate"}


def ensure_dirs() -> None:
    for d in (RAW_DIR, PLOT_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load the merged dataset (built by 00b_merge_datasets.py if missing)."""
    if not MERGED_FILE.exists():
        raise FileNotFoundError(
            f"{MERGED_FILE} not found. Run: python src/00a_arff_to_csv.py && python src/00b_merge_datasets.py"
        )
    return pd.read_csv(MERGED_FILE)


def profile_seen_mask(X_train: pd.DataFrame, X_test: pd.DataFrame) -> np.ndarray:
    """True for test rows whose exact 9-feature profile also appears in training.

    Only ~1,500 distinct feature profiles exist across ~12,000 websites, so most
    test rows have an identical twin in the training set. Reporting accuracy on
    the *unseen* profiles shows how well a model generalises rather than memorises.
    """
    train_keys = set(map(tuple, X_train[FEATURES].to_numpy()))
    return np.array([tuple(r) in train_keys for r in X_test[FEATURES].to_numpy()])
