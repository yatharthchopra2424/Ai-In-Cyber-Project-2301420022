"""Smoke tests for the phishing detection capstone pipeline.

Author : Yatharth Chopra (Roll No. 2301420022)
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import FEATURES, MERGED_FILE, OUT_DIR, load_data
from preprocessing import build_preprocessor, split_data


def test_merged_dataset_integrity():
    assert MERGED_FILE.exists(), "merged_phishing.csv must exist"
    df = load_data()
    assert df.shape == (12408, 13)
    assert int(df.isna().sum().sum()) == 0
    for feat in FEATURES:
        assert feat in df.columns
        assert set(df[feat].unique()).issubset({-1, 0, 1})


def test_preprocessor_and_split():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) + len(X_test) == len(df)
    prep = build_preprocessor()
    Xt = prep.fit_transform(X_train)
    assert Xt.shape[1] == len(FEATURES) * 3


def test_outputs_generated():
    expected = [
        "merge_summary.json",
        "eda_summary.json",
        "harmonization_report.csv",
        "ml_results.csv",
        "model_comparison.csv",
        "model_comparison_repeated.csv",
        "ethical_threshold_tradeoff.csv",
        "ethical_bias_simulation.csv",
    ]
    for name in expected:
        assert (OUT_DIR / name).exists(), f"Missing output: {name}"
