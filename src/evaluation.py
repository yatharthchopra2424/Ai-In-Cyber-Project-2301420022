"""Evaluation helpers shared by Task 3 (ML) and Task 4 (DL comparison).

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import SOURCE_SHORT, profile_seen_mask


def score_predictions(X_train, X_test, y_train, y_test, pred, prob, train_time: float, predict_ms: float) -> dict:
    """Compute comprehensive binary classification metrics (positive class = threat)."""
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    seen = profile_seen_mask(X_train, X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": float(roc_auc_score(y_test, prob)) if prob is not None else np.nan,
        "pr_auc": float(average_precision_score(y_test, prob)) if prob is not None else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_time_s": train_time,
        "predict_time_ms": predict_ms,
        "acc_seen_profiles": accuracy_score(y_test[seen], pred[seen]) if seen.any() else np.nan,
        "acc_unseen_profiles": accuracy_score(y_test[~seen], pred[~seen]) if (~seen).any() else np.nan,
        "share_unseen_profiles": float((~seen).mean()),
    }


def fit_and_score(pipe, X_train, X_test, y_train, y_test) -> tuple[dict, np.ndarray]:
    """Fit a pipeline, time it, and return metrics (positive class = threat) plus predictions."""
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = pipe.predict(X_test)
    predict_ms = (time.perf_counter() - t0) * 1000
    prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    return score_predictions(X_train, X_test, y_train, y_test, pred, prob, train_time, predict_ms), pred


def metrics_by_source(X_test, y_test, pred) -> list[dict]:
    rows = []
    for src, label in SOURCE_SHORT.items():
        m = (X_test["source"] == src).to_numpy()
        rows.append(
            {
                "source": label,
                "test_rows": int(m.sum()),
                "accuracy": accuracy_score(y_test[m], pred[m]),
                "precision": precision_score(y_test[m], pred[m]),
                "recall": recall_score(y_test[m], pred[m]),
                "f1": f1_score(y_test[m], pred[m]),
            }
        )
    return rows
