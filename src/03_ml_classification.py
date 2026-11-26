"""Task 3 - Machine-learning threat classification on the merged data.

Trains Logistic Regression, Decision Tree, and Random Forest to classify websites as
legitimate (0) or threat (1 = phishing or suspicious), then evaluates with accuracy,
precision, recall, F1, and ROC-AUC, overall and separately for each source dataset.

Run:  python src/03_ml_classification.py
Output: outputs/ml_results.csv, outputs/ml_results_by_source.csv, plots 07 and 08

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import FEATURES, OUT_DIR, PLOT_DIR, RANDOM_STATE, ensure_dirs, load_data
from evaluation import fit_and_score, metrics_by_source
from preprocessing import build_preprocessor, split_data


def aggregate_by_feature(values: np.ndarray, names) -> pd.Series:
    """Sum one-hot column scores back to the original 9 features."""
    s = pd.Series(values, index=names)
    return s.groupby(lambda n: next(f for f in FEATURES if f in n)).sum()


def main() -> None:
    ensure_dirs()
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | threat share train {y_train.mean():.3f}, test {y_test.mean():.3f}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
    }

    results, by_source, fitted = [], [], {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, clf) in zip(axes, models.items()):
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        res, pred = fit_and_score(pipe, X_train, X_test, y_train, y_test)
        results.append({"model": name, **res})
        by_source += [{"model": name, **r} for r in metrics_by_source(X_test, y_test, pred)]
        fitted[name] = pipe
        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, target_names=["legitimate", "threat"], digits=3))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["legitimate", "threat"], cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_ml_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Which of the 9 indicators does each model rely on?
    names = fitted["Decision Tree"].named_steps["prep"].get_feature_names_out()
    tree_imp = aggregate_by_feature(fitted["Decision Tree"].named_steps["clf"].feature_importances_, names)
    rf_imp = aggregate_by_feature(fitted["Random Forest"].named_steps["clf"].feature_importances_, names)
    lr_imp = aggregate_by_feature(np.abs(fitted["Logistic Regression"].named_steps["clf"].coef_[0]), names)
    lr_imp = lr_imp / lr_imp.sum()
    imp = pd.DataFrame({
        "Decision Tree (importance)": tree_imp,
        "Random Forest (importance)": rf_imp,
        "Logistic Regression (share of |coef|)": lr_imp,
    })
    imp = imp.sort_values("Random Forest (importance)")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    imp.plot.barh(ax=ax, color=["#2a7ab0", "#f4a261", "#7fb069"])
    ax.set_title("Which indicators drive the classical ML models (summed over one-hot codes)")
    ax.set_xlabel("Relative weight")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_feature_importance.png", dpi=150)
    plt.close(fig)
    imp.round(4).to_csv(OUT_DIR / "feature_importance.csv")

    table = pd.DataFrame(results)
    table.round(4).to_csv(OUT_DIR / "ml_results.csv", index=False)
    pd.DataFrame(by_source).round(4).to_csv(OUT_DIR / "ml_results_by_source.csv", index=False)
    print("\nSummary:\n", table.round(4).to_string(index=False))
    print("\nBy source:\n", pd.DataFrame(by_source).round(4).to_string(index=False))
    print("\nTop indicators (Decision Tree):\n", imp["Decision Tree (importance)"].iloc[::-1].head(4).round(3).to_string())


if __name__ == "__main__":
    main()
