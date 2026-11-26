"""Task 4 - Introduction to deep learning: neural network vs classical ML.

Trains a scikit-learn MLPClassifier (a small feed-forward neural network) on the same
preprocessed merged data as Task 3 and compares it with Logistic Regression, Decision Tree,
and Random Forest on detection quality (including ROC-AUC), training time, and prediction speed.
Because the top models score closely, the comparison is also repeated over 5 different random
train/test splits (mean +/- std).

Run:  python src/04_dl_comparison.py
Output: outputs/model_comparison.csv, outputs/model_comparison_repeated.csv, plots 09-11 & 13

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import OUT_DIR, PLOT_DIR, RANDOM_STATE, ensure_dirs, load_data
from evaluation import fit_and_score, metrics_by_source
from preprocessing import build_preprocessor, split_data

COLORS = ["#2a7ab0", "#7fb069", "#f4a261", "#d1495b"]


def make_models(seed: int) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1),
        # Neural network: 2 hidden layers (32 -> 16 neurons), ReLU, Adam optimiser;
        # early stopping holds out 10% of the training data to stop before overfitting.
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        ),
    }


def main() -> None:
    ensure_dirs()
    df = load_data()

    # ---------- main split (same split as Task 3) ----------
    X_train, X_test, y_train, y_test = split_data(df)
    rows, by_source, fitted_pipes, mlp_pipe = [], [], {}, None
    for name, clf in make_models(RANDOM_STATE).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        res, pred = fit_and_score(pipe, X_train, X_test, y_train, y_test)
        rows.append({"model": name, **res})
        by_source += [{"model": name, **r} for r in metrics_by_source(X_test, y_test, pred)]
        fitted_pipes[name] = pipe
        if name.startswith("Neural"):
            mlp_pipe = pipe
    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print(table.round(4).to_string(index=False))
    pd.DataFrame(by_source).round(4).to_csv(OUT_DIR / "model_comparison_by_source.csv", index=False)
    print("\nBy source:\n", pd.DataFrame(by_source).round(4).to_string(index=False))

    # ---------- repeated random splits ----------
    reps = []
    for seed in range(RANDOM_STATE, RANDOM_STATE + 5):
        Xtr, Xte, ytr, yte = split_data(df, seed=seed)
        for name, clf in make_models(seed).items():
            pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
            res, _ = fit_and_score(pipe, Xtr, Xte, ytr, yte)
            reps.append({"model": name, "seed": seed, **res})
    reps = pd.DataFrame(reps)
    agg = reps.groupby("model", sort=False)[["accuracy", "f1", "recall", "roc_auc", "acc_unseen_profiles", "train_time_s"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.round(4).to_csv(OUT_DIR / "model_comparison_repeated.csv")
    print("\nRepeated over 5 splits (mean/std):\n", agg.round(4).to_string())

    # ---------- plots ----------
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    n_models = len(table)
    x, width = np.arange(len(metrics)), 0.19
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, row in table.iterrows():
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, [row[m] for m in metrics], width, label=row["model"], color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Classical ML vs Neural Network - detection performance (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_model_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(table["model"], table["train_time_s"], color=COLORS)
    axes[0].set_title("Training time (s)")
    axes[1].bar(table["model"], table["predict_time_ms"], color=COLORS)
    axes[1].set_title("Prediction time on test set (ms)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_training_time.png", dpi=150)
    plt.close(fig)

    mlp = mlp_pipe.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mlp.loss_curve_, label="Training loss", color="#d1495b")
    ax.plot(1 - np.array(mlp.validation_scores_), label="Validation error (1 - accuracy)", color="#2a7ab0")
    ax.set_xlabel("Epoch")
    ax.set_title(f"MLP learning curve (stopped after {mlp.n_iter_} epochs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "11_mlp_learning_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for (name, pipe), color in zip(fitted_pipes.items(), COLORS):
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax, color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.50)")
    ax.set_title("ROC Curves - Classical ML vs Neural Network (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "13_roc_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
