"""Task 4 - Deep Learning vs Classical Machine Learning Benchmark.

Compares four classical ML models (Logistic Regression, Decision Tree, Random Forest,
and HistGradientBoosting) against a PyTorch Deep Neural Network (PhishingDNN:
27 -> 128 -> Dropout(0.3) -> 64 -> Dropout(0.2) -> 32 -> 1 with BCEWithLogitsLoss,
Adam, and validation early stopping) on Accuracy, Precision, Recall, F1, ROC-AUC,
PR-AUC, confusion counts (TN/FP/FN/TP), training time, and prediction latency.

Run:  python src/04_dl_comparison.py
Output: outputs/model_comparison.csv, outputs/model_comparison_repeated.csv,
        models/phishing_dnn.pt, plots 09-11 & 13

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset

from common import MODEL_DIR, OUT_DIR, PLOT_DIR, RANDOM_STATE, ensure_dirs, load_data
from evaluation import fit_and_score, metrics_by_source, score_predictions
from preprocessing import build_preprocessor, split_data

COLORS = ["#2a7ab0", "#7fb069", "#f4a261", "#8e6c8a", "#d1495b"]


class PhishingDNN(nn.Module):
    """3-hidden-layer PyTorch neural network for tabular phishing detection."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_pytorch_dnn(X_train, X_test, y_train, y_test, seed: int = RANDOM_STATE, epochs: int = 40, batch_size: int = 256):
    """Train PhishingDNN with a 15% validation split and early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    prep = build_preprocessor()
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
    )
    X_fit_arr = np.asarray(prep.fit_transform(X_fit).toarray() if hasattr(prep.fit_transform(X_fit), "toarray") else prep.fit_transform(X_fit), dtype=np.float32)
    X_val_arr = np.asarray(prep.transform(X_val).toarray() if hasattr(prep.transform(X_val), "toarray") else prep.transform(X_val), dtype=np.float32)
    X_test_arr = np.asarray(prep.transform(X_test).toarray() if hasattr(prep.transform(X_test), "toarray") else prep.transform(X_test), dtype=np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_fit_arr), torch.tensor(y_fit.to_numpy(), dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = PhishingDNN(X_fit_arr.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.perf_counter()
    train_losses, val_losses, val_accs = [], [], []
    best_val_loss, best_state, patience, wait = float("inf"), None, 6, 0

    X_val_t = torch.from_numpy(X_val_arr)
    y_val_t = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            vloss = float(criterion(val_logits, y_val_t).item())
            vacc = float(((torch.sigmoid(val_logits) >= 0.5).numpy() == y_val.to_numpy()).mean())
        val_losses.append(vloss)
        val_accs.append(vacc)

        if vloss < best_val_loss - 1e-4:
            best_val_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_test_arr))).numpy()
        preds = (probs >= 0.5).astype(int)
    predict_ms = (time.perf_counter() - t0) * 1000

    res = score_predictions(X_train, X_test, y_train, y_test, preds, probs, train_time, predict_ms)
    history = {"train_loss": train_losses, "val_loss": val_losses, "val_acc": val_accs}
    return model, res, preds, probs, history


def make_classical_models(seed: int) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, max_depth=8, learning_rate=0.08, random_state=seed),
    }


def main() -> None:
    ensure_dirs()
    df = load_data()

    # ---------- main split (same split as Task 3) ----------
    X_train, X_test, y_train, y_test = split_data(df)
    rows, by_source, fitted_pipes = [], [], {}
    for name, clf in make_classical_models(RANDOM_STATE).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        res, pred = fit_and_score(pipe, X_train, X_test, y_train, y_test)
        rows.append({"model": name, **res})
        by_source += [{"model": name, **r} for r in metrics_by_source(X_test, y_test, pred)]
        fitted_pipes[name] = pipe

    dnn_model, dnn_res, dnn_pred, dnn_prob, dnn_hist = train_pytorch_dnn(X_train, X_test, y_train, y_test, RANDOM_STATE)
    rows.append({"model": "PyTorch DNN", **dnn_res})
    by_source += [{"model": "PyTorch DNN", **r} for r in metrics_by_source(X_test, y_test, dnn_pred)]
    torch.save(dnn_model.state_dict(), MODEL_DIR / "phishing_dnn.pt")

    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print(table[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_time_s"]].round(4).to_string(index=False))
    pd.DataFrame(by_source).round(4).to_csv(OUT_DIR / "model_comparison_by_source.csv", index=False)

    # ---------- repeated random splits ----------
    reps = []
    for seed in range(RANDOM_STATE, RANDOM_STATE + 5):
        Xtr, Xte, ytr, yte = split_data(df, seed=seed)
        for name, clf in make_classical_models(seed).items():
            pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
            res, _ = fit_and_score(pipe, Xtr, Xte, ytr, yte)
            reps.append({"model": name, "seed": seed, **res})
        _, d_res, _, _, _ = train_pytorch_dnn(Xtr, Xte, ytr, yte, seed=seed)
        reps.append({"model": "PyTorch DNN", "seed": seed, **d_res})

    reps_df = pd.DataFrame(reps)
    agg = reps_df.groupby("model", sort=False)[["accuracy", "f1", "recall", "roc_auc", "pr_auc", "acc_unseen_profiles", "train_time_s"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.round(4).to_csv(OUT_DIR / "model_comparison_repeated.csv")
    print("\nRepeated over 5 splits (mean/std):\n", agg.round(4).to_string())

    # ---------- plots ----------
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    n_models = len(table)
    x, width = np.arange(len(metrics)), 0.15
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    for i, row in table.iterrows():
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, [row[m] for m in metrics], width, label=row["model"], color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Classical ML vs PyTorch Deep Neural Network - Test Set Performance")
    ax.legend(loc="lower right", ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_model_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].bar(table["model"], table["train_time_s"], color=COLORS)
    axes[0].set_title("Training time (s)")
    axes[1].bar(table["model"], table["predict_time_ms"], color=COLORS)
    axes[1].set_title("Prediction time on test set (ms)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_training_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    epochs_range = np.arange(1, len(dnn_hist["train_loss"]) + 1)
    ax.plot(epochs_range, dnn_hist["train_loss"], label="Training Loss (BCE)", color="#d1495b", lw=2)
    ax.plot(epochs_range, dnn_hist["val_loss"], label="Validation Loss (BCE)", color="#2a7ab0", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title(f"PyTorch DNN Learning Curve (Early Stopping at Epoch {len(epochs_range)})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "11_mlp_learning_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    for (name, pipe), color in zip(fitted_pipes.items(), COLORS[:-1]):
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax, color=color)
    fpr_d, tpr_d, _ = roc_curve(y_test, dnn_prob)
    ax.plot(fpr_d, tpr_d, color=COLORS[-1], lw=2, label=f"PyTorch DNN (AUC = {auc(fpr_d, tpr_d):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.50)")
    ax.set_title("ROC Curves - Classical ML vs PyTorch DNN (test set)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "13_roc_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
