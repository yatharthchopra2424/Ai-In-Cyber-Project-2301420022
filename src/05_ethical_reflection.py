"""Task 5 - Ethical Reflection and AI Challenges (Simulation + Analysis).

Simulates two real-world operational & ethical risks in AI-driven phishing detection:
  1. FALSE-POSITIVE BLOCKING & THRESHOLD TUNING:
     Varies the decision threshold (0.10 -> 0.90) on the trained Neural Network (MLP)
     to show the direct trade-off between blocking legitimate websites (False Positives)
     and letting phishing attacks slip through (False Negatives).
  2. BIASED TRAINING DATA & UNFAIR DECISIONS:
     Compares a balanced model (trained on A + B) against a biased model trained ONLY
     on Dataset A (excluding Dataset B's distinct web population & suspicious sites),
     measuring the resulting accuracy and false-positive-rate disparity on Dataset B.

Run:  python src/05_ethical_reflection.py
Output: outputs/ethical_threshold_tradeoff.csv, outputs/ethical_bias_simulation.csv,
        outputs/plots/14_threshold_tradeoff.png, outputs/plots/15_bias_simulation.png

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from common import OUT_DIR, PLOT_DIR, RANDOM_STATE, SOURCE_A, SOURCE_B, TARGET, ensure_dirs, load_data
from preprocessing import build_preprocessor, split_data

BLUE, RED, ORANGE = "#2a7ab0", "#d1495b", "#f4a261"


def build_mlp(seed: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    max_iter=300,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def main() -> None:
    ensure_dirs()
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # ==========================================================================
    # 1. Threshold Tuning Simulation (False Positives vs False Negatives)
    # ==========================================================================
    mlp = build_mlp().fit(X_train, y_train)
    probs = mlp.predict_proba(X_test)[:, 1]

    thresholds = np.round(np.arange(0.10, 0.91, 0.10), 2)
    t_rows = []
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        fpr = fp / (fp + tn)  # share of legitimate sites wrongly blocked
        fnr = fn / (fn + tp)  # share of phishing threats missed
        t_rows.append(
            {
                "threshold": float(thr),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "fpr_legit_blocked": fpr,
                "fnr_threats_missed": fnr,
            }
        )
    t_df = pd.DataFrame(t_rows)
    t_df.round(4).to_csv(OUT_DIR / "ethical_threshold_tradeoff.csv", index=False)
    print("=== Threshold Tuning Trade-off (MLP) ===")
    print(t_df[["threshold", "accuracy", "precision", "recall", "false_positives", "false_negatives", "fpr_legit_blocked", "fnr_threats_missed"]].round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(t_df["threshold"], t_df["fpr_legit_blocked"] * 100, marker="o", color=ORANGE, lw=2, label="Legitimate Sites Blocked (FPR %)")
    ax.plot(t_df["threshold"], t_df["fnr_threats_missed"] * 100, marker="s", color=RED, lw=2, label="Phishing Threats Missed (FNR %)")
    ax.plot(t_df["threshold"], t_df["accuracy"] * 100, marker="^", color=BLUE, lw=2, ls="--", label="Overall Accuracy (%)")
    ax.axvline(0.50, color="grey", ls=":", lw=1.2, label="Default Threshold (0.50)")
    ax.set_xlabel("Decision Threshold (Probability >= threshold -> Block as Threat)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Ethical Trade-Off: Blocking Legitimate Users vs. Missing Threats")
    ax.legend(loc="center right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "14_threshold_tradeoff.png", dpi=150)
    plt.close(fig)

    # ==========================================================================
    # 2. Biased Data Simulation (Training on Dataset A only vs Balanced A + B)
    # ==========================================================================
    mask_a_train = X_train["source"] == SOURCE_A
    mlp_biased = build_mlp().fit(X_train[mask_a_train], y_train[mask_a_train])

    b_rows = []
    for scenario, model in [("Balanced Training (A + B)", mlp), ("Biased Training (Dataset A Only)", mlp_biased)]:
        for src_code, src_label in [(SOURCE_A, "Dataset A (UCI 327)"), (SOURCE_B, "Dataset B (UCI 379)")]:
            m = (X_test["source"] == src_code).to_numpy()
            yt = y_test[m]
            yp = model.predict(X_test[m])
            tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
            b_rows.append(
                {
                    "scenario": scenario,
                    "eval_source": src_label,
                    "test_rows": int(m.sum()),
                    "accuracy": accuracy_score(yt, yp),
                    "recall": recall_score(yt, yp),
                    "fpr_legit_blocked": fp / (fp + tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                }
            )
    b_df = pd.DataFrame(b_rows)
    b_df.round(4).to_csv(OUT_DIR / "ethical_bias_simulation.csv", index=False)
    print("\n=== Biased Data Simulation ===")
    print(b_df.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sources = ["Dataset A (UCI 327)", "Dataset B (UCI 379)"]
    x = np.arange(len(sources))
    w = 0.34
    for ax, metric, title, ylabel in [
        (axes[0], "accuracy", "Accuracy Disparity Under Data Bias", "Accuracy (%)"),
        (axes[1], "fpr_legit_blocked", "Unfair Blocking Rate (FPR) Under Data Bias", "Legitimate Sites Blocked (%)"),
    ]:
        bal_vals = [b_df[(b_df.scenario.str.startswith("Balanced")) & (b_df.eval_source == s)][metric].iloc[0] * 100 for s in sources]
        bia_vals = [b_df[(b_df.scenario.str.startswith("Biased")) & (b_df.eval_source == s)][metric].iloc[0] * 100 for s in sources]
        ax.bar(x - w / 2, bal_vals, w, color=BLUE, label="Balanced (A + B)")
        ax.bar(x + w / 2, bia_vals, w, color=RED, label="Biased (A Only)")
        for xi, v in zip(x - w / 2, bal_vals):
            ax.text(xi, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.5)
        for xi, v in zip(x + w / 2, bia_vals):
            ax.text(xi, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sources)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(bal_vals), max(bia_vals)) * 1.18)
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Impact of Underrepresenting a Population in Training Data", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "15_bias_simulation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
