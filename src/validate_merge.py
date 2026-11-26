"""Merge validation - was harmonising Dataset B's feature polarity necessary?

Two checks on the merged data:

1. CROSS-SOURCE TRANSFER. Train on one source, test on the other, using
     (a) "naive" merge  : Dataset B's feature codes exactly as in the raw ARFF file
     (b) "harmonised"   : Dataset B after the polarity correction in 00b_merge_datasets.py
   If the raw codes really meant the same thing in both files, (a) and (b) would score
   alike. A big gap means the naive merge mixes contradictory signals.
   CAVEAT: the polarity signs were chosen using B's labels, so (b) is partly favoured
   by construction. This is a consistency check, not an independent proof.

2. SUSPICIOUS-ROW SENSITIVITY. The 103 "suspicious" sites (Dataset B only) are treated
   as threats in the main analysis. Re-run with them removed to see if that choice matters.

Run:  python src/validate_merge.py
Output: outputs/cross_source_check.csv, outputs/suspicious_sensitivity.csv, plot 12

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import CSV_B, FEATURES, OUT_DIR, PLOT_DIR, RANDOM_STATE, RAW_TARGET, SOURCE_A, SOURCE_B, TARGET, ensure_dirs, load_data
from preprocessing import build_preprocessor, split_data


def pipe(clf):
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    }


def main() -> None:
    ensure_dirs()
    df = load_data()
    a = df[df["source"] == SOURCE_A]
    b_harm = df[df["source"] == SOURCE_B]

    # Dataset B exactly as it appears in the raw file (no polarity correction)
    b_raw = pd.read_csv(CSV_B)
    b_naive = b_raw[FEATURES].copy()
    b_naive[TARGET] = (b_raw[RAW_TARGET] != 1).astype(int)

    rows = []
    for name, clf in models().items():
        for direction, train, tests in [
            ("A -> B", a, {"naive": b_naive, "harmonised": b_harm}),
            ("B -> A", None, None),
        ]:
            if direction == "A -> B":
                p = pipe(clf).fit(train[FEATURES], train[TARGET])
                for variant, test in tests.items():
                    pred = p.predict(test[FEATURES])
                    rows.append(
                        {"model": name, "direction": direction, "variant": variant,
                         "accuracy": accuracy_score(test[TARGET], pred), "f1": f1_score(test[TARGET], pred)}
                    )
            else:
                for variant, train_b in {"naive": b_naive, "harmonised": b_harm}.items():
                    p = pipe(clf).fit(train_b[FEATURES], train_b[TARGET])
                    pred = p.predict(a[FEATURES])
                    rows.append(
                        {"model": name, "direction": direction, "variant": variant,
                         "accuracy": accuracy_score(a[TARGET], pred), "f1": f1_score(a[TARGET], pred)}
                    )
    cross = pd.DataFrame(rows)
    cross.round(4).to_csv(OUT_DIR / "cross_source_check.csv", index=False)
    print("Cross-source transfer:\n", cross.round(4).to_string(index=False))

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, direction in zip(axes, ["A -> B", "B -> A"]):
        sub = cross[cross["direction"] == direction]
        x = np.arange(2)
        for off, variant, c in [(-0.2, "naive", "#9aa5b1"), (0.2, "harmonised", "#2a7ab0")]:
            vals = [sub[(sub.model == m) & (sub.variant == variant)]["accuracy"].iloc[0] for m in models()]
            ax.bar(x + off, vals, 0.4, color=c, label=variant)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(list(models()))
        ax.set_title(f"Train on {direction[0]}, test on {direction[-1]}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    fig.suptitle("Merge check: naive stacking vs polarity-harmonised stacking", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "12_cross_source_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- suspicious-row sensitivity ----
    sens = []
    for label, data in [("suspicious counted as threat (main)", df), ("suspicious rows removed", df[df["result_raw"] != 0])]:
        data = data.reset_index(drop=True)
        Xtr, Xte, ytr, yte = split_data(data)
        for name, clf in {**models(), "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=300, random_state=RANDOM_STATE)}.items():
            p = pipe(clf).fit(Xtr, ytr)
            pred = p.predict(Xte)
            sens.append({"variant": label, "rows": len(data), "model": name,
                         "accuracy": accuracy_score(yte, pred), "f1": f1_score(yte, pred)})
    sens = pd.DataFrame(sens)
    sens.round(4).to_csv(OUT_DIR / "suspicious_sensitivity.csv", index=False)
    print("\nSuspicious-row sensitivity:\n", sens.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
