"""Task 2 - Cyber security threat landscape analysis (EDA) on the merged phishing data.

Shows how legitimate and threat (phishing / suspicious) websites differ, which
indicators matter most, and how well the two source datasets line up after merging.

Run:  python src/02_eda.py
Output: outputs/plots/01..06_*.png, outputs/eda_summary.txt, outputs/eda_summary.json

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import json

import matplotlib

matplotlib.use("Agg")  # save figures to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

from common import (
    CODE_NAMES,
    FEATURES,
    OUT_DIR,
    PLOT_DIR,
    SOURCE_A,
    SOURCE_B,
    SOURCE_SHORT,
    TARGET,
    ensure_dirs,
    load_data,
)

sns.set_theme(style="whitegrid", context="notebook")
BLUE, RED, GREY = "#2a7ab0", "#d1495b", "#9aa5b1"
CLASS_NAMES = {0: "Legitimate", 1: "Threat"}
CODE_ORDER = [-1, 0, 1]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def main() -> None:
    ensure_dirs()
    df = load_data()
    n = len(df)
    lines, summary = [], {}

    # ---- 1. Dataset overview ----
    counts = df[TARGET].value_counts().sort_index()
    lines.append(f"Rows: {n}   Columns: {df.shape[1]}   Missing values: {int(df.isna().sum().sum())}")
    lines.append(f"Class balance: legitimate={counts[0]} ({counts[0]/n:.1%}), threat={counts[1]} ({counts[1]/n:.1%})")
    by_source = pd.crosstab(df["source"], df["result_raw"])
    lines.append("\nResult code by source (1 legitimate, 0 suspicious, -1 phishing):\n" + by_source.to_string())
    summary["rows"], summary["threat_share"] = n, float(counts[1] / n)
    summary["threat_share_by_source"] = {s: float(v) for s, v in df.groupby("source")[TARGET].mean().items()}

    # ---- Fig 1: class balance + source composition ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar([CLASS_NAMES[i] for i in counts.index], counts.values, color=[BLUE, RED])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 80, f"{v} ({v/n:.0%})", ha="center")
    axes[0].set_title("Legitimate vs threat websites (merged)")
    axes[0].set_ylabel("Websites")
    axes[0].set_ylim(0, counts.max() * 1.12)
    raw_names = {1: "Legitimate", 0: "Suspicious", -1: "Phishing"}
    colors = {1: BLUE, 0: GREY, -1: RED}
    bottoms = np.zeros(2)
    srcs = [SOURCE_A, SOURCE_B]
    for code in [1, 0, -1]:
        vals = np.array([by_source.loc[s].get(code, 0) / by_source.loc[s].sum() * 100 for s in srcs])
        axes[1].bar([SOURCE_SHORT[s] for s in srcs], vals, bottom=bottoms, color=colors[code], label=raw_names[code])
        for i, v in enumerate(vals):
            if v > 4:
                axes[1].text(i, bottoms[i] + v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9)
        bottoms += vals
    axes[1].set_title("Class mix by source dataset")
    axes[1].set_ylabel("% of that dataset's websites")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    save(fig, "01_class_balance_by_source.png")

    # ---- Fig 2: distribution of each feature's codes, by class ----
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, col in zip(axes.ravel(), FEATURES):
        share = pd.crosstab(df[TARGET], df[col], normalize="index").reindex(columns=CODE_ORDER, fill_value=0) * 100
        x = np.arange(3)
        ax.bar(x - 0.2, share.loc[0], 0.4, color=BLUE, label="Legitimate")
        ax.bar(x + 0.2, share.loc[1], 0.4, color=RED, label="Threat")
        ax.set_xticks(x)
        ax.set_xticklabels([CODE_NAMES[c] for c in CODE_ORDER], fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("% of class", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("How often each feature code occurs in legitimate vs threat websites", y=1.01, fontsize=13)
    save(fig, "02_feature_code_distributions.png")

    # ---- Fig 3: threat rate for every feature code ----
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=True)
    for ax, col in zip(axes.ravel(), FEATURES):
        g = df.groupby(col)[TARGET].agg(["mean", "size"]).reindex(CODE_ORDER)
        ax.bar([CODE_NAMES[c] for c in g.index], g["mean"].fillna(0) * 100, color=RED)
        for i, (m, s) in enumerate(zip(g["mean"], g["size"])):
            if pd.notna(m):
                ax.text(i, m * 100 + 2, f"{m:.0%}\n(n={int(s)})", ha="center", fontsize=7)
        ax.set_title(col, fontsize=10)
        ax.set_ylim(0, 118)
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("% threat")
    fig.suptitle("Share of websites that are threats, by feature code", y=1.01, fontsize=13)
    save(fig, "03_threat_rate_by_feature_code.png")

    # ---- Fig 4: number of 'phishy' red flags per site ----
    df["red_flags"] = (df[FEATURES] == -1).sum(axis=1)
    flags = df.groupby("red_flags")[TARGET].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(flags.index, flags["mean"] * 100, color=RED)
    for k, (m, s) in flags.iterrows():
        ax.text(k, m * 100 + 2, f"{m:.0%}\nn={int(s)}", ha="center", fontsize=8)
    ax.set_xlabel("Number of features coded 'phishy' (-1), out of 9")
    ax.set_ylabel("% threat")
    ax.set_ylim(0, 118)
    ax.set_title("Red flags add up: threat rate vs number of phishy indicators")
    save(fig, "04_red_flag_count.png")
    summary["red_flag_rate"] = {int(k): float(v) for k, v in flags["mean"].items()}
    summary["red_flag_n"] = {int(k): int(v) for k, v in flags["size"].items()}

    # ---- Fig 5: correlation heatmap ----
    corr = df[FEATURES + [TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Spearman correlation between features and the threat label")
    save(fig, "05_correlation_heatmap.png")

    # ---- Fig 6: do the two sources line up after harmonisation? ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    w = 0.4
    x = np.arange(len(FEATURES))
    for ax, code, title in zip(axes, [-1, 1], ["Share coded 'phishy' (-1)", "Share coded 'legitimate' (+1)"]):
        for off, s, c in [(-w / 2, SOURCE_A, BLUE), (w / 2, SOURCE_B, RED)]:
            vals = [(df[df["source"] == s][f] == code).mean() * 100 for f in FEATURES]
            ax.bar(x + off, vals, w, color=c, label=SOURCE_SHORT[s])
        ax.set_xticks(x)
        ax.set_xticklabels(FEATURES, rotation=40, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("% of websites")
    axes[0].legend(fontsize=8)
    fig.suptitle("Merge check: feature-code shares per source, after harmonisation", y=1.02, fontsize=13)
    save(fig, "06_source_alignment.png")

    # ---- Which indicators matter most? ----
    mi = pd.Series(mutual_info_classif(df[FEATURES], df[TARGET], discrete_features=True, random_state=42), index=FEATURES)
    mi = mi.sort_values(ascending=False)
    top_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    lines.append("\nSpearman correlation with threat label (negative = value +1 means legitimate):\n" + top_corr.round(3).to_string())
    lines.append("\nMutual information with threat label (nats):\n" + mi.round(3).to_string())
    summary["corr_with_label"] = {k: float(v) for k, v in top_corr.items()}
    summary["mutual_info"] = {k: float(v) for k, v in mi.items()}
    summary["threat_rate_by_code"] = {
        f: {str(c): float(v) for c, v in df.groupby(f)[TARGET].mean().items()} for f in FEATURES
    }

    (OUT_DIR / "eda_summary.txt").write_text("\n".join(lines))
    (OUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
