"""Merge validation - do the 3 hard-to-verify features change the conclusions?

Of the 9 shared features, three cannot have their coding checked against Dataset A:
  * popUpWidnow        - A uses {-1, 1}, B uses {-1, 0, 1} (different value sets) and its
                         correlation with the label in A is 0.00
  * having_IP_Address  - A uses {-1, 1}, B uses {0, 1}, and its correlation in A is only 0.09
  * URL_Length         - same value set, but the correlation in A is only 0.05
For these, the direction of B's coding was inferred from B alone (see 00b_merge_datasets.py).
This script re-runs the three models with only the 6 features whose direction can be
confirmed in BOTH files, and compares with the 9-feature result.

Run:  python src/validate_feature_subset.py
Output: outputs/feature_subset_sensitivity.csv

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from common import FEATURES, OUT_DIR, RANDOM_STATE, SOURCE_A, SOURCE_B, TARGET, load_data
from preprocessing import split_data

UNVERIFIABLE = ["popUpWidnow", "URL_Length", "having_IP_Address"]
VERIFIED = [f for f in FEATURES if f not in UNVERIFIABLE]


def make_pipe(features, clf):
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(categories=[[-1, 0, 1]] * len(features), handle_unknown="ignore")),
        ]
    )
    return Pipeline([("prep", ColumnTransformer([("cat", cat, features)])), ("clf", clf)])


def models(seed):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=300, random_state=seed
        ),
    }


def main() -> None:
    df = load_data()
    a, b = df[df["source"] == SOURCE_A], df[df["source"] == SOURCE_B]
    rows = []
    for label, feats in [("9 shared features (main)", FEATURES), ("6 verifiable features", VERIFIED)]:
        for seed in range(RANDOM_STATE, RANDOM_STATE + 5):  # same 5 splits as Task 4
            Xtr, Xte, ytr, yte = split_data(df, seed=seed)
            for name, clf in models(seed).items():
                pred = make_pipe(feats, clf).fit(Xtr, ytr).predict(Xte)
                rows.append({"features": label, "model": name, "seed": seed,
                             "accuracy": accuracy_score(yte, pred), "f1": f1_score(yte, pred)})
    res = pd.DataFrame(rows)
    out = res.groupby(["features", "model"], sort=False)[["accuracy", "f1"]].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]

    # cross-source transfer (A -> B and B -> A) with a Decision Tree and Logistic Regression
    cross = []
    for label, feats in [("9 shared features (main)", FEATURES), ("6 verifiable features", VERIFIED)]:
        for name in ["Logistic Regression", "Decision Tree"]:
            for direction, tr, te in [("A -> B", a, b), ("B -> A", b, a)]:
                pipe = make_pipe(feats, models(RANDOM_STATE)[name]).fit(tr[feats], tr[TARGET])
                cross.append({"features": label, "model": name, "direction": direction,
                              "cross_source_accuracy": accuracy_score(te[TARGET], pipe.predict(te[feats]))})
    cross = pd.DataFrame(cross).pivot_table(index=["features", "model"], columns="direction",
                                            values="cross_source_accuracy", sort=False)
    cross.columns = [f"cross_source_acc_{c.replace(' -> ', 'to')}" for c in cross.columns]
    final = out.join(cross).round(4)
    final.to_csv(OUT_DIR / "feature_subset_sensitivity.csv")
    print(final.to_string())


if __name__ == "__main__":
    main()
