"""Reusable preprocessing pipeline (Task 3) shared by the ML and DL scripts.

All 9 features are CATEGORY CODES, not measurements: -1 = phishy, 0 = suspicious,
1 = legitimate. Treating them as numbers would wrongly imply that "legitimate" is
twice as far from "suspicious" as "phishy" is, so they are one-hot encoded.

Steps (fitted on the training split only, to avoid data leakage):
  * Missing values -> most-frequent imputation. The merged data has none (checked in
    00b_merge_datasets.py), so this step is a safety net that changes nothing.
  * Categorical codes -> one-hot encoding with the three possible codes fixed up
    front, so a code absent from the training split cannot shift the columns.
  * Normalisation -> not needed: one-hot columns are already 0/1 on a common scale.
    (The MLP, which is scale-sensitive, therefore needs no StandardScaler here.)
The `source` column is deliberately NOT a model input: the model must learn what
makes a site look phishy, not which dataset it came from.

Author : Yatharth Chopra (Roll No. 2301420022)
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import FEATURES, RANDOM_STATE, TARGET


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(categories=[[-1, 0, 1]] * len(FEATURES), handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("cat", categorical, FEATURES)])


def split_data(df, test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Stratified 80/20 split on label x source, so each half keeps the same mix of
    legitimate/threat rows AND the same share of Dataset A / Dataset B rows."""
    X = df[FEATURES + ["source"]]
    y = df[TARGET]
    strata = df[TARGET].astype(str) + "_" + df["source"]
    return train_test_split(X, y, test_size=test_size, stratify=strata, random_state=seed)
