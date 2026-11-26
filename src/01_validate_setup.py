"""Task 1 - validate that the environment and the project data are correctly set up.

Imports every required package, prints its version, runs a tiny scikit-learn model
as a smoke test, and checks that the converted / merged CSV files exist and have the
expected shape.

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import importlib
import sys

from common import CSV_A, CSV_B, FEATURES, MERGED_FILE

PACKAGES = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "reportlab"]


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    for name in PACKAGES:
        module = importlib.import_module(name)
        print(f"  [OK] {name:<12} {getattr(module, '__version__', getattr(module, 'Version', '?'))}")

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    print(f"Smoke-test model accuracy: {LogisticRegression().fit(X, y).score(X, y):.2f}")

    import pandas as pd

    print("\nData files:")
    for path, rows, cols in [(CSV_A, 11055, 31), (CSV_B, 1353, 10), (MERGED_FILE, 12408, 13)]:
        if not path.exists():
            print(f"  [MISSING] {path.name} - run 00a_arff_to_csv.py and 00b_merge_datasets.py first")
            sys.exit(1)
        shape = pd.read_csv(path).shape
        status = "OK" if shape == (rows, cols) else f"UNEXPECTED (wanted {(rows, cols)})"
        print(f"  [{status}] {path.name}: {shape[0]} rows x {shape[1]} columns")
    print(f"  Shared features used for the merge: {len(FEATURES)}")
    print("\nEnvironment validated successfully.")


if __name__ == "__main__":
    main()
