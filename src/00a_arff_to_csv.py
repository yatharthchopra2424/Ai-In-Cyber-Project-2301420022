"""Step 0a - convert the two UCI ARFF files into CSV.

ARFF = Attribute-Relation File Format (Weka). A file has a header
(@relation, @attribute lines) followed by an @data section of comma-separated rows.
This script parses that format with plain Python (no extra dependency), checks the
result against the row/column counts published on the UCI pages, and writes CSVs.

Run:  python src/00a_arff_to_csv.py
Output: data/raw/phishing_websites_uci327.csv, data/raw/website_phishing_uci379.csv

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import pandas as pd

from common import ARFF_A, ARFF_B, CSV_A, CSV_B, ensure_dirs


def read_arff(path) -> tuple[str, pd.DataFrame, dict]:
    """Return (relation_name, dataframe, nominal_value_sets)."""
    relation, columns, nominal, rows, in_data = "", [], {}, [], False
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.strip()
        if not line or line.startswith("%"):  # blank line or ARFF comment
            continue
        low = line.lower()
        if low.startswith("@relation"):
            relation = line.split(None, 1)[1].strip("'\"")
        elif low.startswith("@attribute"):
            name = line.split()[1]
            columns.append(name)
            if "{" in line:  # nominal attribute: {v1,v2,...}
                nominal[name] = [v.strip() for v in line[line.index("{") + 1 : line.index("}")].split(",")]
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append([v.strip() for v in line.split(",")])
    df = pd.DataFrame(rows, columns=columns)
    if (df == "?").any().any():  # '?' is ARFF's missing-value marker
        df = df.replace("?", pd.NA)
    return relation, df.astype(int), nominal


def convert(arff_path, csv_path, expected_rows, expected_cols):
    relation, df, nominal = read_arff(arff_path)
    assert df.shape == (expected_rows, expected_cols), f"{arff_path.name}: got {df.shape}"
    for col, allowed in nominal.items():  # every value must be one the header declares
        bad = set(df[col].astype(str)) - set(allowed)
        assert not bad, f"{col}: undeclared values {bad}"
    df.to_csv(csv_path, index=False)
    print(f"{arff_path.name}: {df.shape[0]} rows x {df.shape[1]} columns -> {csv_path.name}")
    print(f"  relation: {relation[:90]}{'...' if len(relation) > 90 else ''}")
    print(f"  missing values: {int(df.isna().sum().sum())}")
    return df


def main() -> None:
    ensure_dirs()
    # Expected shapes from the UCI pages: 11,055 x (30 features + Result), 1,353 x (9 features + Result)
    convert(ARFF_A, CSV_A, 11055, 31)
    convert(ARFF_B, CSV_B, 1353, 10)


if __name__ == "__main__":
    main()
