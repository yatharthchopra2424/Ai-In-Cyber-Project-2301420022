"""Step 0b - merge the two phishing datasets on their common features.

HOW (AND WHY) THE DATASETS ARE JOINED
-------------------------------------
* Neither file has a website ID, URL or timestamp, so there is no key to match
  rows one-to-one. A relational (key) join is therefore impossible, and unnecessary:
  the two files describe DIFFERENT websites. The correct merge is a ROW-WISE UNION
  (stack the rows) on the attributes both files share.
* Join basis = the SCHEMA: the 9 attributes whose name is identical in both ARFF
  headers (the associated papers share a co-author, which points to one feature family).
  The 21 attributes only in Dataset A are dropped (they would be empty for all
  1,353 rows of Dataset B).
* Labels: the UCI 379 page documents 1 = legitimate, 0 = suspicious, -1 = phishing
  and its class counts (548 / 702 / 103) match the file; Dataset A uses the same
  convention (its features correlate positively with its label). Labels are kept as documented.
* Feature polarity: under that convention a feature value of +1 means "looks legitimate".
  That implies every feature should correlate POSITIVELY with the label. It does in
  Dataset A. In Dataset B 8 of the 9 features correlate NEGATIVELY, i.e. they are
  coded in the opposite direction. Stacking the files unchanged would make the same
  code mean opposite things in the two halves. So, for B only, each feature whose
  Spearman correlation with B's own label is negative gets its polarity reversed
  (x -> -x). having_IP_Address is {0,1} in B but {-1,1} in A, so it is mapped onto
  {-1,1} using the same rule.
  This step uses B's labels (9 sign decisions); it is a documented harmonisation
  choice, listed in outputs/harmonization_report.csv and validated in validate_merge.py.
* Target: label = 0 for legitimate, 1 for phishing OR suspicious (anything an analyst
  would not wave through). The original 3-way code is kept in `result_raw`.

Run:  python src/00b_merge_datasets.py
Output: data/merged_phishing.csv, outputs/harmonization_report.csv, outputs/merge_summary.json

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import json

import pandas as pd

from common import (
    CSV_A,
    CSV_B,
    FEATURES,
    MERGED_FILE,
    OUT_DIR,
    RAW_TARGET,
    SOURCE_A,
    SOURCE_B,
    TARGET,
    ensure_dirs,
)


def spearman_with_label(df: pd.DataFrame, col: str) -> float:
    """Spearman correlation of a feature with Result, ignoring 'suspicious' rows."""
    clear = df[df[RAW_TARGET] != 0]
    return float(clear[col].corr(clear[RAW_TARGET], method="spearman"))


def harmonise_b(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reverse the polarity of B's features that contradict the documented coding."""
    b = b.copy()
    rows = []
    for col in FEATURES:
        corr_raw = spearman_with_label(b, col)
        flip = corr_raw < 0
        if col == "having_IP_Address":  # B stores {0,1}; A stores {-1,1}
            b[col] = b[col].map({1: -1, 0: 1} if flip else {0: -1, 1: 1})
            action = "recoded {0,1} -> {-1,1}, polarity reversed" if flip else "recoded {0,1} -> {-1,1}"
        elif flip:
            b[col] = -b[col]
            action = "polarity reversed (x -> -x)"
        else:
            action = "kept as is"
        rows.append({"feature": col, "corr_B_raw": corr_raw, "action": action, "corr_B_after": spearman_with_label(b, col)})
    return b, pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    a_full, b_full = pd.read_csv(CSV_A), pd.read_csv(CSV_B)

    shared = [c for c in b_full.columns if c in a_full.columns and c != RAW_TARGET]
    assert shared == FEATURES, f"shared attributes differ from expected: {shared}"
    only_a = [c for c in a_full.columns if c not in b_full.columns]
    print(f"Shared attributes ({len(shared)}): {shared}")
    print(f"Dropped (only in A, {len(only_a)} features): {only_a}")

    a = a_full[FEATURES + [RAW_TARGET]].copy()
    b, report = harmonise_b(b_full[FEATURES + [RAW_TARGET]])
    report.insert(1, "corr_A", [spearman_with_label(a, c) for c in report["feature"]])
    report["share_minus1_A"] = [(a[c] == -1).mean() for c in report["feature"]]
    report["share_minus1_B_after"] = [(b[c] == -1).mean() for c in report["feature"]]
    report.round(3).to_csv(OUT_DIR / "harmonization_report.csv", index=False)
    print("\nHarmonisation of Dataset B:\n", report.round(3).to_string(index=False))

    a.insert(0, "source", SOURCE_A)
    b.insert(0, "source", SOURCE_B)
    merged = pd.concat([a, b], ignore_index=True)
    merged = merged.rename(columns={RAW_TARGET: "result_raw"})
    merged[TARGET] = (merged["result_raw"] != 1).astype(int)  # 1 = phishing or suspicious
    merged.insert(0, "record_id", range(1, len(merged) + 1))
    merged.to_csv(MERGED_FILE, index=False)

    # ---- summary numbers used by the report ----
    profiles = merged.groupby(FEATURES)[TARGET].agg(["size", "nunique", "sum"])
    lookup_ceiling = float(
        pd.concat([profiles["sum"], profiles["size"] - profiles["sum"]], axis=1).max(axis=1).sum() / len(merged)
    )
    summary = {
        "rows_total": int(len(merged)),
        "rows_A": int(len(a)),
        "rows_B": int(len(b)),
        "shared_features": shared,
        "features_dropped_from_A": int(len(only_a)),
        "label_counts": {str(k): int(v) for k, v in merged[TARGET].value_counts().sort_index().items()},
        "result_raw_counts": {str(k): int(v) for k, v in merged["result_raw"].value_counts().sort_index().items()},
        "missing_values": int(merged.isna().sum().sum()),
        "duplicate_rows_feature_label": int(merged.duplicated(FEATURES + [TARGET]).sum()),
        "unique_feature_profiles": int(len(profiles)),
        "profiles_with_mixed_labels": int((profiles["nunique"] > 1).sum()),
        "lookup_table_ceiling_accuracy": lookup_ceiling,
        "profiles_in_both_sources": int(
            len(set(map(tuple, a[FEATURES].to_numpy())) & set(map(tuple, b[FEATURES].to_numpy())))
        ),
    }
    (OUT_DIR / "merge_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nMerged:", merged.shape, "->", MERGED_FILE)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
