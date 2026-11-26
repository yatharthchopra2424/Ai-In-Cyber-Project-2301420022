# Runs the whole pipeline in order on Windows PowerShell.
# Usage: .\run_all.ps1
$ErrorActionPreference = "Stop"

python src/00a_arff_to_csv.py
python src/00b_merge_datasets.py
python src/01_validate_setup.py
python src/02_eda.py
python src/03_ml_classification.py
python src/04_dl_comparison.py
python src/validate_merge.py
python src/validate_feature_subset.py
python src/make_report.py
