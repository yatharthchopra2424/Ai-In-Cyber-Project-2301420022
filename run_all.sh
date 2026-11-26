#!/usr/bin/env bash
# Runs the whole pipeline in order. Usage: bash run_all.sh   (from the project root, venv active)
set -euo pipefail
python src/00a_arff_to_csv.py        # ARFF -> CSV (data/raw/)
python src/00b_merge_datasets.py     # merge on common features -> data/merged_phishing.csv
python src/01_validate_setup.py      # Task 1
python src/02_eda.py                 # Task 2
python src/03_ml_classification.py   # Task 3
python src/04_dl_comparison.py       # Task 4
python src/05_ethical_reflection.py  # Task 5
python src/validate_merge.py         # merge validation + sensitivity
python src/validate_feature_subset.py # 6 vs 9 feature sensitivity
python src/make_report.py            # report/Technical_Report.pdf
python src/make_word_report.py       # report/Capstone_Project_Report_Yatharth_Chopra.docx
