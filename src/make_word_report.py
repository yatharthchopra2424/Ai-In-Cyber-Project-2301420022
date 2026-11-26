"""Build a print-ready Word (.docx) Capstone Project Report from saved results, plots, and notebook screenshots.

Run AFTER all pipeline scripts and notebook screenshot capture:
    python src/make_word_report.py
Output: report/Capstone_Project_Report_Yatharth_Chopra.docx

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

from common import OUT_DIR, PLOT_DIR, REPORT_DIR, ensure_dirs

SHOT_DIR = OUT_DIR / "screenshots"
NAVY = RGBColor(0x1B, 0x3A, 0x5C)
ACCENT = RGBColor(0x2A, 0x7A, 0xB0)
MUTED = RGBColor(0x55, 0x5E, 0x68)


def set_cell_bg(cell, hex_color: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def add_styled_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = ""
        run = hdr_cells[i].paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(hdr_cells[i], "2A7AB0")

    for r_idx, row in enumerate(rows):
        row_cells = table.add_row().cells
        bg = "F0F4F8" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = ""
            run = row_cells[c_idx].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
            set_cell_bg(row_cells[c_idx], bg)
    doc.add_paragraph()


def add_figure(doc: Document, path: Path, caption: str, width_in: float = 6.0) -> None:
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width_in))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = MUTED


def add_heading_1(doc: Document, text: str) -> None:
    h = doc.add_heading(text, level=1)
    for run in h.runs:
        run.font.color.rgb = NAVY
        run.font.size = Pt(15)


def add_heading_2(doc: Document, text: str) -> None:
    h = doc.add_heading(text, level=2)
    for run in h.runs:
        run.font.color.rgb = ACCENT
        run.font.size = Pt(12)


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def main() -> None:
    ensure_dirs()
    ms = json.loads((OUT_DIR / "merge_summary.json").read_text())
    eda = json.loads((OUT_DIR / "eda_summary.json").read_text())
    harm = pd.read_csv(OUT_DIR / "harmonization_report.csv")
    ml = pd.read_csv(OUT_DIR / "ml_results.csv").set_index("model")
    cmp_ = pd.read_csv(OUT_DIR / "model_comparison.csv").set_index("model")
    rep = pd.read_csv(OUT_DIR / "model_comparison_repeated.csv").set_index("model")
    cross = pd.read_csv(OUT_DIR / "cross_source_check.csv")
    sens = pd.read_csv(OUT_DIR / "suspicious_sensitivity.csv")

    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        footer_p = section.footer.paragraphs[0]
        footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fr = footer_p.add_run("Yatharth Chopra | Roll No. 2301420022 | ENSP355 Capstone Project (Assignments 1-4)")
        fr.font.size = Pt(8)
        fr.font.color.rgb = MUTED

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    # ==================== COVER PAGE ====================
    for _ in range(3):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = t.add_run("AI-Driven Cyber Threat Awareness\nand Phishing Detection Prototype")
    tr.bold = True
    tr.font.size = Pt(24)
    tr.font.color.rgb = NAVY

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Capstone Project Report — Lab Assignments 1 to 4 (Unit 1)\nPhishing Website Detection on Two Harmonised UCI Datasets")
    sr.font.size = Pt(12.5)
    sr.font.color.rgb = ACCENT

    doc.add_paragraph()
    meta_lines = [
        ("Course:", "AI in Cyber Security Lab (ENSP355)"),
        ("University:", "K.R. Mangalam University"),
        ("Student Name:", "Yatharth Chopra"),
        ("Roll No.:", "2301420022"),
        ("Program:", "B.Tech CSE (Data Science), Semester 7"),
        ("Submitted To:", "Monika Khatkar"),
        ("GitHub Repository:", "github.com/yatharthchopra2424/Ai-In-Cyber-Project-2301420022"),
    ]
    for label, val in meta_lines:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p.add_run(f"{label} ")
        r1.bold = True
        r2 = p.add_run(val)
        r1.font.size = r2.font.size = Pt(11)

    doc.add_page_break()

    # ==================== 1. PROBLEM OVERVIEW ====================
    add_heading_1(doc, "1. Problem Overview & Objectives")
    doc.add_paragraph(
        "Phishing is one of the most common initial-access vectors in cybersecurity: attackers deploy spoofed "
        "websites that mimic trusted services to harvest user credentials. Static blocklists and rigid heuristic "
        "rules only detect previously reported domains, whereas supervised Machine Learning and Deep Learning "
        "models learn combinations of URL, SSL/TLS, and HTML indicators to flag newly created threats."
    )
    doc.add_paragraph(
        "This capstone project executes a complete, reproducible cybersecurity data-science workflow across four "
        "assignments: (1) environment validation, ARFF-to-CSV conversion, and schema harmonisation; (2) exploratory "
        "threat-landscape analysis; (3) classical machine-learning classification (Logistic Regression, Decision Tree, "
        "Random Forest); and (4) deep learning (MLP) comparison, ROC-AUC evaluation, 5-split stability testing, and "
        "merge sensitivity validation."
    )

    # ==================== 2. ASSIGNMENT 1 ====================
    add_heading_1(doc, "2. Assignment 1 — Environment Setup, Data Ingestion & Schema Harmonisation")
    n, nA, nB = ms["rows_total"], ms["rows_A"], ms["rows_B"]
    n_leg, n_thr = ms["label_counts"]["0"], ms["label_counts"]["1"]

    add_styled_table(
        doc,
        ["Property", "Dataset A (UCI 327)", "Dataset B (UCI 379)"],
        [
            ["Dataset Name", "Phishing Websites (327)", "Website Phishing (379)"],
            ["DOI", "10.24432/C51W2X", "10.24432/C5B301"],
            ["Websites (Rows)", f"{nA:,}", f"{nB:,}"],
            ["Attributes (excl. label)", "30", "9"],
            ["Missing Values", "0", "0"],
            ["Raw Classes", "Legitimate: 6,157 | Phishing: 4,898", "Legitimate: 548 | Phishing: 702 | Suspicious: 103"],
        ],
    )

    doc.add_paragraph(
        "Because neither file contains a URL or timestamp key and the two collections represent distinct websites, "
        "they were combined via a row-wise union on the 9 shared attributes. Spearman correlation checks revealed "
        "that 8 of the 9 features in Dataset B were coded with inverted polarity relative to Dataset A (and "
        "having_IP_Address used {0, 1} instead of {-1, 1}). Polarity was harmonised in Dataset B prior to merging, "
        f"producing {n:,} labelled rows ({n_leg:,} legitimate, {pct(n_leg/n)}; {n_thr:,} threats, {pct(n_thr/n)}) "
        f"across {ms['unique_feature_profiles']:,} unique 9-feature profiles (theoretical lookup ceiling: {pct(ms['lookup_table_ceiling_accuracy'])})."
    )

    harm_rows = [
        [r["feature"], f"{r['corr_A']:+.3f}", f"{r['corr_B_raw']:+.3f}", r["action"], f"{r['corr_B_after']:+.3f}"]
        for _, r in harm.iterrows()
    ]
    add_styled_table(doc, ["Feature", "Corr. in A", "Corr. in B (Raw)", "Action on B", "Corr. in B (After)"], harm_rows)

    add_figure(doc, SHOT_DIR / "01_setup_and_data_pipeline_top.png", "Screenshot 1: Assignment 1 Jupyter Notebook — Environment & Data Validation.", 5.8)
    add_figure(doc, SHOT_DIR / "01_setup_and_data_pipeline_results.png", "Screenshot 2: Assignment 1 Jupyter Notebook — Harmonisation & Merge Summary.", 5.8)
    doc.add_page_break()

    # ==================== 3. ASSIGNMENT 2 ====================
    add_heading_1(doc, "3. Assignment 2 — Cyber Security Threat Landscape Analysis (EDA)")
    corr = eda["corr_with_label"]
    doc.add_paragraph(
        f"The merged dataset is well balanced ({pct(n_thr/n)} threats), ensuring accuracy is not distorted by class "
        "imbalance. SSLfinal_State (Spearman r = -0.711, MI = 0.305 nats) and URL_of_Anchor (r = -0.630, MI = 0.259 nats) "
        "are the strongest individual threat indicators, followed by web_traffic, SFH, and Request_URL. Counting "
        "phishy (-1) indicators per website demonstrates cumulative risk: sites with 0 red flags rarely represent threats, "
        "whereas sites with 7 or more red flags are virtually 100% threats."
    )
    add_figure(doc, PLOT_DIR / "01_class_balance_by_source.png", "Figure 1: Class balance (merged) and source composition.", 5.6)
    add_figure(doc, PLOT_DIR / "03_threat_rate_by_feature_code.png", "Figure 2: Threat rate by feature code across all 9 indicators.", 5.8)
    add_figure(doc, PLOT_DIR / "04_red_flag_count.png", "Figure 3: Cumulative risk — threat rate vs. number of phishy (-1) indicators.", 5.2)
    add_figure(doc, PLOT_DIR / "05_correlation_heatmap.png", "Figure 4: Spearman correlation heatmap.", 5.0)
    add_figure(doc, SHOT_DIR / "02_eda_threat_landscape_results.png", "Screenshot 3: Assignment 2 Jupyter Notebook — Executed EDA & Charts.", 5.8)
    doc.add_page_break()

    # ==================== 4. ASSIGNMENT 3 ====================
    add_heading_1(doc, "4. Assignment 3 — Classical Machine Learning Threat Classification")
    doc.add_paragraph(
        "All 9 categorical indicators (-1, 0, 1) were one-hot encoded inside a scikit-learn Pipeline fitted strictly "
        "on the stratified 80/20 training split (9,926 train / 2,482 test, stratified on label x source). Three "
        "classical models were trained: Logistic Regression, Decision Tree (max_depth=6), and Random Forest (200 trees, max_depth=12)."
    )
    ml_rows = []
    for m in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
        r = ml.loc[m]
        ml_rows.append([m, f"{r.accuracy:.4f}", f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.f1:.4f}", f"{r.roc_auc:.4f}", f"{r.pr_auc:.4f}"])
    add_styled_table(doc, ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"], ml_rows)

    add_figure(doc, PLOT_DIR / "07_ml_confusion_matrices.png", "Figure 5: Confusion matrices for Logistic Regression, Decision Tree, and Random Forest.", 6.0)
    add_figure(doc, PLOT_DIR / "08_feature_importance.png", "Figure 6: Relative indicator importance across classical ML models.", 5.5)
    add_figure(doc, SHOT_DIR / "03_ml_classification_results.png", "Screenshot 4: Assignment 3 Jupyter Notebook — Classification Reports & Confusion Matrices.", 5.8)
    doc.add_page_break()

    # ==================== 5. ASSIGNMENT 4 ====================
    add_heading_1(doc, "5. Assignment 4 — PyTorch Deep Neural Network vs. Classical ML")
    doc.add_paragraph(
        "A 3-hidden-layer PyTorch Deep Neural Network (PhishingDNN: 27 -> 128 -> Dropout(0.3) -> 64 -> Dropout(0.2) -> "
        "32 -> 1, trained with BCEWithLogitsLoss, Adam, and validation early stopping) was compared against all four "
        "classical models on the exact same split and across 5 repeated stratified random splits (mean +/- std)."
    )
    cmp_rows = []
    for m in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "PyTorch DNN"]:
        r, q = cmp_.loc[m], rep.loc[m]
        cmp_rows.append([
            m, f"{r.accuracy:.4f}", f"{r.f1:.4f}", f"{r.roc_auc:.4f}", f"{r.pr_auc:.4f}",
            f"{r.train_time_s:.2f}s", f"{q.accuracy_mean:.4f} +/- {q.accuracy_std:.4f}"
        ])
    add_styled_table(doc, ["Model", "Accuracy", "F1", "ROC-AUC", "PR-AUC", "Train Time", "5-Split Acc. (mean +/- std)"], cmp_rows)

    add_figure(doc, PLOT_DIR / "09_model_comparison.png", "Figure 7: Detection performance comparison (Accuracy, Precision, Recall, F1, ROC-AUC).", 5.8)
    add_figure(doc, PLOT_DIR / "13_roc_curves.png", "Figure 8: ROC curves comparing Classical ML and Neural Network (MLP).", 5.0)
    add_figure(doc, PLOT_DIR / "10_training_time.png", "Figure 9: Training and prediction latency comparison.", 5.5)
    add_figure(doc, PLOT_DIR / "11_mlp_learning_curve.png", "Figure 10: MLP training loss and validation error curve.", 4.8)
    add_figure(doc, PLOT_DIR / "12_cross_source_check.png", "Figure 11: Cross-source transfer validation (naive vs. harmonised merge).", 5.4)
    add_figure(doc, SHOT_DIR / "04_dl_vs_ml_comparison_results.png", "Screenshot 5: Assignment 4 Jupyter Notebook — ROC Curves & Learning Curve.", 5.8)

    # ==================== 6. TASK 5: ETHICAL REFLECTION ====================
    thr_df = pd.read_csv(OUT_DIR / "ethical_threshold_tradeoff.csv")
    bias_df = pd.read_csv(OUT_DIR / "ethical_bias_simulation.csv")

    add_heading_1(doc, "6. Task 5 — Ethical Reflection & AI Challenges")
    doc.add_paragraph(
        "Two operational simulations were run on the Neural Network (MLP) to evaluate ethical risks, bias, "
        "privacy, and explainability in cybersecurity deployment:"
    )
    add_heading_2(doc, "6.1 Threshold Tuning: Blocking Legitimate Users vs. Missing Threats")
    thr_rows = [
        [f"{r.threshold:.2f}", f"{r.accuracy:.3f}", f"{r.precision:.3f}", f"{r.recall:.3f}",
         f"{int(r.false_positives)}", f"{int(r.false_negatives)}", f"{r.fpr_legit_blocked*100:.1f}%"]
        for _, r in thr_df[thr_df["threshold"].isin([0.1, 0.3, 0.5, 0.7, 0.9])].iterrows()
    ]
    add_styled_table(doc, ["Threshold", "Accuracy", "Precision", "Recall", "FP (Legit Blocked)", "FN (Missed)", "FPR (%)"], thr_rows)
    add_figure(doc, PLOT_DIR / "14_threshold_tradeoff.png", "Figure 12: Ethical trade-off between blocking legitimate users (FPR) and missing threats (FNR).", 5.6)

    add_heading_2(doc, "6.2 Biased Training Data Simulation")
    bias_rows = [
        [r.scenario, r.eval_source, f"{r.accuracy:.3f}", f"{r.recall:.3f}", f"{r.fpr_legit_blocked*100:.1f}%"]
        for _, r in bias_df.iterrows()
    ]
    add_styled_table(doc, ["Training Scenario", "Evaluated On", "Accuracy", "Recall", "Legit Blocked (FPR)"], bias_rows)
    add_figure(doc, PLOT_DIR / "15_bias_simulation.png", "Figure 13: Accuracy disparity and unfair blocking rate under biased training data.", 5.6)
    add_figure(doc, SHOT_DIR / "05_ethical_reflection_results.png", "Screenshot 6: Task 5 Jupyter Notebook — Ethical Simulation & Analysis.", 5.8)

    doc.add_paragraph(
        "Ethical, Privacy & Explainability Takeaways: (1) Aggressive thresholds (0.10) block 21.7% of legitimate sites; "
        "(2) Omitting Dataset B from training drops its accuracy from 89.3% to 76.0% and nearly triples its false-positive "
        "blocking rate (12.7% to 34.5%); (3) Inspecting live URLs/traffic requires anonymisation to protect user privacy; "
        "(4) Black-box MLP decisions require post-hoc explainability (SHAP/LIME) and human-in-the-loop appeal."
    )
    doc.add_page_break()

    # ==================== 7. CONCLUSION & REFERENCES ====================
    add_heading_1(doc, "7. Conclusion, Limitations & References")
    doc.add_paragraph(
        "Across 12,408 harmonised websites, SSLfinal_State and URL_of_Anchor are the dominant phishing indicators. "
        "While Logistic Regression and Decision Tree provide fast, interpretable ~90% baselines, Random Forest achieves "
        "the highest 5-split mean accuracy (92.42%) and ROC-AUC (0.9785), and the MLP Neural Network achieves the "
        "highest single-split recall (90.71%) and unseen-profile generalisation (86.89%), both approaching the 93.68% "
        "theoretical information ceiling of the 9 shared features."
    )
    doc.add_paragraph(
        "Limitations: (1) Polarity harmonisation in Dataset B is an empirical inference based on correlation alignment; "
        "(2) 9 pre-engineered categorical indicators yield 1,508 unique profiles, bounding maximum achievable accuracy; "
        "(3) raw URLs and live page content are not included."
    )
    add_heading_2(doc, "References")
    refs = [
        "[1] Mohammad, R. & McCluskey, L. Phishing Websites [Dataset]. UCI ML Repository. DOI: 10.24432/C51W2X.",
        "[2] Mohammad, R., Thabtah, F. & McCluskey, L. (2012). An assessment of features related to phishing websites. ICITST.",
        "[3] Abdelhamid, N. Website Phishing [Dataset]. UCI ML Repository. DOI: 10.24432/C5B301.",
        "[4] Abdelhamid, N., Ayesh, A. & Thabtah, F. (2014). Phishing detection based Associative Classification. Expert Systems with Applications.",
        "[5] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.",
    ]
    for r in refs:
        doc.add_paragraph(r, style="List Bullet")

    out_path = REPORT_DIR / "Capstone_Project_Report_Yatharth_Chopra.docx"
    try:
        doc.save(str(out_path))
    except PermissionError:
        out_path = REPORT_DIR / "Capstone_Project_Report_Yatharth_Chopra_Final.docx"
        doc.save(str(out_path))
    print("Saved", out_path)


if __name__ == "__main__":
    main()
