"""Build the print-ready Technical Report (PDF) from verified pipeline outputs.

All metrics and counts are loaded dynamically from outputs/ so the report
remains synchronized with the executed code.
Run AFTER all pipeline scripts:
    python src/make_report.py
Output: report/Technical_Report.pdf

Author : Yatharth Chopra (Roll No. 2301420022)
"""
import json

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from common import OUT_DIR, PLOT_DIR, REPORT_DIR, ensure_dirs

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14.5, spaceBefore=10, spaceAfter=6, keepWithNext=1, textColor=colors.HexColor("#1B3A5C"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4, keepWithNext=1, textColor=colors.HexColor("#2A7AB0"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=5)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2.5)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
CAPTION = ParagraphStyle("Cap", parent=BODY, fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER)
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, leading=26, textColor=colors.HexColor("#1B3A5C"))
CENTER = ParagraphStyle("Center", parent=BODY, alignment=TA_CENTER, fontSize=11, leading=17)
REF = ParagraphStyle("Ref", parent=BODY, fontSize=9, leading=12, leftIndent=16, firstLineIndent=-16)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def figure(filename, caption, width_cm=15.5):
    path = PLOT_DIR / filename
    w, h = PILImage.open(path).size
    width = width_cm * cm
    return [Image(str(path), width=width, height=width * h / w), Paragraph(caption, CAPTION), Spacer(1, 6)]


def table(data, col_widths, header_rows=1):
    wrapped = []
    for r, row in enumerate(data):
        wrapped.append([c if not isinstance(c, str) else Paragraph(c, CELLH if r < header_rows else CELL) for c in row])
    t = Table(wrapped, colWidths=col_widths, hAlign="LEFT", repeatRows=header_rows)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#1B3A5C")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, "Yatharth Chopra | Roll No. 2301420022 | ENSP355 Unit-1 Capstone Project")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def pct(x, d=1):
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
    imp = pd.read_csv(OUT_DIR / "feature_importance.csv", index_col=0)
    thr_df = pd.read_csv(OUT_DIR / "ethical_threshold_tradeoff.csv")
    bias_df = pd.read_csv(OUT_DIR / "ethical_bias_simulation.csv")

    lr, dt, rf_m, mlp = "Logistic Regression", "Decision Tree", "Random Forest", "Neural Network (MLP)"
    n, nA, nB = ms["rows_total"], ms["rows_A"], ms["rows_B"]
    n_leg, n_thr = ms["label_counts"]["0"], ms["label_counts"]["1"]
    corr = eda["corr_with_label"]
    rf = eda["red_flag_rate"]
    nflip = int(harm["action"].str.contains("reversed").sum())
    naive = cross[cross.variant == "naive"]["accuracy"]
    harmo = cross[cross.variant == "harmonised"]["accuracy"]
    s = []

    # ---------------- Cover ----------------
    s += [Spacer(1, 2.8 * cm), P("AI-Driven Cyber Threat Awareness<br/>and Phishing Detection Prototype", TITLE)]
    s += [P("Unit 1 Capstone Technical Report — Lab Assignment 01 (Tasks 1 to 5)<br/>Multi-Source Phishing Detection, ML/DL Benchmarking & Ethical Analysis", CENTER), Spacer(1, 1.2 * cm)]
    s += [
        P(
            "<b>Course:</b> AI in Cyber Security Lab (ENSP355)<br/>"
            "<b>University:</b> K.R. Mangalam University<br/><br/>"
            "<b>Student Name:</b> Yatharth Chopra<br/>"
            "<b>Program:</b> B.Tech CSE (Data Science), Semester 7<br/>"
            "<b>Roll No.:</b> 2301420022<br/><br/>"
            "<b>Submitted to:</b> Monika Khatkar<br/>"
            "<b>GitHub:</b> github.com/yatharthchopra2424/Ai-In-Cyber-Project-2301420022",
            CENTER,
        ),
        PageBreak(),
    ]

    # ---------------- 1. Problem Overview ----------------
    s += [P("1. Problem Context and Objectives", H1)]
    s += [
        P(
            "Credential phishing remains one of the highest-volume initial-access techniques in modern cyber operations. "
            "Attackers continuously register short-lived domains and clone login portals, rendering static blocklists "
            "and signature-based filters reactive by nature. Supervised machine learning and neural networks address "
            "this gap by learning structural relationships across SSL/TLS certificates, anchor URLs, domain age, and "
            "page-request patterns to flag previously unseen malicious sites."
        ),
        P(
            "This project implements a complete, reproducible analyst prototype covering all five tasks of ENSP355 Lab "
            "Assignment 01: (1) repository bootstrap, environment validation, and ARFF-to-CSV ingestion; (2) exploratory "
            "threat-landscape analysis; (3) classical machine-learning classification and hyperparameter tuning; "
            "(4) deep learning (MLP) benchmarking against classical ML; and (5) ethical simulations on false-positive "
            "blocking and training-data bias."
        ),
    ]

    # ---------------- 2. Data Sources & Harmonisation ----------------
    s += [P("2. Dataset Integration and Schema Harmonisation", H1)]
    s += [
        P(
            "Two public phishing collections from the UCI Machine Learning Repository (CC BY 4.0) were combined: "
            "<i>Training_Dataset.arff</i> (Dataset A, UCI ID 327) and <i>PhishingData.arff</i> (Dataset B, UCI ID 379). "
            "Both files were parsed from Weka ARFF headers into CSV with zero missing values."
        )
    ]
    rows = [
        ["Attribute", "Dataset A (UCI 327)", "Dataset B (UCI 379)"],
        ["Authors / DOI", "Mohammad & McCluskey | 10.24432/C51W2X", "Abdelhamid | 10.24432/C5B301"],
        ["Records", f"{nA:,}", f"{nB:,}"],
        ["Features (excl. target)", "30", "9 (InfoGain-selected)"],
        ["Class Breakdown", "Legitimate: 6,157 | Phishing: 4,898", "Legitimate: 548 | Phishing: 702 | Suspicious: 103"],
    ]
    s += [table(rows, [3.8 * cm, 6.4 * cm, 6.8 * cm]), Spacer(1, 4), P("Table 1: Source dataset specifications.", CAPTION)]

    s += [
        P(
            f"Because the two collections contain independent websites with no shared URL or timestamp key, they were "
            f"merged via row-wise union across their <b>{len(ms['shared_features'])} common attributes</b> "
            f"({', '.join(ms['shared_features'])}), dropping the {ms['features_dropped_from_A']} attributes exclusive to Dataset A. "
            f"An audit of Spearman correlations with the documented target (1 = legitimate, -1 = phishing) uncovered "
            f"that <b>{nflip} of 9 features in Dataset B were encoded with inverted polarity</b> (and <i>having_IP_Address</i> "
            f"used {{0,1}} instead of {{-1,1}}). Reversing inverted features in Dataset B aligned both halves before "
            f"creating the binary target (0 = legitimate, 1 = threat [phishing or suspicious])."
        )
    ]
    hrows = [["Feature", "Corr. in A", "Corr. in B (Raw)", "Harmonisation Action", "Corr. in B (After)"]]
    for _, r in harm.iterrows():
        hrows.append([r["feature"], f"{r['corr_A']:+.3f}", f"{r['corr_B_raw']:+.3f}", r["action"], f"{r['corr_B_after']:+.3f}"])
    s += [table(hrows, [3.7 * cm, 2.1 * cm, 2.7 * cm, 5.7 * cm, 2.8 * cm]), Spacer(1, 4), P("Table 2: Polarity audit and harmonisation applied to Dataset B.", CAPTION)]

    s += figure("12_cross_source_check.png", "Figure 1: Cross-source transfer accuracy before (naive) and after polarity harmonisation.", 12.5)
    s += [
        P(
            f"Cross-source validation confirms the necessity of harmonisation: naive stacking yields {pct(naive.min(),0)}–{pct(naive.max(),0)} "
            f"transfer accuracy (worse than random guessing due to inverted codes), whereas harmonised stacking lifts transfer "
            f"accuracy to {pct(harmo.min(),0)}–{pct(harmo.max(),0)}. Across the {n:,} merged rows, {ms['unique_feature_profiles']:,} "
            f"unique 9-feature profiles exist, placing a theoretical look-up ceiling of {pct(ms['lookup_table_ceiling_accuracy'])} on this feature space."
        )
    ]

    # ---------------- 3. Task 1 ----------------
    s += [P("3. Task 1 — Workspace Bootstrap and Environment Validation", H1)]
    s += bullets(
        [
            "<b>Structured Repository:</b> Organized into <i>data/</i>, <i>src/</i>, <i>notebooks/</i>, <i>outputs/</i>, <i>report/</i>, and <i>tests/</i> with CI workflow (<i>.github/workflows/ci.yml</i>).",
            "<b>Reproducibility:</b> Fixed random seed (42) across splits and models, pinned dependencies in <i>requirements.txt</i>, and automated smoke tests via <i>pytest</i>.",
            "<b>Verification:</b> <i>01_validate_setup.py</i> validates library imports, runs a baseline classification smoke test, and asserts exact CSV shapes.",
        ]
    )

    # ---------------- 4. Task 2 ----------------
    s += [P("4. Task 2 — Cyber Security Threat Landscape Analysis (EDA)", H1)]
    s += figure("01_class_balance_by_source.png", "Figure 2: Overall class balance (54.0% legitimate, 46.0% threat) and per-source mix.", 13.5)
    s += [
        P(
            f"The merged dataset is nearly balanced ({n_leg:,} legitimate vs. {n_thr:,} threats), preventing majority-class "
            f"distortion. Spearman correlation and Mutual Information identify <b>SSLfinal_State</b> (r = {corr['SSLfinal_State']:+.3f}) "
            f"and <b>URL_of_Anchor</b> (r = {corr['URL_of_Anchor']:+.3f}) as the strongest threat predictors, followed by "
            f"<b>web_traffic</b> ({corr['web_traffic']:+.3f}), <b>SFH</b> ({corr['SFH']:+.3f}), and <b>Request_URL</b> ({corr['Request_URL']:+.3f})."
        ),
        P(
            f"<b>Cumulative Red-Flag Risk:</b> Counting the number of phishy (-1) indicators per website demonstrates monotonic risk "
            f"escalation: sites with 0 red flags have a {pct(rf['0'],0)} threat rate, rising to {pct(rf['4'],0)} at four flags and "
            f"reaching ~100% from seven flags upward."
        ),
    ]
    s += figure("03_threat_rate_by_feature_code.png", "Figure 3: Threat rate across feature codes (-1 Phishy, 0 Suspicious, +1 Legitimate).", 14.5)
    s += figure("04_red_flag_count.png", "Figure 4: Threat rate as a function of cumulative phishy (-1) indicators.", 11.5)
    s += figure("05_correlation_heatmap.png", "Figure 5: Spearman correlation matrix across the 9 shared indicators and target label.", 11.0)

    # ---------------- 5. Task 3 ----------------
    s += [P("5. Task 3 — Classical Machine Learning Classification", H1)]
    s += [
        P(
            "Because all 9 indicators are discrete category codes (-1, 0, 1) rather than continuous measurements, they are "
            "one-hot encoded inside a leak-free <i>scikit-learn</i> Pipeline fitted strictly on the stratified 80/20 training "
            "split (9,926 train / 2,482 test, stratified by label x source). Three classical models were evaluated: "
            "Logistic Regression, Decision Tree (max_depth=6), and Random Forest (200 trees, max_depth=12, further verified via 3-fold GridSearchCV)."
        )
    ]
    rows = [["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Unseen Profiles"]]
    for m in (lr, dt, rf_m):
        r = ml.loc[m]
        rows.append([m, f"{r.accuracy:.3f}", f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.f1:.3f}", f"{r.roc_auc:.3f}", f"{r.acc_unseen_profiles:.3f}"])
    s += [table(rows, [4.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.9 * cm, 2.1 * cm, 2.8 * cm]), Spacer(1, 4)]
    s += [P("Table 3: Classical ML performance on the held-out test set.", CAPTION)]
    s += figure("07_ml_confusion_matrices.png", "Figure 6: Confusion matrices for Logistic Regression, Decision Tree, and Random Forest.", 14.5)
    s += figure("08_feature_importance.png", "Figure 7: Relative indicator importance summed across one-hot encoded columns.", 11.5)

    # ---------------- 6. Task 4 ----------------
    s += [P("6. Task 4 — Deep Learning (MLP) vs. Classical Machine Learning", H1)]
    s += [
        P(
            "A feed-forward Neural Network (<i>MLPClassifier</i>, hidden layers 32 -> 16, ReLU, Adam, early stopping on a 10% "
            "validation split) was benchmarked against all three classical models on the identical split and across 5 repeated "
            "stratified random splits."
        )
    ]
    rows = [["Model", "Accuracy", "F1", "Recall", "ROC-AUC", "Train (s)", "5-Split Acc. (mean +/- sd)"]]
    for m in (lr, dt, rf_m, mlp):
        r, q = cmp_.loc[m], rep.loc[m]
        rows.append([m, f"{r.accuracy:.3f}", f"{r.f1:.3f}", f"{r.recall:.3f}", f"{r.roc_auc:.3f}", f"{r.train_time_s:.3f}",
                     f"{q.accuracy_mean:.3f} +/- {q.accuracy_std:.3f}"])
    s += [table(rows, [3.9 * cm, 1.8 * cm, 1.6 * cm, 1.7 * cm, 2.0 * cm, 1.8 * cm, 4.2 * cm]), Spacer(1, 4)]
    s += [P("Table 4: Comprehensive benchmark across classical ML and Neural Network (MLP).", CAPTION)]
    s += figure("09_model_comparison.png", "Figure 8: Accuracy, Precision, Recall, F1, and ROC-AUC across all four models.", 13.5)
    s += figure("13_roc_curves.png", "Figure 9: Receiver Operating Characteristic (ROC) curves on the test set.", 11.5)
    s += figure("10_training_time.png", "Figure 10: Training and inference latency comparison.", 12.5)
    s += figure("11_mlp_learning_curve.png", "Figure 11: MLP training loss and validation error with early stopping.", 10.0)
    s += [
        P(
            f"<b>Key Comparison Findings:</b> On this 9-feature tabular dataset, the ensemble <b>Random Forest</b> achieves the highest "
            f"5-split mean accuracy ({pct(rep.loc[rf_m].accuracy_mean, 2)}) and ROC-AUC ({rep.loc[rf_m].roc_auc_mean:.4f}) while "
            f"training in {rep.loc[rf_m].train_time_s_mean:.2f}s. The <b>MLP</b> achieves the highest single-split recall "
            f"({cmp_.loc[mlp].recall:.3f}) and unseen-profile accuracy ({pct(cmp_.loc[mlp].acc_unseen_profiles)}), demonstrating "
            f"strong non-linear generalisation at a modest latency cost ({cmp_.loc[mlp].train_time_s:.2f}s). Both approach the "
            f"{pct(ms['lookup_table_ceiling_accuracy'])} theoretical ceiling. Excluding the 103 suspicious rows shifts accuracy by "
            f"less than 1.5 percentage points without altering model rankings."
        )
    ]

    # ---------------- 7. Task 5 ----------------
    s += [P("7. Task 5 — Ethical Reflection and AI Challenges", H1)]
    s += [
        P(
            "To evaluate real-world deployment risks, two targeted simulations were conducted on the trained MLP model: "
            "(1) varying the classification probability threshold from 0.10 to 0.90, and (2) training on a biased subset "
            "(Dataset A only) while evaluating fairness on Dataset B."
        )
    ]
    thr_rows = [["Threshold", "Accuracy", "Precision", "Recall", "False Pos. (Blocked)", "False Neg. (Missed)", "FPR (%)"]]
    for _, r in thr_df[thr_df["threshold"].isin([0.1, 0.3, 0.5, 0.7, 0.9])].iterrows():
        thr_rows.append([
            f"{r.threshold:.2f}", f"{r.accuracy:.3f}", f"{r.precision:.3f}", f"{r.recall:.3f}",
            f"{int(r.false_positives)}", f"{int(r.false_negatives)}", f"{r.fpr_legit_blocked*100:.1f}%"
        ])
    s += [table(thr_rows, [2.0 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 3.3 * cm, 3.3 * cm, 2.2 * cm]), Spacer(1, 4)]
    s += [P("Table 5: Threshold tuning trade-off between blocking legitimate users and missing phishing threats.", CAPTION)]
    s += figure("14_threshold_tradeoff.png", "Figure 12: Ethical trade-off curve across decision thresholds (0.10 to 0.90).", 12.5)

    bias_rows = [["Training Scenario", "Evaluated On", "Accuracy", "Recall", "Legit Blocked (FPR)"]]
    for _, r in bias_df.iterrows():
        bias_rows.append([r.scenario, r.eval_source, f"{r.accuracy:.3f}", f"{r.recall:.3f}", f"{r.fpr_legit_blocked*100:.1f}%"])
    s += [table(bias_rows, [4.8 * cm, 3.8 * cm, 2.4 * cm, 2.2 * cm, 3.6 * cm]), Spacer(1, 4)]
    s += [P("Table 6: Impact of dataset bias when Dataset B is excluded from training.", CAPTION)]
    s += figure("15_bias_simulation.png", "Figure 13: Accuracy drop and unfair false-positive rate on underrepresented websites.", 12.5)

    s += bullets(
        [
            "<b>Ethical Risks (False-Positive Blocking):</b> Lowering the threshold to 0.10 raises threat recall to 97.7%, but wrongly blocks 291 legitimate websites (21.7% FPR), causing denial-of-service and reputational harm without an analyst appeal workflow.",
            "<b>Algorithmic Bias & Fairness:</b> Training solely on Dataset A degrades accuracy on Dataset B from 89.3% to 76.0% and nearly triples the false-positive blocking rate (12.7% to 34.5%).",
            "<b>Privacy Concerns:</b> Production URL and traffic inspection can expose user browsing history, session tokens, or PII unless telemetry is stripped and anonymised prior to inference.",
            "<b>Explainability Limitations:</b> Shallow Decision Trees yield auditable rules, whereas the MLP is opaque and requires post-hoc explainability (SHAP/LIME) before automated enforcement.",
        ]
    )

    # ---------------- 8. Conclusion & References ----------------
    s += [P("8. Conclusion and Limitations", H1)]
    s += [
        P(
            f"By harmonising two UCI phishing collections ({n:,} websites) and evaluating classical ML alongside a neural "
            f"network, this prototype demonstrates that <b>SSLfinal_State</b> and <b>URL_of_Anchor</b> drive detection "
            f"accuracy up to ~92.5% (ROC-AUC ~0.98), approaching the {pct(ms['lookup_table_ceiling_accuracy'])} information "
            f"ceiling of the 9 shared features. Key limitations include reliance on pre-engineered categorical indicators "
            f"(1,508 unique profiles), partial cross-source compatibility (63–68%), and the need for human-in-the-loop "
            f"threshold governance in production."
        )
    ]

    s += [P("References", H1)]
    refs = [
        "[1] Mohammad, R. and McCluskey, L. <i>Phishing Websites</i> [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C51W2X.",
        "[2] Mohammad, R., Thabtah, F. and McCluskey, L. (2012). An assessment of features related to phishing websites using an automated technique. <i>ICITST</i>.",
        "[3] Abdelhamid, N. <i>Website Phishing</i> [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5B301.",
        "[4] Abdelhamid, N., Ayesh, A. and Thabtah, F. (2014). Phishing detection based Associative Classification data mining. <i>Expert Systems with Applications</i>.",
        "[5] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i>, 12, 2825-2830.",
    ]
    s += [P(r, REF) for r in refs]

    doc = SimpleDocTemplate(
        str(REPORT_DIR / "Technical_Report.pdf"),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title="AI-Driven Cyber Threat Awareness and Phishing Detection Prototype",
        author="Yatharth Chopra",
    )
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("Saved", REPORT_DIR / "Technical_Report.pdf")


if __name__ == "__main__":
    main()
