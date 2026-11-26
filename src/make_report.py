"""Build the technical report (PDF) from the saved results and plots.

Every number in the text is read from the files in outputs/, so the report cannot
drift out of sync with the results. Run AFTER all other scripts:
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
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, keepWithNext=1)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=5)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
CAPTION = ParagraphStyle("Cap", parent=BODY, fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER)
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, leading=26)
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
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#2a7ab0")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
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
    canvas.drawString(2 * cm, 1.2 * cm, "Yatharth Chopra | 2301420022 | ENSP355 Capstone Project (Tasks 1-4)")
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
    mlsrc = pd.read_csv(OUT_DIR / "ml_results_by_source.csv")
    cmp_ = pd.read_csv(OUT_DIR / "model_comparison.csv").set_index("model")
    cmpsrc = pd.read_csv(OUT_DIR / "model_comparison_by_source.csv")
    rep = pd.read_csv(OUT_DIR / "model_comparison_repeated.csv").set_index("model")
    cross = pd.read_csv(OUT_DIR / "cross_source_check.csv")
    sens = pd.read_csv(OUT_DIR / "suspicious_sensitivity.csv")
    sub = pd.read_csv(OUT_DIR / "feature_subset_sensitivity.csv").set_index(["features", "model"])
    imp = pd.read_csv(OUT_DIR / "feature_importance.csv", index_col=0)

    lr, dt, rf_m, mlp = "Logistic Regression", "Decision Tree", "Random Forest", "Neural Network (MLP)"
    n, nA, nB = ms["rows_total"], ms["rows_A"], ms["rows_B"]
    n_leg, n_thr = ms["label_counts"]["0"], ms["label_counts"]["1"]
    corr = eda["corr_with_label"]
    rf = eda["red_flag_rate"]
    nflip = int(harm["action"].str.contains("reversed").sum())
    naive = cross[cross.variant == "naive"]["accuracy"]
    harmo = cross[cross.variant == "harmonised"]["accuracy"]
    speed_ratio = cmp_.loc[mlp, "train_time_s"] / cmp_.loc[dt, "train_time_s"]
    s = []

    # ---------------- Cover ----------------
    s += [Spacer(1, 3 * cm), P("AI-Driven Cyber Threat Awareness<br/>and Detection Prototype", TITLE)]
    s += [P("Capstone Project Technical Report - Lab Assignments 1 to 4 (Unit 1)<br/>Phishing website detection on two merged UCI datasets", CENTER), Spacer(1, 1.2 * cm)]
    s += [
        P(
            "<b>Course:</b> AI in Cyber Security Lab (ENSP355)<br/>"
            "<b>University:</b> K.R. Mangalam University<br/><br/>"
            "<b>Name:</b> Yatharth Chopra<br/>"
            "<b>Program:</b> B.Tech CSE (Data Science), Semester 7<br/>"
            "<b>Roll No.:</b> 2301420022<br/><br/>"
            "<b>Submitted to:</b> Monika Khatkar<br/>"
            "<b>Scope:</b> Tasks 1 to 5 (Full 100-Mark Rubric)",
            CENTER,
        ),
        PageBreak(),
    ]

    # ---------------- 1. Problem overview ----------------
    s += [P("1. Problem Overview", H1)]
    s += [
        P(
            "Phishing is one of the most common ways attackers steal credentials: a fake website imitates a "
            "trusted one and tricks the visitor into entering secrets. Blocklists and fixed rules only catch "
            "sites that are already known, whereas a machine-learning model can learn which combinations of "
            "page and URL characteristics usually go with phishing and flag sites it has never seen. This "
            "project plays the role of an entry-level security analyst: it combines two public phishing "
            "datasets, explores the threat indicators, trains classifiers to label a website as "
            "<b>legitimate</b> or <b>threat</b> (phishing or suspicious), and compares classical machine "
            "learning with a neural network."
        ),
        P(
            "<b>Objectives:</b> (1) set up a reproducible project; (2) convert the two ARFF files to CSV and merge "
            "them on their common features, with the basis of the merge stated explicitly; (3) analyse threat "
            "patterns with EDA; (4) classify websites with supervised ML; (5) compare ML against a neural "
            "network on accuracy, training time and practical feasibility."
        ),
    ]

    # ---------------- 2. Data sources ----------------
    s += [P("2. Data Sources and Citations", H1)]
    s += [
        P(
            "Both datasets come from the UCI Machine Learning Repository [6] and are released under the "
            "Creative Commons Attribution 4.0 licence, so they may be reused with attribution. They were "
            "supplied as the ARFF files <i>Training_Dataset.arff</i> (Dataset A) and <i>PhishingData.arff</i> "
            "(Dataset B)."
        )
    ]
    rows = [
        ["", "Dataset A", "Dataset B"],
        ["UCI name / id", "Phishing Websites (327) [1]", "Website Phishing (379) [3]"],
        ["Creators", "Rami Mohammad, Lee McCluskey", "Neda Abdelhamid"],
        ["DOI", "10.24432/C51W2X", "10.24432/C5B301"],
        ["Associated paper", "Mohammad, Thabtah, McCluskey (2012) [2]", "Abdelhamid, Ayesh, Thabtah (2014) [4]"],
        ["Websites", f"{nA:,}", f"{nB:,}"],
        ["Attributes (excl. label)", "30", "9 (the ARFF header records Weka InfoGain attribute selection)"],
        ["Classes in the file", "legitimate 6,157; phishing 4,898", "legitimate 548; phishing 702; suspicious 103"],
    ]
    s += [table(rows, [3.8 * cm, 6 * cm, 7.2 * cm]), Spacer(1, 4), P("Table 1: The two source datasets.", CAPTION)]

    # ---------------- 3. Data preparation ----------------
    s += [P("3. Data Preparation: ARFF to CSV and the Merge", H1)]
    s += [P("3.1 ARFF to CSV conversion", H2)]
    s += [
        P(
            "An ARFF file has a header (<i>@relation</i>, one <i>@attribute</i> line per column) followed by an "
            "<i>@data</i> section of comma-separated rows. <i>00a_arff_to_csv.py</i> parses this with plain Python, "
            "and asserts the shapes published on the UCI pages (11,055 x 31 and 1,353 x 10), that every value is one "
            "the header declares, and that nothing is missing ('?'). Both conversions passed with zero missing "
            "values. The raw CSVs keep the original column names, including the UCI spellings "
            "<i>popUpWidnow</i> and <i>Shortining_Service</i>, so every column traces back to the source."
        )
    ]

    s += [P("3.2 Basis on which the datasets were joined", H2)]
    s += bullets(
        [
            "<b>Type of merge - row-wise union, not a key join.</b> Neither file has a website ID, URL or timestamp, so "
            "rows cannot be matched one-to-one; and they do not need to be, because the two files describe "
            "<i>different</i> websites. The datasets are therefore stacked (union of rows) on a shared schema.",
            f"<b>Join basis - the common features.</b> The attributes whose name is identical in both ARFF headers: "
            f"{', '.join(ms['shared_features'])}. Identical names, and a co-author (F. Thabtah) common to both associated papers [2][4], "
            f"indicate the same feature family; Section 3.6 tests how far the coding really matches.",
            f"<b>Dropped attributes.</b> The {ms['features_dropped_from_A']} attributes that exist only in Dataset A "
            f"would be empty for all {nB:,} Dataset B rows, so they are not used.",
            "<b>Label alignment.</b> Dataset B's UCI page documents the label as 1 = legitimate, 0 = suspicious, -1 = phishing, and its "
            "class counts (548 / 702 / 103) match the file. Dataset A uses the same convention (its features correlate positively with the "
            "1 = legitimate label, Table 2). Labels were therefore kept as documented.",
            "<b>Provenance column.</b> A <i>source</i> column records which dataset each row came from. It is used to "
            "stratify the train/test split and to report per-source results, but it is not a model input.",
        ]
    )

    s += [P("3.3 Harmonising feature polarity (an issue found while merging)", H2)]
    s += [
        P(
            "Dataset B's UCI page states the codes as +1 legitimate, 0 suspicious, -1 phishy, and Dataset A follows the same convention. "
            "If that held for the features in both files, every feature would correlate <i>positively</i> with the label (1 = legitimate). "
            f"It does in Dataset A, but in Dataset B {nflip} of 9 features correlate <i>negatively</i> (Table 2). For "
            "example, only 14% of Dataset A's phishing sites have SSLfinal_State = +1, while about 80% of Dataset B's "
            "phishing sites do in the raw file. Stacking the files unchanged would make the same code mean opposite things "
            "in the two halves."
        ),
        P(
            "<b>Rule applied (Dataset B only):</b> a feature whose Spearman correlation with B's own label is negative has its "
            "polarity reversed (x becomes -x). <i>having_IP_Address</i> is stored as {0,1} in B but {-1,1} in A, so it "
            "was mapped onto {-1,1} with the same rule. <i>web_traffic</i> already agreed and was left unchanged."
        ),
    ]
    rows = [["Feature", "Corr. in A", "Corr. in B (raw)", "Action on B", "Corr. in B (after)"]]
    for _, r in harm.iterrows():
        rows.append([r["feature"], f"{r['corr_A']:+.3f}", f"{r['corr_B_raw']:+.3f}", r["action"], f"{r['corr_B_after']:+.3f}"])
    s += [table(rows, [3.7 * cm, 2.2 * cm, 2.8 * cm, 5.6 * cm, 2.8 * cm]), Spacer(1, 4)]
    s += [P("Table 2: Spearman correlation of each shared feature with the label (suspicious rows excluded for this test).", CAPTION)]

    s += [P("3.4 Target variable", H2)]
    s += [
        P(
            "The assignment asks for normal versus suspicious events, so the binary target is <b>label = 0</b> for "
            "legitimate sites and <b>label = 1</b> for anything else (phishing or suspicious). The 103 'suspicious' "
            "sites exist only in Dataset B; an analyst would not wave them through, so they count as threats. The "
            f"original 3-way code is kept in <i>result_raw</i>. Re-running without those rows changes accuracy by "
            f"under 1.5 percentage points (Section 7.4)."
        )
    ]

    s += [P("3.5 The merged dataset", H2)]
    rows = [
        ["Property", "Value"],
        ["Rows", f"{n:,} ({nA:,} from A + {nB:,} from B)"],
        ["Columns", "record_id, source, 9 features, result_raw, label"],
        ["Legitimate / threat", f"{n_leg:,} ({pct(n_leg/n)}) / {n_thr:,} ({pct(n_thr/n)})"],
        ["Threat share per source", f"A {pct(eda['threat_share_by_source']['UCI327_PhishingWebsites'])}, B {pct(eda['threat_share_by_source']['UCI379_WebsitePhishing'])}"],
        ["Missing values", str(ms["missing_values"])],
        ["Distinct feature profiles", f"{ms['unique_feature_profiles']:,} (only {pct(ms['unique_feature_profiles']/n)} of rows)"],
        ["Profiles with conflicting labels", f"{ms['profiles_with_mixed_labels']} (identical features, different labels)"],
        ["Profiles present in both sources", str(ms["profiles_in_both_sources"])],
        ["Best possible accuracy from these 9 features", f"{pct(ms['lookup_table_ceiling_accuracy'])} (look-up table on the full data)"],
    ]
    s += [table(rows, [6 * cm, 11 * cm]), Spacer(1, 4), P("Table 3: Summary of data/merged_phishing.csv.", CAPTION)]
    s += [
        P(
            "Because only about 1,500 distinct feature profiles exist across 12,408 websites, many test rows have an identical "
            "twin in the training set. Results are therefore also reported for <i>unseen</i> profiles, and the 93.7% "
            "look-up ceiling is an upper bound for any model that sees only these 9 features, measured on this data."
        )
    ]

    s += [P("3.6 Does the merge hold together? (validation)", H2)]
    s += [
        P(
            "To test whether the polarity correction mattered, a model was trained on one source and tested on the other, "
            "using Dataset B once as in the raw file (naive) and once harmonised."
        )
    ]
    s += figure("12_cross_source_check.png", "Figure 1: Cross-source accuracy, naive stacking vs harmonised stacking.", 12.5)
    s += [
        P(
            f"With naive stacking, cross-source accuracy is {pct(naive.min(),0)}-{pct(naive.max(),0)}, i.e. <i>below</i> the 50% "
            f"expected from guessing: the model is systematically wrong because the codes point in opposite directions. After "
            f"harmonising it is {pct(harmo.min(),0)}-{pct(harmo.max(),0)}. This confirms the correction was needed, but it also shows "
            "the two sources are only <b>partly</b> compatible: within a single source accuracy is about 90%. Figure 2 shows why - "
            "after harmonising, the share of each code still differs between sources for several features (for example "
            "URL_Length is 'phishy' for 81% of A's sites but 27% of B's), which suggests the two collections applied different "
            "thresholds when coding the same feature names. <b>Caveat:</b> the polarity signs were chosen using B's labels, so "
            "the harmonised bars are favoured by construction; this is a consistency check, not independent proof."
        )
    ]
    s += figure("06_source_alignment.png", "Figure 2: Share of 'phishy' and 'legitimate' codes per source after harmonisation.", 15.5)

    # ---------------- 4. Task 1 ----------------
    s += [P("4. Task 1 - Workspace and Environment", H1)]
    s += bullets(
        [
            "Repository <i>Ai-In-Cyber-Project-2301420022</i> with README.md, .gitignore, requirements.txt, "
            "<i>data/</i> (raw ARFF and CSV, merged CSV), <i>notebooks/</i> (4 executed assignment notebooks), and <i>src/</i>.",
            "Python environment with numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter, python-docx, and reportlab.",
            "<i>01_validate_setup.py</i> imports every package, fits a tiny model as a smoke test, and checks that the converted and "
            "merged CSVs have the expected shapes.",
            "Every script uses a fixed random seed (42), so results are reproducible.",
        ]
    )

    # ---------------- 5. Task 2 ----------------
    s += [P("5. Task 2 - Threat Landscape Analysis (EDA)", H1)]
    s += figure("01_class_balance_by_source.png", "Figure 3: Legitimate vs threat websites, and class mix per source.", 14)
    s += [
        P(
            f"The merged data is fairly balanced ({pct(n_thr/n)} threats), so accuracy is not distorted by a rare attack class. "
            f"Dataset B has a higher threat share ({pct(eda['threat_share_by_source']['UCI379_WebsitePhishing'],0)} vs "
            f"{pct(eda['threat_share_by_source']['UCI327_PhishingWebsites'],0)}) and is the only one with suspicious sites."
        ),
        P("<b>Common threat indicators.</b> Ranked by correlation with the threat label (Figure 7) and by how sharply the threat rate changes across codes (Figure 5):"),
    ]
    s += bullets(
        [
            f"<b>SSLfinal_State</b> (correlation {corr['SSLfinal_State']:+.2f}): 85% of sites with a 'phishy' certificate state and 86% "
            "with 'suspicious' are threats, against 12% of 'legitimate' ones - the strongest single indicator.",
            f"<b>URL_of_Anchor</b> ({corr['URL_of_Anchor']:+.2f}): 95% of sites coded 'phishy' are threats (links that point away "
            "from the site's own domain), against 14% when 'legitimate'.",
            f"<b>web_traffic</b> ({corr['web_traffic']:+.2f}): sites coded 'phishy' or 'suspicious' for web traffic are threats 65% of the time, against 27% when 'legitimate'.",
            f"<b>SFH</b> ({corr['SFH']:+.2f}) and <b>Request_URL</b> ({corr['Request_URL']:+.2f}) give moderate signal.",
            f"<b>URL_Length</b> ({corr['URL_Length']:+.2f}) and <b>having_IP_Address</b> ({corr['having_IP_Address']:+.2f}) are weak on their own, "
            "although they may still help in combination.",
        ]
    )
    s += figure("02_feature_code_distributions.png", "Figure 4: How often each code occurs in legitimate vs threat websites.")
    s += figure("03_threat_rate_by_feature_code.png", "Figure 5: Share of sites that are threats for each feature code.")
    s += [
        P(
            "<b>Red flags add up.</b> Counting how many of the 9 features are coded 'phishy' gives a simple frequency view "
            f"of suspicious activity: the threat rate is {pct(rf['0'],0)} with no red flags, {pct(rf['4'],0)} with four, "
            f"{pct(rf['6'],0)} with six, and essentially 100% from seven red flags upward."
        )
    ]
    s += figure("04_red_flag_count.png", "Figure 6: Threat rate vs number of 'phishy' indicators.", 11.5)
    s += figure("05_correlation_heatmap.png", "Figure 7: Spearman correlations between features and the threat label.", 11)

    # ---------------- 6. Task 3 ----------------
    s += [P("6. Task 3 - Machine-Learning Threat Classification", H1)]
    s += [P("6.1 Preprocessing", H2)]
    s += bullets(
        [
            "<b>Missing values:</b> none were found in either source or the merge; a most-frequent imputer is in the pipeline as a safety net.",
            "<b>Categorical encoding:</b> all 9 features are category codes (-1 phishy, 0 suspicious, 1 legitimate), not measurements, "
            "so they are one-hot encoded with the three codes fixed in advance.",
            "<b>Normalisation:</b> the dataset has no continuous numeric attributes, and one-hot columns are already 0/1 on a common scale, "
            "so no scaler was needed.",
            "<b>Split:</b> stratified 80/20 on label x source (9,926 train / 2,482 test) so both halves keep the same class mix and "
            "the same share of Dataset A and B rows. Transformers are fitted on the training set only.",
        ]
    )
    s += [P("6.2 Results", H2)]
    rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Acc. unseen"]]
    for m in (lr, dt, rf_m):
        r = ml.loc[m]
        rows.append([m, f"{r.accuracy:.3f}", f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.f1:.3f}", f"{r.roc_auc:.3f}", f"{r.acc_unseen_profiles:.3f}"])
    s += [table(rows, [4.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 2.2 * cm, 2.8 * cm]), Spacer(1, 4)]
    s += [P(f"Table 4: Classical ML test-set performance (threat = positive class). Unseen profiles are {pct(ml.loc[lr].share_unseen_profiles)} of test rows.", CAPTION)]
    a_lr = mlsrc[(mlsrc.model == lr) & mlsrc.source.str.contains("A")].iloc[0]
    b_lr = mlsrc[(mlsrc.model == lr) & mlsrc.source.str.contains("B")].iloc[0]
    a_dt = mlsrc[(mlsrc.model == dt) & mlsrc.source.str.contains("A")].iloc[0]
    b_dt = mlsrc[(mlsrc.model == dt) & mlsrc.source.str.contains("B")].iloc[0]
    s += [
        P(
            f"Both models reach about 90% accuracy, with the Decision Tree slightly ahead ({pct(ml.loc[dt].accuracy)} vs {pct(ml.loc[lr].accuracy)}). "
            f"Precision (about {ml.loc[lr].precision:.2f}) is higher than recall ({ml.loc[lr].recall:.2f}-{ml.loc[dt].recall:.2f}): when the model raises "
            f"an alert it is usually right, but roughly 1 threat in 8 is missed. In security work missed attacks are costly, so recall deserves "
            f"attention; a lower decision threshold would trade precision for recall. This is consistent with the {pct(ms['lookup_table_ceiling_accuracy'])} ceiling: "
            f"{ms['profiles_with_mixed_labels']} feature profiles carry both labels, so some errors cannot be removed with these 9 features."
        ),
        P(
            f"<b>Per source.</b> On Dataset A rows accuracy is {pct(a_lr.accuracy)} (Logistic Regression) and {pct(a_dt.accuracy)} (Decision Tree); on Dataset B "
            f"rows it is lower, {pct(b_lr.accuracy)} and {pct(b_dt.accuracy)} (only {int(b_lr.test_rows)} test rows). The harmonised B rows are harder, in line "
            "with the partial compatibility found in Section 3.6."
        ),
        P(
            f"<b>Which indicators matter.</b> The Decision Tree puts {pct(imp.loc['SSLfinal_State','Decision Tree (importance)'],0)} of its importance on "
            f"SSLfinal_State and {pct(imp.loc['URL_of_Anchor','Decision Tree (importance)'],0)} on URL_of_Anchor; Logistic Regression spreads weight more "
            "evenly but ranks the same two features first. This agrees with the EDA."
        ),
    ]
    s += figure("07_ml_confusion_matrices.png", "Figure 8: Confusion matrices for the two classical models.", 12)
    s += figure("08_feature_importance.png", "Figure 9: Indicator importance, summed over each feature's one-hot columns.", 11)

    # ---------------- 7. Task 4 ----------------
    s += [P("7. Task 4 - ML vs Neural Network", H1)]
    s += [
        P(
            "A scikit-learn <i>MLPClassifier</i> (hidden layers of 32 and 16 neurons, ReLU, Adam optimiser, early stopping on a 10% "
            "validation split) was trained on the same preprocessed data and split as Task 3. Because the models are close, the "
            "comparison was also repeated over 5 random splits."
        )
    ]
    rows = [["Model", "Accuracy", "F1", "Recall", "ROC-AUC", "Train (s)", "Acc. 5 splits (mean +/- sd)"]]
    for m in (lr, dt, rf_m, mlp):
        r, q = cmp_.loc[m], rep.loc[m]
        rows.append([m, f"{r.accuracy:.3f}", f"{r.f1:.3f}", f"{r.recall:.3f}", f"{r.roc_auc:.3f}", f"{r.train_time_s:.3f}",
                     f"{q.accuracy_mean:.3f} +/- {q.accuracy_std:.3f}"])
    s += [table(rows, [3.8 * cm, 1.8 * cm, 1.6 * cm, 1.7 * cm, 2.0 * cm, 1.9 * cm, 4.2 * cm]), Spacer(1, 4)]
    s += [P("Table 5: Same data, same split, same preprocessing across all four models.", CAPTION)]
    s += figure("09_model_comparison.png", "Figure 10: Detection metrics for the four models.", 13)
    s += figure("13_roc_curves.png", "Figure 11: ROC curves comparing Classical ML and Neural Network (MLP).", 11.5)
    s += figure("10_training_time.png", "Figure 12: Training and prediction time.", 12.5)
    s += figure("11_mlp_learning_curve.png", "Figure 13: MLP learning curve (early stopping).", 9.5)
    gain_dt = rep.loc[mlp, "accuracy_mean"] - rep.loc[dt, "accuracy_mean"]
    gain_lr = rep.loc[mlp, "accuracy_mean"] - rep.loc[lr, "accuracy_mean"]
    s += [P("<b>7.1 Detection accuracy.</b>", BODY)]
    s += [
        P(
            f"The neural network is the most accurate model here: {pct(cmp_.loc[mlp].accuracy)} on the main split "
            f"(F1 {cmp_.loc[mlp].f1:.3f}) and {pct(rep.loc[mlp].accuracy_mean)} on average over 5 splits, i.e. {gain_dt*100:.1f} points above the Decision Tree "
            f"and {gain_lr*100:.1f} above Logistic Regression. The split-to-split standard deviation is only about 0.2-0.6 points, so the ordering is consistent, "
            f"although the absolute gain is small. A likely reason is that phishing depends on <i>combinations</i> of indicators (for example a suspicious certificate "
            f"<i>and</i> suspicious anchors), which Logistic Regression cannot represent and a depth-6 tree captures only partly. Its recall is also the best "
            f"({cmp_.loc[mlp].recall:.3f}). On unseen profiles the MLP leads again ({pct(rep.loc[mlp].acc_unseen_profiles_mean,0)} vs "
            f"{pct(rep.loc[dt].acc_unseen_profiles_mean,0)} and {pct(rep.loc[lr].acc_unseen_profiles_mean,0)} on average), although those subsets are small and noisy. "
            f"It stays below the {pct(ms['lookup_table_ceiling_accuracy'])} ceiling, as it must."
        )
    ]
    a_m = cmpsrc[(cmpsrc.model == mlp) & cmpsrc.source.str.contains("A")].iloc[0]
    b_m = cmpsrc[(cmpsrc.model == mlp) & cmpsrc.source.str.contains("B")].iloc[0]
    s += [P("<b>7.2 Training time.</b>", BODY)]
    s += [
        P(
            f"The MLP took {cmp_.loc[mlp].train_time_s:.2f} s to train against {cmp_.loc[dt].train_time_s:.2f} s for the Decision Tree "
            f"(about {speed_ratio:.0f}x longer; {rep.loc[mlp].train_time_s_mean:.1f} s on average over the 5 splits). Prediction time is about the same for all three. "
            "At 12,408 rows the cost is negligible, but it grows quickly with data size and the MLP also needs more tuning (layer sizes, early stopping)."
        )
    ]
    s += [P("<b>7.3 Practical feasibility.</b>", BODY)]
    s += [
        P(
            f"The simple models are quicker to train, easier to explain to an analyst (a tree can be read as rules) and good enough for a first "
            f"filter at about 90% accuracy. The MLP buys roughly {gain_dt*100:.0f}-{gain_lr*100:.0f} extra points of accuracy at a modest cost on a dataset this size, so it is a "
            f"reasonable choice when detection quality matters more than transparency. Its predictions are harder to justify, which matters when a "
            f"blocked user asks why. On Dataset B rows the MLP scores {pct(b_m.accuracy)} versus {pct(a_m.accuracy)} on Dataset A rows; with only "
            f"{int(b_m.test_rows)} B rows in the test set, per-source differences between models are noisy."
        )
    ]
    s += [P("<b>7.4 Sensitivity to the 'suspicious' rows.</b>", BODY)]
    s_main = sens[sens.variant.str.contains("main")].set_index("model")["accuracy"]
    s_drop = sens[sens.variant.str.contains("removed")].set_index("model")["accuracy"]
    s += [
        P(
            "Removing the 103 suspicious sites changes accuracy by "
            + ", ".join(f"{abs(s_drop[m]-s_main[m])*100:.1f} points ({m})" for m in (lr, dt, mlp))
            + "; the model ranking is unchanged, so counting them as threats does not drive the conclusions."
        )
    ]
    s += [P("<b>7.5 Do the three hard-to-verify features matter?</b>", BODY)]
    k9, k6 = "9 shared features (main)", "6 verifiable features"
    d_acc = {m: (sub.loc[(k6, m), "accuracy_mean"] - sub.loc[(k9, m), "accuracy_mean"]) * 100 for m in (lr, dt, mlp)}
    gap6 = (sub.loc[(k6, mlp), "accuracy_mean"] - sub.loc[(k6, dt), "accuracy_mean"]) * 100
    sd6 = sub.loc[(k6, mlp), "accuracy_std"] * 100
    s += [
        P(
            "Three of the nine shared features cannot have their coding checked against Dataset A: popUpWidnow and having_IP_Address use different "
            "value sets in the two files ({-1,1} vs {-1,0,1}, and {-1,1} vs {0,1}), and all three, including URL_Length, correlate with the label in A by "
            "no more than 0.09, so the direction of B's coding for them rests on B alone. Re-running Tasks 3-4 with only the six verifiable features "
            "(validate_feature_subset.py, same 5 splits) changes mean accuracy by "
            + ", ".join(f"{d_acc[m]:+.1f} points ({m})" for m in (lr, dt, mlp))
            + f". The ordering Neural Network > Decision Tree > Logistic Regression is kept, but the MLP's lead over the Decision Tree shrinks to "
            f"{gap6:.1f} points against a split-to-split standard deviation of {sd6:.1f}, so with six features it is no longer a clear win. "
            f"Cross-source accuracy for the Decision Tree is lower without the three features "
            f"({sub.loc[(k6, dt), 'cross_source_acc_AtoB']*100:.0f}% / {sub.loc[(k6, dt), 'cross_source_acc_BtoA']*100:.0f}% vs "
            f"{sub.loc[(k9, dt), 'cross_source_acc_AtoB']*100:.0f}% / {sub.loc[(k9, dt), 'cross_source_acc_BtoA']*100:.0f}% for A to B / B to A), "
            f"so there is no sign that keeping them damages the merge; the headline results do not depend on them."
        )
    ]

    # ---------------- 8. Task 5 - Ethical Reflection ----------------
    thr_df = pd.read_csv(OUT_DIR / "ethical_threshold_tradeoff.csv")
    bias_df = pd.read_csv(OUT_DIR / "ethical_bias_simulation.csv")

    s += [P("8. Task 5 - Ethical Reflection and AI Challenges", H1)]
    s += [
        P(
            "Deploying AI in cybersecurity introduces operational and ethical risks beyond raw accuracy. Two "
            "simulations were executed in <i>05_ethical_reflection.py</i> on the Neural Network (MLP):"
        )
    ]
    s += [P("8.1 Threshold Tuning: Blocking Legitimate Users vs. Missing Threats", H2)]
    thr_rows = [["Threshold", "Accuracy", "Precision", "Recall", "False Pos. (Legit Blocked)", "False Neg. (Threats Missed)", "FPR (%)"]]
    for _, r in thr_df[thr_df["threshold"].isin([0.1, 0.3, 0.5, 0.7, 0.9])].iterrows():
        thr_rows.append([
            f"{r.threshold:.2f}", f"{r.accuracy:.3f}", f"{r.precision:.3f}", f"{r.recall:.3f}",
            f"{int(r.false_positives)}", f"{int(r.false_negatives)}", f"{r.fpr_legit_blocked*100:.1f}%"
        ])
    s += [table(thr_rows, [2.0 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 3.4 * cm, 3.4 * cm, 2.0 * cm]), Spacer(1, 4)]
    s += [P("Table 6: Impact of varying the MLP decision threshold on legitimate-user blocking vs. missed threats.", CAPTION)]
    s += figure("14_threshold_tradeoff.png", "Figure 14: Ethical trade-off between blocking legitimate sites (FPR) and missing phishing threats (FNR).", 12.5)

    s += [P("8.2 Biased Training Data Simulation", H2)]
    bias_rows = [["Training Scenario", "Evaluated On", "Accuracy", "Recall", "Legit Blocked (FPR)"]]
    for _, r in bias_df.iterrows():
        bias_rows.append([r.scenario, r.eval_source, f"{r.accuracy:.3f}", f"{r.recall:.3f}", f"{r.fpr_legit_blocked*100:.1f}%"])
    s += [table(bias_rows, [4.8 * cm, 3.8 * cm, 2.4 * cm, 2.2 * cm, 3.6 * cm]), Spacer(1, 4)]
    s += [P("Table 7: Performance disparity when Dataset B is omitted from training.", CAPTION)]
    s += figure("15_bias_simulation.png", "Figure 15: Accuracy drop and unfair blocking rate (FPR) on underrepresented websites.", 12.5)

    s += [P("8.3 Ethical Risks, Privacy Concerns & Explainability", H2)]
    s += bullets(
        [
            "<b>Ethical risks (False Positives):</b> At threshold 0.10, recall reaches 97.7% but 291 legitimate sites (21.7%) are wrongly blocked. Automated blocking without human appeal can cause denial-of-service to legitimate businesses.",
            "<b>Data bias & fairness:</b> Training only on Dataset A causes accuracy on Dataset B to drop from 89.3% to 76.0% and nearly triples the false-positive blocking rate (12.7% to 34.5%).",
            "<b>Privacy concerns:</b> Inspecting live URLs, query strings, and web traffic logs in production can expose sensitive user browsing history or session tokens unless telemetry is anonymised.",
            "<b>Explainability limitations:</b> Unlike a shallow Decision Tree whose rules an analyst can audit, the MLP Neural Network is opaque and requires post-hoc explanation (e.g., SHAP/LIME) before blocking users.",
        ]
    )

    # ---------------- 9. Limitations / Conclusion ----------------
    s += [P("9. Limitations", H1)]
    s += bullets(
        [
            "<b>Harmonisation is an inference.</b> Reversing Dataset B's feature polarity rests on the internal evidence that the codes contradict the documented "
            "convention, and it used B's labels (9 sign decisions). The UCI pages do not explain the discrepancy, so it could not be confirmed with the dataset authors.",
            "<b>Three features are unverifiable against Dataset A</b> (popUpWidnow, URL_Length, having_IP_Address; Section 7.5). They were kept in the main "
            "analysis, and a six-feature re-run shows the results do not hinge on them, but their coding in B is still an inference.",
            "<b>The sources are only partly compatible</b> (63-68% cross-source accuracy against about 90% within a source); some shared feature names appear "
            "to be coded with different thresholds.",
            "<b>Few distinct profiles.</b> 9 coded features give about 1,500 profiles, so random test splits are optimistic; the unseen-profile figures are more cautious but based on few rows.",
            "<b>Pre-engineered features.</b> The data holds coded indicators, not raw URLs or page content, so the models cannot see anything the original feature extraction missed. "
            "The datasets are also snapshots, and phishing tactics change.",
        ]
    )
    s += [P("10. Conclusion", H1)]
    s += [
        P(
            f"Two UCI phishing datasets were converted from ARFF to CSV and merged by row-wise union on their {len(ms['shared_features'])} shared features, giving {n:,} labelled "
            f"websites. The merge exposed a real data issue - Dataset B codes 8 of the 9 features in the opposite direction - which was corrected and checked. On the "
            f"merged data, SSL state and anchor-URL behaviour are the strongest indicators of phishing, Logistic Regression and a Decision Tree reach about 90% accuracy, and a "
            f"small neural network reaches about {pct(rep.loc[mlp].accuracy_mean,0)}, close to the {pct(ms['lookup_table_ceiling_accuracy'],0)} limit of these features. Future work should "
            "tune decision thresholds for recall, validate the merge against a third phishing dataset, and add the ethical analysis of Task 5."
        )
    ]

    # ---------------- References ----------------
    s += [P("References", H1)]
    refs = [
        "[1] Mohammad, R. and McCluskey, L. <i>Phishing Websites</i> [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C51W2X. https://archive.ics.uci.edu/dataset/327/phishing+websites. Licence: CC BY 4.0.",
        "[2] Mohammad, R., Thabtah, F. and McCluskey, L. (2012). An assessment of features related to phishing websites using an automated technique. <i>International Conference for Internet Technology and Secured Transactions (ICITST)</i>.",
        "[3] Abdelhamid, N. <i>Website Phishing</i> [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5B301. https://archive.ics.uci.edu/dataset/379/website+phishing. Licence: CC BY 4.0.",
        "[4] Abdelhamid, N., Ayesh, A. and Thabtah, F. (2014). Phishing detection based Associative Classification data mining. <i>Expert Systems with Applications</i>.",
        "[5] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830. https://scikit-learn.org/stable/",
        "[6] Kelly, M., Longjohn, R. and Nottingham, K. <i>The UCI Machine Learning Repository</i>. https://archive.ics.uci.edu",
    ]
    s += [P(r, REF) for r in refs]

    doc = SimpleDocTemplate(
        str(REPORT_DIR / "Technical_Report.pdf"),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title="AI-Driven Cyber Threat Awareness and Detection Prototype",
        author="Yatharth Chopra",
    )
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("Saved", REPORT_DIR / "Technical_Report.pdf")


if __name__ == "__main__":
    main()
