# HeartGuard AI – Heart Disease Risk Prediction System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Library-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **B.Tech Computer Science & Engineering Mini Project (15 Marks)**  
> **Evaluation Criteria:**  
> - Data Cleaning and Splitting – 3 Marks  
> - State-of-the-Art Models Comparison and Performance Analytics – 4 Marks  
> - User Interface Development, Deployment, and Cross-Platform Accessibility – 4 Marks  
> - Code Quality, GitHub Repository, Report Writing, and Submission – 4 Marks  

---

## ⚠️ Mandatory Educational & Medical Disclaimer

> **"This application is an educational machine learning demonstration, not a medical device. Its predictions are based on a limited public dataset (UCI Cleveland database) and must not be used to diagnose, exclude, or treat heart disease. Consult a qualified healthcare professional for medical concerns."**

---

## 📌 Project Overview
**HeartGuard AI** is an end-to-end clinical machine learning framework and decision support web dashboard. Built on the canonical **UCI Cleveland Heart Disease dataset**, the system compares five diverse supervised learning paradigms, addresses the critical **"Accuracy Fallacy"** in clinical AI by prioritizing **Sensitivity (Recall)** and **ROC-AUC** to penalize lethal False Negatives, and provides an interactive, medical-grade Streamlit web interface.

### Key Highlights
- **100% Reproducible Pipeline:** Leakage-free preprocessing using Scikit-Learn `ColumnTransformer` with median/mode imputation, one-hot encoding, and standard scaling.
- **5 Comparative Classifiers:** Logistic Regression, Support Vector Machine (SVM), Random Forest, K-Nearest Neighbors (KNN), and XGBoost.
- **Rigorous Validation:** 5-Fold Stratified Cross-Validation on the training partition ($N=242$, 80%) with final unbiased evaluation on a held-out test partition ($N=61$, 20%).
- **Clinical Safety Scoring:** Algorithmic selection powered by a weighted composite score penalizing False Negatives ($FN$):
  $$\text{Score}_{\text{clinical}} = 0.40 \times \text{Recall} + 0.35 \times \text{ROC-AUC} + 0.15 \times \text{Specificity} + 0.10 \times \text{F1}$$
- **Responsive Web Dashboard:** 5 interactive pages with clinical parameter input validation, presets, real-time risk gauges, inter-feature correlation heatmaps, and publication-ready figures.

---

## 📊 Empirical Model Performance (Held-Out Test Set N=61)

All metrics below are generated dynamically from training runs on the held-out test partition (33 Healthy, 28 Heart Disease):

| Model Name | Test Accuracy | Recall (Sensitivity) | Specificity | Precision | F1-Score | Test ROC-AUC | False Negatives (FN) | False Positives (FP) | Inference Latency (ms) | Clinical Safety Score |
|---|---|---|---|---|---|---|---|---|---|---|
| **K-Nearest Neighbors** | **90.16%** | **89.29%** | **90.91%** | **89.29%** | **89.29%** | **0.9610** | **3** | **3** | **0.27 ms** | **0.9192 (Selected Best)** |
| **Support Vector Machine** | **86.89%** | **92.86%** | **81.82%** | **81.25%** | **86.67%** | **0.9643** | **2** | **6** | **0.03 ms** | **0.9183** |
| **Logistic Regression** | **86.89%** | **89.29%** | **84.85%** | **83.33%** | **86.21%** | **0.9643** | **3** | **5** | **0.04 ms** | **0.9082** |
| **XGBoost Classifier** | **85.25%** | **85.71%** | **84.85%** | **82.76%** | **84.21%** | **0.9318** | **4** | **5** | **0.21 ms** | **0.8805** |
| **Random Forest** | **80.33%** | **75.00%** | **84.85%** | **80.77%** | **77.78%** | **0.9329** | **7** | **5** | **0.15 ms** | **0.8316** |

> **Clinical Interpretation:** While KNN achieved the highest overall composite score (0.9192) and 90.16% accuracy, Support Vector Machine achieved the lowest False Negatives ($FN=2$, Sensitivity = 92.86%), which is exceptionally valuable in clinical frontline screening where catching disease is paramount.

---

## 📁 Repository Structure

```
heartguard-ai/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation
├── .gitignore                  # Git ignore rules
├── data/
│   ├── README.md               # Dataset provenance and attribute schema
│   ├── download_data.py        # Automated dataset downloader & cleaner
│   ├── heart_disease_raw.csv   # Raw Cleveland snapshot (303 rows, 14 cols)
│   └── heart_disease_clean.csv # Binarized target clean dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Leakage-free ColumnTransformer & validation
│   ├── train_models.py         # 5-fold CV & hyperparameter grid search
│   ├── evaluate_models.py      # Test-set evaluation, curves & figures generator
│   └── predict.py              # Inference engine with clinical feature flags
├── models/
│   ├── README.md
│   ├── preprocessor.joblib     # Fitted Scikit-Learn ColumnTransformer
│   ├── logistic_regression.joblib
│   ├── support_vector_machine.joblib
│   ├── random_forest.joblib
│   ├── k-nearest_neighbors.joblib
│   ├── xgboost.joblib
│   └── dataset_metadata.joblib
├── results/
│   ├── README.md
│   ├── model_comparison.csv    # Exported test metrics table
│   └── model_comparison.json   # Full ROC, PR, and confusion matrix coordinates
├── figures/
│   ├── README.md
│   ├── model_comparison_bar.png# Bar chart comparing all 5 models
│   ├── confusion_matrices.png  # 1x5 confusion matrices subplot
│   ├── roc_curves.png          # ROC curves with AUC scores
│   ├── pr_curves.png           # Precision-Recall curves
│   └── feature_importance.png  # Tree feature importance rankings
├── reports/
│   └── report.md               # Complete 10-page academic B.Tech report
└── tests/
    ├── __init__.py
    └── test_preprocessing.py   # Unit tests for preprocessing & validation
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- Git installed.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/heartguard-ai.git
cd heartguard-ai
```

### 3. Create a Virtual Environment (Recommended)
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Reproducing the ML Pipeline

### Step 1: Download & Prepare the Dataset
Downloads the raw UCI Cleveland dataset, inspects missing values in `ca` and `thal`, binarizes the target, and outputs clean CSVs:
```bash
python data/download_data.py
```

### Step 2: Run Unit Tests
Verifies input validation, schema bounds, imputation, and target binarization:
```bash
python -m pytest -v tests/test_preprocessing.py
```
*(All 6 unit tests pass in under 5 seconds).*

### Step 3: Train & Tune All 5 Models
Executes 5-Fold Stratified Cross-Validation on the 80% training set and persists `.joblib` artifacts into `models/`:
```bash
python -m src.train_models
```

### Step 4: Evaluate Models & Generate Figures
Evaluates models on the held-out test set ($N=61$), calculates clinical metrics, applies the Clinical Safety Decision Rule, and saves figures:
```bash
python -m src.evaluate_models
```

---

## 💻 Launching the Streamlit Web Application

To start the interactive clinical dashboard locally:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### Web Dashboard Pages
1. **🏠 Home Dashboard:** Project overview, cohort metrics (303 patients, 54.1% healthy baseline vs 45.9% disease prevalence), model overview, and mandatory clinical disclaimers.
2. **🩺 Risk Prediction Form:** Clinical form for all 13 patient biomarkers, input validation, "Low Risk" and "Elevated Risk" quick presets, model selector dropdown, dynamic risk progress bar, and clinical alerts.
3. **📊 Model Comparison & Analytics:** Test set performance table, metric comparison bar chart, ROC curves, confusion matrices, and feature importance rankings.
4. **🔍 Dataset Explorer & EDA:** Summary statistics, interactive Pearson correlation heatmap, and distribution histograms.
5. **📚 About & Methodology:** Comprehensive documentation of data provenance, preprocessing pipelines, recent literature review (2021–2023), research gaps, and ethical considerations.

---

## 🌐 Deployment to Streamlit Community Cloud

This repository is ready for one-click deployment:
1. Push your repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and select:
   - **Repository:** `your-username/heartguard-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!** Streamlit Cloud automatically installs `requirements.txt` and serves the app online.

---

## 📚 Academic Literature Review (2021–2023)

1. **Karim et al. (2021)**: *"Efficient Prediction of Cardiovascular Disease Using Machine Learning Algorithms With Relief and LASSO Feature Selection Techniques"*, **IEEE Access**, Vol. 9, pp. 24604–24626. DOI: [10.1109/ACCESS.2021.3053759](https://doi.org/10.1109/ACCESS.2021.3053759).
2. **Reddy et al. (2021)**: *"Heart Disease Risk Prediction Using Machine Learning Classifiers with Attribute Evaluators"*, **Applied Sciences (MDPI)**, Vol. 11, No. 18, p. 8352. DOI: [10.3390/app11188352](https://doi.org/10.3390/app11188352).
3. **Bhatt et al. (2023)**: *"Effective heart disease prediction using machine learning"*, **Biomedical Signal Processing and Control (Elsevier)**, Vol. 84, p. 104763. DOI: [10.1016/j.bspc.2023.104763](https://doi.org/10.1016/j.bspc.2023.104763).
4. **Pan et al. (2022)**: *"Impact of categorical and numerical features in ensemble machine learning frameworks for heart disease prediction"*, **Biomedical Signal Processing and Control (Elsevier)**, Vol. 76, p. 103666. DOI: [10.1016/j.bspc.2022.103666](https://doi.org/10.1016/j.bspc.2022.103666).
5. **Shorewala (2021)**: *"Early Detection of Coronary Heart Disease Using Ensemble Techniques"*, **Informatics in Medicine Unlocked (Elsevier)**, Vol. 26, p. 100655. DOI: [10.1016/j.imu.2021.100655](https://doi.org/10.1016/j.imu.2021.100655).

---

## 📄 Academic Project Report & Viva Guide

- The complete 10-page B.Tech academic report is available in [`reports/report.md`](reports/report.md).
- A 12-question **Viva Voce Preparation Guide** with beginner-friendly model answers is included in Section 9.3 of the report.

### Converting Report to PDF
To export the academic report to PDF:
- **Option A (VS Code / Markdown Preview):** Open `reports/report.md`, preview in Markdown, right-click and select "Print / Export to PDF".
- **Option B (Pandoc):**
  ```bash
  pandoc reports/report.md -o reports/report.pdf --pdf-engine=wkhtmltopdf
  ```

---

## ⚖️ License & Acknowledgments
- **Dataset:** Cleveland Clinic Foundation & UCI Machine Learning Repository (Detrano et al., 1989).
- **License:** Released under the MIT License for educational and academic research demonstration.
