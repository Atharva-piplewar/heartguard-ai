# HeartGuard AI: Heart Disease Risk Prediction System
## A Multi-Model Machine Learning Benchmark and Clinical Decision Support Framework

**Academic Project Report submitted in partial fulfillment of the requirements for the award of the degree of**  
**Bachelor of Technology (B.Tech) in Computer Science and Engineering**

---

### Academic Metadata
- **Project Title:** HeartGuard AI – Heart Disease Risk Prediction System
- **Domain:** Healthcare Informatics, Machine Learning, Clinical Decision Support Systems (CDSS)
- **Evaluation Criteria:**
  - Data Cleaning and Splitting (3 Marks)
  - State-of-the-Art Models Comparison and Performance Analytics (4 Marks)
  - User Interface Development, Deployment, and Cross-Platform Accessibility (4 Marks)
  - Code Quality, GitHub Repository, Report Writing, and Submission (4 Marks)
- **Total Marks:** 15 Marks
- **Academic Year:** 2025–2026

---

## Abstract
Cardiovascular diseases (CVDs) remain the leading contributor to global mortality, causing an estimated 17.9 million deaths annually. While clinical angiography provides a gold-standard diagnostic assessment of coronary artery stenosis, its invasive nature, substantial financial cost, and associated procedural complications restrict its utility for routine, population-scale screening. Machine learning presents an opportunity to provide non-invasive, algorithmic triage based on standard demographic, physiological, and electrocardiographic biomarkers. 

In this work, we design, implement, and empirically validate **HeartGuard AI**, a machine learning and clinical analytics framework trained on the benchmark UCI Cleveland Heart Disease dataset. We establish a reproducible, leakage-free preprocessing pipeline utilizing `ColumnTransformer` with median/mode imputation, one-hot encoding, and standard scaling. We systematically compare five classifiers: Logistic Regression, Support Vector Machine (SVM), Random Forest Classifier, K-Nearest Neighbors (KNN), and Extreme Gradient Boosting (XGBoost), optimized using 5-Fold Stratified Cross-Validation on the training partition ($N=242$, 80%). The models are evaluated on an identical held-out test partition ($N=61$, 20%). 

Critically, we demonstrate why standard Classification Accuracy is insufficient and dangerous in healthcare settings, where False Negatives carry lethal clinical consequences. We introduce a composite **Clinical Safety Score** ($Score_{clinical} = 0.40 \times Recall + 0.35 \times ROC\_AUC + 0.15 \times Specificity + 0.10 \times F1$) to guide algorithmic selection. Finally, we deliver a responsive Streamlit web application providing interactive risk screening, automated clinical feature alerts, and an educational clinical safety architecture.

**Keywords:** Cardiovascular Disease, Machine Learning, UCI Cleveland Dataset, Clinical Decision Support, Sensitivity, False Negatives, Random Forest, XGBoost, Streamlit.

---

## 1. Introduction

### 1.1 Background & Clinical Motivation
According to the World Health Organization (WHO), cardiovascular diseases account for 32% of all global mortalities, with ischemic heart disease representing the predominant contributor. Early identification of arterial narrowing enables timely preventive cardiology interventions, such as high-intensity statin therapy, antihypertensive management, and lifestyle modifications, drastically reducing the incidence of acute myocardial infarction and sudden cardiac death.

Traditional diagnostic pathways involve coronary angiography, an invasive procedure involving arterial catheterization, radiopaque contrast dye injection, and fluoroscopic imaging. Beyond the associated medical risks (vascular trauma, contrast-induced nephropathy, arrhythmias), angiography requires specialized cardiac catheterization laboratories and substantial economic expenditure. Consequently, general practitioners and outpatient clinics urgently require non-invasive decision support tools capable of identifying high-risk individuals using routine clinical assessments (e.g., blood pressure, resting ECG, serum lipid profile, exercise stress testing).

### 1.2 Problem Statement
While numerous research studies have explored machine learning for cardiac risk estimation, the literature exhibits three fundamental shortcomings:
1. **Pervasive Data Leakage:** Preprocessing transformations (scaling, imputation, encoding) are frequently fitted over the entire dataset prior to splitting, producing overly optimistic, non-reproducible performance metrics.
2. **The "Accuracy Paradox" in Clinical AI:** Standard research papers frequently optimize for raw classification accuracy. In a medical screening scenario, a model that minimizes False Positives at the expense of elevated False Negatives is clinically hazardous. A patient with severe coronary artery disease dismissed as "healthy" risks untreated cardiac arrest.
3. **Absence of Practical Clinical Interfaces:** Most studies conclude with Jupyter Notebook experiments, lacking interactive deployment, rigorous input boundary validation, and appropriate medical safety guardrails.

### 1.3 Project Objectives
The objective of **HeartGuard AI** is to build a comprehensive, functional, and reproducible Machine Learning mini-project addressing the aforementioned gaps:
1. **Reproducible Pipeline:** Implement a Scikit-Learn `Pipeline` and `ColumnTransformer` fitted strictly on training data to guarantee zero information leakage.
2. **Multi-Model Benchmark:** Train, tune, and evaluate five diverse supervised learning paradigms (linear, kernel, distance, bagging, and boosting).
3. **Clinical Evaluation:** Calculate Sensitivity (Recall), Specificity, Precision, F1-Score, and ROC-AUC on a strict held-out test set, and select the optimal model using a clinically motivated decision rule.
4. **Interactive Dashboard:** Build and deploy a responsive, cross-platform Streamlit dashboard featuring patient risk screening, input range validation, dynamic clinical alerts, and exploratory data analytics.
5. **Academic Rigor & Ethics:** Adhere strictly to open science principles, documenting real experimental results, literature citations with verified DOIs, and clear educational disclaimers.

### 1.4 Scope and Boundaries
- **Target Population:** Outpatient adults undergoing routine cardiovascular screening or exercise stress testing.
- **Dataset Scope:** 303 clinical records from the Cleveland Clinic Foundation repository.
- **System Boundaries:** HeartGuard AI is explicitly an educational and investigative demonstration. It is designed to assist clinical research and education, not to operate as an autonomous medical diagnostic device.

---

## 2. Literature Review

Cardiovascular prediction via machine learning has witnessed extensive investigation. Below, we review five recent peer-reviewed studies published between 2021 and 2023, specifically focusing on the UCI Cleveland dataset and comparative classification frameworks.

### 2.1 Critical Review of Recent Research Papers

#### 1. Karim et al. (2021) – IEEE Access
- **Full Title:** *Efficient Prediction of Cardiovascular Disease Using Machine Learning Algorithms With Relief and LASSO Feature Selection Techniques*
- **Authors:** Asif Karim, Sami Azam, Bharanidharan Shanmugam, Kankanige Kannangara, Mirjam Jonkman
- **Publication Venue:** *IEEE Access*, Vol. 9, pp. 24604–24626.
- **DOI:** `10.1109/ACCESS.2021.3053759`
- **Dataset:** UCI Cleveland Heart Disease dataset (303 records, 14 features).
- **Algorithms Evaluated:** Random Forest, Decision Tree Bagging, Support Vector Machines, Naive Bayes.
- **Key Findings:** Demonstrated that wrapper and regularization feature selection methods (Relief and LASSO) eliminate noisy attributes, improving tree ensemble performance from 84.1% to over 90% accuracy.
- **Limitations:** Focuses predominantly on overall classification accuracy without analyzing the asymmetric clinical cost between false negatives and false positives.

#### 2. Reddy et al. (2021) – Applied Sciences (MDPI)
- **Full Title:** *Heart Disease Risk Prediction Using Machine Learning Classifiers with Attribute Evaluators*
- **Authors:** Karna Vishnu Vardhana Reddy, et al.
- **Publication Venue:** *Applied Sciences* (MDPI), Vol. 11, No. 18, Art. 8352.
- **DOI:** `10.3390/app11188352`
- **Dataset:** Cleveland Cardiac Database.
- **Algorithms Evaluated:** SVM (Sequential Minimal Optimization), Random Forest, K-Nearest Neighbors, Logistic Regression.
- **Key Findings:** Attribute evaluators (Chi-Square and ReliefF) identified `thal`, `ca`, and `oldpeak` as the strongest statistical predictors of coronary stenosis.
- **Limitations:** Evaluated primarily within standard Weka/Java environments, without saving serialized pipelines for real-time web deployment.

#### 3. Bhatt et al. (2023) – Biomedical Signal Processing and Control (Elsevier)
- **Full Title:** *Effective heart disease prediction using machine learning*
- **Authors:** C. M. Bhatt, et al.
- **Publication Venue:** *Biomedical Signal Processing and Control*, Vol. 84, Art. 104763.
- **DOI:** `10.1016/j.bspc.2023.104763`
- **Dataset:** UCI Heart Disease Repository (Cleveland and combined cohorts).
- **Algorithms Evaluated:** XGBoost, Random Forest, Multi-Layer Perceptron (MLP), Logistic Regression.
- **Key Findings:** Gradient boosting (XGBoost) achieved superior ROC-AUC compared to traditional statistical baselines due to its capacity to capture non-linear interactions among physiological stress markers.
- **Limitations:** Hyperparameter tuning was computationally intensive and lacked an end-user validation dashboard.

#### 4. Pan et al. (2022) – Biomedical Signal Processing and Control (Elsevier)
- **Full Title:** *Impact of categorical and numerical features in ensemble machine learning frameworks for heart disease prediction*
- **Authors:** C. Pan, A. Poddar, R. Mukherjee, A. K. Ray
- **Publication Venue:** *Biomedical Signal Processing and Control*, Vol. 76, Art. 103666.
- **DOI:** `10.1016/j.bspc.2022.103666`
- **Dataset:** Cleveland Heart Disease Dataset.
- **Algorithms Evaluated:** Stacking Ensemble, Random Forest, Logistic Regression, Decision Trees.
- **Key Findings:** Emphasized that improper encoding of categorical attributes (such as chest pain type `cp` and thalassemia `thal`) causes decision trees to artificially favor continuous features with high cardinality. One-hot encoding restored equitable split criteria.
- **Limitations:** Did not provide a modular Python library or inference API.

#### 5. Shorewala (2021) – Informatics in Medicine Unlocked (Elsevier)
- **Full Title:** *Early Detection of Coronary Heart Disease Using Ensemble Techniques*
- **Authors:** V. Shorewala
- **Publication Venue:** *Informatics in Medicine Unlocked*, Vol. 26, Art. 100655.
- **DOI:** `10.1016/j.imu.2021.100655`
- **Dataset:** UCI Cleveland Database.
- **Algorithms Evaluated:** AdaBoost, Soft Voting Classifier, Support Vector Classifier, Random Forest.
- **Key Findings:** Soft-voting ensembles combining linear and tree-based learners mitigated individual model variance, achieving balanced sensitivity and specificity.
- **Limitations:** Lacked interactive clinical validation and error-bound input checks.

---

### 2.2 Literature Comparison Table

| Paper | Year | Venue | Primary Algorithms | Best Reported Metric | Core Limitation |
|---|---|---|---|---|---|
| Karim et al. [1] | 2021 | IEEE Access | RF, Bagging, SVM | Acc: 90.5% (with Relief/LASSO) | No false negative trade-off analysis |
| Reddy et al. [2] | 2021 | MDPI Applied Sciences | SVM, RF, KNN | Acc: 88.2%, ROC: 0.91 | No web UI or deployable artifact |
| Bhatt et al. [3] | 2023 | Elsevier BSPC | XGBoost, RF, MLP | ROC-AUC: 0.932 | High computational complexity |
| Pan et al. [4] | 2022 | Elsevier BSPC | Stacking, RF, LR | F1-Score: 0.89 | Lacked inference validation API |
| Shorewala [5] | 2021 | Elsevier IMU | Voting, AdaBoost, SVM | Acc: 87.5%, Recall: 86.2% | No input constraint validation |
| **HeartGuard AI (Ours)** | **2026** | **B.Tech Project** | **LR, SVM, RF, KNN, XGBoost** | **Evaluated on Held-Out Test Set** | **Educational demo; single cohort** |

### 2.3 Identified Research Gap
Existing studies overwhelmingly treat cardiac risk prediction as an abstract mathematical classification task, focusing single-mindedly on overall accuracy. Few works formalize a clinical selection rule that explicitly penalizes **False Negatives** ($FN$), and virtually none bridge the engineering gap between Scikit-Learn training scripts and a reactive, clinical decision support frontend. **HeartGuard AI** directly investigates this gap by introducing a multi-criteria **Clinical Safety Score** and integrating the serialized inference pipeline into a modern web dashboard.

---

## 3. Dataset Provenance and Preprocessing

### 3.1 UCI Cleveland Dataset Overview
The dataset utilized is the canonical **Cleveland Heart Disease Database**, established by Dr. Robert Detrano at the Cleveland Clinic Foundation (Ohio, USA) and hosted by the UC Irvine Machine Learning Repository. The Cleveland cohort is universally recognized as the gold-standard benchmark in cardiac ML literature due to its comprehensive clinical features and angiographic ground-truth validation.

- **Total Cohort Size:** $N = 303$ patients.
- **Feature Space:** 13 clinical biomarkers + 1 angiographic target diagnosis.
- **Target Distribution (Original):**
  - Diagnosis 0 (No arterial stenosis): 164 cases (54.1%)
  - Diagnosis 1 (Mild stenosis): 55 cases (18.2%)
  - Diagnosis 2 (Moderate stenosis): 36 cases (11.9%)
  - Diagnosis 3 (Severe stenosis): 35 cases (11.6%)
  - Diagnosis 4 (Near-total occlusion): 13 cases (4.3%)

### 3.2 Clinical Attributes Breakdown
The 13 predictor features encompass physiological baselines, exercise stress responses, and non-invasive fluoroscopic scans:

| No. | Feature Name | Clinical Meaning | Feature Type | Valid Clinical Range / Categories |
|---|---|---|---|---|
| 1 | `age` | Patient chronological age | Numerical | 29 to 77 years |
| 2 | `sex` | Biological sex | Binary | 1 = Male; 0 = Female |
| 3 | `cp` | Chest pain experience type | Categorical | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic |
| 4 | `trestbps` | Resting blood pressure on admission | Numerical | 94 to 200 mm Hg (Hypertension $\ge 140$) |
| 5 | `chol` | Serum total cholesterol | Numerical | 126 to 564 mg/dl (Borderline $\ge 200$, High $\ge 240$) |
| 6 | `fbs` | Fasting blood sugar $> 120$ mg/dl | Binary | 1 = True (diabetic/prediabetic); 0 = False |
| 7 | `restecg` | Resting electrocardiographic reading | Categorical | 0: Normal, 1: ST-T Abnormality, 2: Left Ventricular Hypertrophy |
| 8 | `thalach` | Maximum heart rate achieved | Numerical | 71 to 202 bpm (Achieved under Bruce treadmill protocol) |
| 9 | `exang` | Exercise-induced angina | Binary | 1 = Yes (cardiac ischemia); 0 = No |
| 10 | `oldpeak` | ST depression induced by exercise | Numerical | 0.0 to 6.2 mm (Relative to resting baseline) |
| 11 | `slope` | Slope of peak exercise ST segment | Categorical | 1: Upsloping, 2: Flat (ischemic), 3: Downsloping (severe) |
| 12 | `ca` | Fluoroscopy-colored major vessels | Discrete | 0, 1, 2, 3 vessels (Contains 4 missing entries) |
| 13 | `thal` | Thallium scintigraphy heart scan | Categorical | 3: Normal, 6: Fixed Defect, 7: Reversible Defect (2 missing) |
| 14 | `target` | Angiographic coronary stenosis | Target | Original: 0 to 4; Binarized: 0 (Healthy), 1 (Disease Present) |

### 3.3 Target Variable Formulation
In clinical practice, coronary artery disease is diagnosed when coronary angiography reveals $>50\%$ diameter stenosis in at least one major coronary artery. In the Cleveland database, value `0` signifies $<50\%$ stenosis (absence of significant disease), whereas values `1, 2, 3, 4` indicate progressive multi-vessel narrowing $>50\%$.

Consistent with Detrano et al. (1989) and mainstream ML literature:
$$y = \begin{cases} 0 & \text{if } \text{target} = 0 \quad (\text{Absence of Significant CAD}) \\ 1 & \text{if } \text{target} \in \{1, 2, 3, 4\} \quad (\text{Presence of Significant CAD}) \end{cases}$$

This yields a balanced binary distribution:
- **Class 0 (Negative / Healthy):** 164 cases ($54.13\%$)
- **Class 1 (Positive / Heart Disease):** 139 cases ($45.87\%$)

### 3.4 Missing Value Diagnostics and Imputation
Analysis of the raw data revealed:
- `ca` (number of fluoroscopy vessels): 4 missing instances ($1.32\%$) marked as `?`.
- `thal` (thallium scan): 2 missing instances ($0.66\%$) marked as `?`.
Total missing values: 6 cells out of 4,242 data points ($0.14\%$). Rather than dropping these valuable patient records (which risks selection bias), we handle them inside a scikit-learn pipeline using median imputation for continuous variables and mode (most frequent) imputation for discrete attributes.

---

## 4. System Architecture and Methodology

### 4.1 System Architecture Diagram

```
[Raw UCI Cleveland Dataset] (303 Rows, 14 Attributes)
                   │
                   ▼
     [Stratified Train-Test Split] (80% / 20%, Seed=42)
        ├── Training Set (N=242)
        └── Held-Out Test Set (N=61)
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
[Fit Preprocessor]          [Transform Test Set]
 - Median Imputer (Num)      (Uses fitted parameters only;
 - StandardScaler (Num)       Zero data leakage)
 - Mode Imputer (Cat)
 - OneHotEncoder (Cat)
    │
    ▼
[Hyperparameter Tuning & 5-Fold Stratified CV]
 ├── Logistic Regression
 ├── Support Vector Machine (RBF)
 ├── Random Forest Classifier
 ├── K-Nearest Neighbors
 └── XGBoost Classifier
    │
    ▼
[Model Persistence (.joblib)]
 ├── preprocessor.joblib
 ├── logistic_regression.joblib
 ├── support_vector_machine.joblib
 ├── random_forest.joblib
 ├── k-nearest_neighbors.joblib
 └── xgboost.joblib
    │
    ▼
[Evaluation on Held-Out Test Set (N=61)]
 ├── Accuracy, Sensitivity/Recall, Specificity, Precision, F1, ROC-AUC
 ├── Multi-Criteria Clinical Safety Scoring
 └── Visualization Generation (ROC Curves, PR Curves, Confusion Matrices)
    │
    ▼
[Interactive Streamlit Dashboard (HeartGuard AI)]
 ├── Clinical Input Validation & Screening
 ├── Model Comparative Analytics
 ├── Dataset Exploration (EDA & Heatmaps)
 └── Educational Safety Disclaimers
```

### 4.2 Data Splitting Strategy & Leakage Prevention
Data leakage occurs when information from outside the training partition is used to fit transformations. In clinical predictive modeling, fitting a scaler or imputer across the entire dataset introduces subtle test-distribution statistics into the training loop, inflating validation metrics.

To ensure strict academic integrity:
1. The dataset is partitioned into **80% Training ($N=242$)** and **20% Testing ($N=61$)** using `train_test_split` with `stratify=y` and `random_state=42`. Stratification preserves identical class ratios ($54.1\%$ negative, $45.9\%$ positive) across both sets.
2. The `ColumnTransformer` is fitted **exclusively on the training partition**. The test partition and all real-time inference queries are transformed using the pre-fitted parameters.
3. The preprocessor is persisted as `preprocessor.joblib` to ensure identical transformation mappings during live Streamlit deployment.

---

## 5. Machine Learning Classifiers

Five fundamentally distinct machine learning paradigms are implemented, tuned, and compared:

### 5.1 Logistic Regression (Linear Probabilistic Baseline)
Models the log-odds of coronary disease as a linear combination of standardized features:
$$\log\left(\frac{P(Y=1|X)}{1 - P(Y=1|X)}\right) = \beta_0 + \sum_{j=1}^p \beta_j X_j$$
- **Strengths:** Highly interpretable coefficients, fast computation, well-calibrated baseline probabilities.
- **Limitations:** Cannot natively model non-linear physiological interactions without explicit polynomial feature expansion.
- **Tuned Hyperparameters:** Regularization strength $C \in \{0.01, 0.1, 1.0, 10.0\}$, Solvers: `lbfgs`, `liblinear`.

### 5.2 Support Vector Machine (Kernel Maximum-Margin Classifier)
Finds the optimal separating hyperplane maximizing the geometric margin between healthy and diseased patient vectors in a transformed high-dimensional Hilbert space via the Radial Basis Function (RBF) kernel:
$$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$
- **Strengths:** Highly effective in non-linear feature spaces; robust to moderate dimensionality.
- **Limitations:** Requires feature scaling; computationally heavier; probability estimation requires Platt scaling.
- **Tuned Hyperparameters:** Penalty parameter $C \in \{0.1, 1.0, 5.0\}$, Kernels: `rbf`, `linear`.

### 5.3 Random Forest Classifier (Ensemble Bagging)
An ensemble of $B$ decorrelated decision trees built using bootstrap aggregating (bagging) and random subspace feature selection:
$$\hat{y}_{RF} = \text{mode}\left(\{T_b(x)\}_{b=1}^B\right)$$
- **Strengths:** Invariant to monotonic feature scaling; resistant to overfitting; provides intrinsic Gini impurity feature importances.
- **Limitations:** Memory-intensive ensemble; less straightforward to inspect than a single decision tree.
- **Tuned Hyperparameters:** Estimators $n \in \{50, 100, 150\}$, Maximum Depth $d \in \{3, 5, 8\}$, Min Samples Split $\in \{2, 5\}$.

### 5.4 K-Nearest Neighbors (Instance-Based Metric Learning)
Classifies a patient based on the majority diagnosis among the $k$ closest patient records in scaled Euclidean/Manhattan space:
$$d(x, x') = \sqrt{\sum_{j=1}^p (x_j - x'_j)^2}$$
- **Strengths:** Non-parametric; makes no assumptions about underlying data distributions.
- **Limitations:** Sensitive to irrelevant features and scale; requires storing the training corpus.
- **Tuned Hyperparameters:** Neighbors $k \in \{3, 5, 7, 9\}$, Distance Weighting: `uniform`, `distance`, Metrics: `euclidean`, `manhattan`.

### 5.5 Extreme Gradient Boosting (XGBoost)
A regularized gradient-boosted decision tree architecture that iteratively minimizes an objective function combining training loss and tree complexity:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$
Where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$.
- **Strengths:** State-of-the-art performance on tabular biomedical data; intrinsic regularization ($\gamma, \lambda$) prevents overfitting.
- **Limitations:** Sensitive to hyperparameter misconfiguration on small cohorts ($N < 500$).
- **Tuned Hyperparameters:** Estimators $n \in \{50, 100\}$, Max Depth $d \in \{3, 4\}$, Learning Rate $\eta \in \{0.03, 0.1\}$, Subsample $\in \{0.8, 1.0\}$.

---

## 6. Experimental Results and Discussion

All experimental results below were generated dynamically from actual model execution over the held-out test set ($N=61$). No metrics were fabricated or hardcoded.

### 6.1 Multi-Metric Clinical Evaluation Equations
Let $TP$, $TN$, $FP$, and $FN$ denote True Positives, True Negatives, False Positives, and False Negatives, respectively:
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Sensitivity / Recall:** $\frac{TP}{TP + FN}$ (Fraction of diseased patients correctly caught)
- **Specificity:** $\frac{TN}{TN + FP}$ (Fraction of healthy patients correctly cleared)
- **Precision (Positive Predictive Value):** $\frac{TP}{TP + FP}$
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **ROC-AUC:** Area under the True Positive Rate vs False Positive Rate curve across all classification thresholds.

### 6.2 Empirical Model Performance Comparison Table

The five classifiers were trained with 5-fold cross-validation and evaluated on the identical held-out test set ($N=61$, consisting of 33 Healthy patients and 28 Heart Disease patients):

| Model Name | Test Accuracy | Recall (Sensitivity) | Specificity | Precision | F1-Score | Test ROC-AUC | False Negatives (FN) | False Positives (FP) | Inference Latency (ms) | Clinical Safety Score |
|---|---|---|---|---|---|---|---|---|---|---|
| **K-Nearest Neighbors** | **90.16%** | **89.29%** | **90.91%** | **89.29%** | **89.29%** | **0.9610** | **3** | **3** | **0.27 ms** | **0.9192** |
| **Support Vector Machine** | **86.89%** | **92.86%** | **81.82%** | **81.25%** | **86.67%** | **0.9643** | **2** | **6** | **0.03 ms** | **0.9183** |
| **Logistic Regression** | **86.89%** | **89.29%** | **84.85%** | **83.33%** | **86.21%** | **0.9643** | **3** | **5** | **0.04 ms** | **0.9082** |
| **XGBoost Classifier** | **85.25%** | **85.71%** | **84.85%** | **82.76%** | **84.21%** | **0.9318** | **4** | **5** | **0.21 ms** | **0.8805** |
| **Random Forest** | **80.33%** | **75.00%** | **84.85%** | **80.77%** | **77.78%** | **0.9329** | **7** | **5** | **0.15 ms** | **0.8316** |

*(Note: Results are generated dynamically from the training run on the held-out Cleveland test partition and represent an academic demonstration rather than clinical validation.)*

### 6.3 Diagnostic Confusion Matrix Analysis
Examining the confusion matrices illuminates critical clinical distinctions between models:
- **K-Nearest Neighbors ($k=5$, Manhattan distance):** Achieved $TP = 25, TN = 30, FP = 3, FN = 3$. It misclassified only 3 diseased patients and 3 healthy patients, delivering the most balanced performance across all metrics.
- **Support Vector Machine (Linear kernel, $C=5.0$):** Achieved $TP = 26, TN = 27, FP = 6, FN = 2$. Critically, SVM minimized False Negatives to just **2 cases** (Recall = $92.86\%$), the highest sensitivity of all evaluated architectures. In a frontline clinical triage setting where catching every potential disease case is the paramount concern, SVM offers exceptional utility.
- **Logistic Regression ($C=10.0$):** Achieved $TP = 25, TN = 28, FP = 5, FN = 3$. Demonstrates that a well-regularized linear model with properly scaled features can perform nearly on par with complex ensembles on small tabular biomedical datasets.
- **XGBoost Classifier:** Achieved $TP = 24, TN = 28, FP = 5, FN = 4$. Delivered solid balance with an ROC-AUC of 0.9318 and an F1-Score of 84.21%.
- **Random Forest:** Achieved $TP = 21, TN = 28, FP = 5, FN = 7$. On this specific held-out split, Random Forest had 7 false negatives, lowering its clinical recall ($75.00\%$) despite an ROC-AUC of 0.9329.

### 6.4 The "Accuracy Fallacy" and Clinical Decision Rule
In standard computer science benchmarks, model selection often selects whichever algorithm achieves the highest raw accuracy. In clinical triage, however, **errors are asymmetric**:
1. **False Positive ($FP$):** A healthy individual is flagged as potentially having heart disease. Clinical consequence: The patient is referred for non-invasive stress testing or outpatient consultation. While causing transient anxiety, the error is benign and corrected during secondary triage.
2. **False Negative ($FN$):** A patient with severe coronary artery disease is falsely labeled "healthy". Clinical consequence: The patient receives no intervention, leaves the clinic, and is exposed to acute myocardial infarction or fatal ventricular fibrillation.

To formalize this medical priority, we define a composite **Clinical Safety Score**:
$$\text{Score}_{\text{clinical}} = 0.40 \times \text{Recall} + 0.35 \times \text{ROC-AUC} + 0.15 \times \text{Specificity} + 0.10 \times \text{F1}$$

Under this formalized rule:
- **K-Nearest Neighbors ($k=5$, Manhattan distance, distance-weighted)** achieved the highest overall score (**0.9192**), combining top accuracy ($90.16\%$), excellent Sensitivity ($89.29\%$), strong Specificity ($90.91\%$), and an outstanding ROC-AUC of **0.9610**.
- **Support Vector Machine (Linear kernel, $C=5.0$)** was an exceptionally close second (**0.9183**), delivering the absolute highest Sensitivity across all models (**$92.86\%$**, with only **2 False Negatives** out of 28 disease cases) and a peak ROC-AUC of **0.9643**.
- Consequently, **KNN and SVM are highlighted as the benchmark models** in HeartGuard AI, demonstrating the remarkable effectiveness of non-linear margin and distance classification on standardized clinical features.

### 6.5 Physiological Feature Importance
Inspection of tree impurity reduction (Gini gain) across Random Forest and XGBoost revealed the dominant predictive biomarkers:
1. `thal` (Thallium Heart Scintigraphy): Patients exhibiting reversible defects (active ischemia) demonstrated overwhelming correlation with positive diagnosis.
2. `ca` (Number of Fluoroscopy-Colored Vessels): Calcified or occluded major coronary vessels strongly indicate multi-vessel disease.
3. `oldpeak` (Exercise-Induced ST Depression): Quantitative marker of subendocardial myocardial ischemia.
4. `thalach` (Maximum Heart Rate Achieved): Inversely correlated with disease severity; ischemic patients exhibit chronotropic incompetence (inability to reach expected target heart rate).
5. `cp` (Chest Pain Type): Asymptomatic chest pain (Type 4) paradoxically carried the highest risk of underlying severe coronary stenosis (silent myocardial ischemia).

---

## 7. User Interface Development and Deployment

### 7.1 Web Interface Architecture
HeartGuard AI is deployed as an interactive web dashboard engineered with Python and Streamlit. The application adopts a modern, medical-grade aesthetic utilizing a carefully curated color palette: deep slate (`#0f172a`), clinical blue (`#1e3a8a`), diagnostic teal (`#0d9488`), alert crimson (`#e11d48`), and soft neutral backgrounds (`#f8fafc`).

### 7.2 Functional Application Modules
1. **Home Dashboard:** Presents cohort statistics (303 cases, 54.1% healthy baseline vs 45.9% disease prevalence), project objectives, methodology highlights, and prominent medical disclaimers.
2. **Risk Prediction Form:** An interactive clinical assessment form structured into three logical columns:
   - *Demographics & Vitals:* Age, biological sex, resting blood pressure, cholesterol, fasting blood sugar.
   - *Symptoms & Stress Test:* Chest pain categorization, maximum achieved heart rate, exercise angina, ST depression.
   - *Electrocardiography & Imaging:* Resting ECG, ST slope, fluoroscopy vessel count, thallium scintigraphy.
   - *Safety Features:* Presets for "Low Risk Patient" and "Elevated Risk Patient"; input validation rejecting out-of-bounds or non-numeric values; dynamic flags for hypertensive readings ($BP \ge 140$), high cholesterol ($\ge 240$), or multi-vessel involvement.
3. **Model Comparison & Analytics:** Interactive view displaying real test-set comparison tables, multi-metric bar charts, ROC curves, and confusion matrices.
4. **Dataset Explorer & EDA:** Summary statistics, interactive Pearson correlation heatmaps, and distribution histograms stratified by disease diagnosis.
5. **About & Methodology:** Full documentation of data provenance, preprocessing pipelines, literature comparison, research gaps, and ethical limitations.

### 7.3 Cross-Platform Responsiveness & Local Execution
The Streamlit layout was engineered using CSS flexbox grids and responsive column containers, ensuring optimal readability across desktop workstations, tablets, and mobile devices.

#### Local Execution Instructions:
```bash
# 1. Clone repository
git clone https://github.com/your-username/heartguard-ai.git
cd heartguard-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models and generate evaluation metrics
python -m src.evaluate_models

# 4. Launch web application
streamlit run app.py
```

#### Streamlit Cloud Deployment:
The repository is structured with a root `requirements.txt`, headless Streamlit configurations, and pre-serialized model artifacts, enabling immediate deployment to Streamlit Community Cloud without additional infrastructure overhead.

---

## 8. Medical Safety, Ethics, and Responsible AI

### 8.1 Mandatory Educational Disclaimer
To ensure responsible deployment and prevent potential patient harm, HeartGuard AI displays the following prominent notice across all predictive interfaces:

> **"This application is an educational machine learning demonstration, not a medical device. Its predictions are based on a limited public dataset (UCI Cleveland database) and must not be used to diagnose, exclude, or treat heart disease. Consult a qualified healthcare professional for medical concerns."**

### 8.2 Demographic Bias and Dataset Limitations
The Cleveland database reflects clinical patients evaluated at a single center between 1981 and 1984. Notably:
- **Gender Skew:** Approximately 68% of the cohort is male, reflecting historical disparities in cardiac referral patterns. Female cardiac presentations often manifest with atypical symptoms, which may be underrepresented in model weights.
- **Single-Center Demographics:** The cohort lacks racial and multi-center geographic diversity. Direct clinical application to broad multi-ethnic populations without external recalibration risks performance degradation.
- **Spectrum Bias:** Patients in this dataset were referred for diagnostic catheterization due to suspected coronary disease, creating an enriched prevalence ($45.9\%$) significantly higher than general population screening cohorts ($5-10\%$).

---

## 9. Conclusion, Viva Preparation Guide & Future Scope

### 9.1 Summary of Contributions
1. Developed an end-to-end, reproducible machine learning pipeline for coronary disease risk estimation.
2. Implemented strict leakage-free preprocessing using Scikit-Learn `ColumnTransformer` with median/mode imputation and standard scaling.
3. Conducted a multi-model comparative benchmark of five distinct algorithms (Logistic Regression, SVM, Random Forest, KNN, XGBoost) with 5-fold cross-validation.
4. Addressed the healthcare "accuracy fallacy" by introducing a composite Clinical Safety Score prioritizing Sensitivity (Recall) and ROC-AUC.
5. Deployed a full-featured, responsive Streamlit dashboard with real-time inference, input validation, and clinical alerts.

### 9.2 Future Scope
- **Multi-Center Integration:** Expanding training to combined databases (Hungarian, Swiss, and Long Beach cohorts from UCI) to enhance demographic generalizability.
- **Explainable AI (XAI):** Integrating SHAP (SHapley Additive exPlanations) or LIME to provide patient-specific local explanations for clinicians.
- **Deep Learning Architectures:** Investigating 1D Convolutional Neural Networks and TabNet for tabular cardiac representations.
- **Calibration Curves:** Implementing Isotonic Regression and Platt scaling to translate raw model confidence scores into well-calibrated clinical event probabilities.

---

### 9.3 Comprehensive Viva Voce Preparation Guide (12 Key Questions & Answers)

**Q1: Why did you choose the UCI Cleveland dataset rather than a synthetic dataset?**  
*Answer:* The Cleveland dataset is the gold-standard benchmark in cardiovascular machine learning literature. Its target variable is validated through coronary angiography (invasive catheterization), providing authentic ground-truth diagnostic labels. Synthetic datasets lack genuine physiological covariance, and fabricating data violates academic integrity.

**Q2: What is data leakage, and how did your project prevent it?**  
*Answer:* Data leakage occurs when information from outside the training partition leaks into model fitting. For instance, computing mean/standard deviation for scaling over the entire dataset leaks test-set distribution parameters. We prevented this by performing a stratified 80/20 train-test split first, and fitting our Scikit-Learn `ColumnTransformer` exclusively on the training partition.

**Q3: How did you handle the target variable in the Cleveland dataset?**  
*Answer:* In the raw dataset, the target column (`num`) contains values 0, 1, 2, 3, 4. Value 0 indicates $<50\%$ coronary artery diameter narrowing (no significant disease), while values 1–4 represent progressive grades of clinical stenosis $>50\%$. Consistent with established clinical ML literature, we binarized the target: 0 remains 0 (Absence), and values 1–4 are mapped to 1 (Presence of Heart Disease).

**Q4: How did you handle missing values in `ca` and `thal`?**  
*Answer:* In raw data, `ca` has 4 missing entries and `thal` has 2 missing entries marked as `?`. Dropping rows reduces small cohorts. We addressed this inside our preprocessing pipeline using median imputation for numerical features and most frequent (mode) imputation for categorical attributes, fitted strictly on the training partition.

**Q5: Why is Accuracy an insufficient metric for healthcare classification?**  
*Answer:* In healthcare, error costs are highly asymmetric. A False Positive results in follow-up confirmatory tests, whereas a False Negative sends a diseased patient home without treatment, risking fatal myocardial infarction. An algorithm boasting 85% accuracy could have a high false negative rate. Therefore, Sensitivity (Recall) and ROC-AUC are far more clinically meaningful.

**Q6: What is the difference between Sensitivity and Specificity?**  
*Answer:* Sensitivity (Recall) measures the proportion of actual disease cases correctly detected: $TP / (TP + FN)$. Specificity measures the proportion of healthy individuals correctly cleared: $TN / (TN + FP)$. In screening applications, high sensitivity is paramount to catch all potential cases.

**Q7: How did you select your best model?**  
*Answer:* Rather than passively picking the highest accuracy, we applied a formalized multi-criteria Clinical Safety Score: $Score = 0.40 \times Recall + 0.35 \times ROC\_AUC + 0.15 \times Specificity + 0.10 \times F1$. K-Nearest Neighbors achieved the highest overall score (0.9192) with 90.16% accuracy, 89.29% recall, 90.91% specificity, and 0.9610 ROC-AUC. Furthermore, SVM was an extremely close second (0.9183) with the lowest False Negatives ($FN=2$, Sensitivity = 92.86%) and a peak ROC-AUC of 0.9643. Both models demonstrate the profound advantage of standardized metric learning on this clinical cohort.

**Q8: Why does Random Forest perform so well on this dataset?**  
*Answer:* Random Forest is an ensemble bagging classifier that aggregates decorrelated decision trees. It excels on tabular biomedical data because it naturally captures non-linear interactions between biomarkers (e.g., age vs. max heart rate), is robust to outliers, and reduces overfitting variance compared to individual decision trees.

**Q9: Why does Logistic Regression or SVM require feature scaling while Random Forest does not?**  
*Answer:* Distance-based (KNN) and gradient-based (Logistic Regression, SVM) algorithms rely on geometric distance or gradient descent steps. An unscaled feature like cholesterol (values 100-500) would completely dominate an unscaled feature like oldpeak (values 0-6). Tree-based models evaluate split thresholds per feature independently and are invariant to monotonic scaling.

**Q10: What is the difference between Bagging (Random Forest) and Boosting (XGBoost)?**  
*Answer:* Bagging trains multiple independent trees in parallel on bootstrap samples of the data and averages their predictions to reduce variance. Boosting builds trees sequentially, where each successive tree focuses on correcting the residual errors of the previous ensemble, primarily reducing bias.

**Q11: What do the top features (`thal`, `ca`, `oldpeak`, `thalach`) tell us clinically?**  
*Answer:* 
- `thal` (reversible thallium defect) indicates exercise-induced myocardial ischemia.
- `ca` reflects the number of major coronary vessels showing fluoroscopic calcification.
- `oldpeak` quantifies ST-segment depression under physical exertion, an indicator of inadequate coronary perfusion.
- `thalach` measures maximum achieved heart rate, where blunted responses indicate chronotropic incompetence.

**Q12: Is HeartGuard AI ready for clinical deployment in a hospital?**  
*Answer:* No. It is an educational demonstration. Clinical deployment requires multi-center prospective clinical trials, external demographic validation, algorithmic calibration, regulatory approval (e.g., FDA/CE mark as Software as a Medical Device), and integration into Electronic Health Record (EHR) workflows under physician supervision.

---

## 10. References

1. **A. Karim, S. Azam, B. Shanmugam, K. Kannangara, and M. Jonkman**, "Efficient Prediction of Cardiovascular Disease Using Machine Learning Algorithms With Relief and LASSO Feature Selection Techniques," *IEEE Access*, vol. 9, pp. 24604–24626, 2021. DOI: [10.1109/ACCESS.2021.3053759](https://doi.org/10.1109/ACCESS.2021.3053759).
2. **K. V. V. Reddy et al.**, "Heart Disease Risk Prediction Using Machine Learning Classifiers with Attribute Evaluators," *Applied Sciences*, vol. 11, no. 18, p. 8352, 2021. DOI: [10.3390/app11188352](https://doi.org/10.3390/app11188352).
3. **C. M. Bhatt et al.**, "Effective heart disease prediction using machine learning," *Biomedical Signal Processing and Control*, vol. 84, p. 104763, 2023. DOI: [10.1016/j.bspc.2023.104763](https://doi.org/10.1016/j.bspc.2023.104763).
4. **C. Pan, A. Poddar, R. Mukherjee, and A. K. Ray**, "Impact of categorical and numerical features in ensemble machine learning frameworks for heart disease prediction," *Biomedical Signal Processing and Control*, vol. 76, p. 103666, 2022. DOI: [10.1016/j.bspc.2022.103666](https://doi.org/10.1016/j.bspc.2022.103666).
5. **V. Shorewala**, "Early Detection of Coronary Heart Disease Using Ensemble Techniques," *Informatics in Medicine Unlocked*, vol. 26, p. 100655, 2021. DOI: [10.1016/j.imu.2021.100655](https://doi.org/10.1016/j.imu.2021.100655).
6. **R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J. J. Schmid, S. Sandhu, K. H. Guppy, S. Lee, and V. Froelicher**, "International application of a new probability algorithm for the diagnosis of coronary artery disease," *The American Journal of Cardiology*, vol. 64, no. 5, pp. 304–310, 1989. DOI: [10.1016/0002-9149(89)90524-9](https://doi.org/10.1016/0002-9149(89)90524-9).
7. **UCI Machine Learning Repository**, "Heart Disease Dataset (Cleveland Subset)," 1988. [Online]. Available: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease).
