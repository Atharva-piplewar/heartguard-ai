"""
HeartGuard AI – Heart Disease Risk Prediction System
===================================================
A Streamlit web application providing clinical machine learning analytics,
multi-model comparative evaluation, interactive risk screening, and dataset exploration.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="HeartGuard AI – Clinical Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Aesthetic Styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0369a1 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #bae6fd;
        font-size: 1.1rem;
        margin: 0;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    }
    
    .disclaimer-box {
        background: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        color: #9f1239;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .disclaimer-box strong {
        color: #881337;
        font-weight: 700;
    }
    
    .selected-badge {
        background: #ecfdf5;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-badge-low {
        background: #dcfce7;
        color: #166534;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .risk-badge-mod {
        background: #fef9c3;
        color: #854d0e;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .risk-badge-high {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to safely load dataset and metadata
@st.cache_data
def get_cached_dataset():
    data_path = "data/heart_disease_clean.csv"
    if not os.path.exists(data_path):
        from data.download_data import download_dataset
        df = download_dataset(output_dir="data")
    else:
        df = pd.read_csv(data_path)
    return df

@st.cache_data
def get_cached_results():
    res_path = "results/model_comparison.json"
    if os.path.exists(res_path):
        with open(res_path, "r") as f:
            return json.load(f)
    return None

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/822/822143.png", width=64)
st.sidebar.title("HeartGuard AI")
st.sidebar.caption("Clinical Decision Support & ML Benchmark")

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home Dashboard",
        "🩺 Risk Prediction Form",
        "📊 Model Comparison & Analytics",
        "🔍 Dataset Explorer & EDA",
        "📚 About & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project:** B.Tech CS Mini Project  
**Benchmark:** UCI Cleveland Heart Disease  
**Models:** 5 Comparative Classifiers  
**Split:** 80% Train / 20% Test (Stratified)
""")

# ==========================================
# PAGE 1: HOME DASHBOARD
# ==========================================
if menu == "🏠 Home Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>HeartGuard AI</h1>
        <p>A Machine Learning System for Cardiovascular Risk Prediction and Multi-Model Clinical Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # Mandatory Medical Safety Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Mandatory Medical & Educational Disclaimer:</strong><br>
        This application is an educational machine learning demonstration, not a medical device. 
        Its predictions are based on a limited public dataset (UCI Cleveland database) and must 
        not be used to diagnose, exclude, or treat heart disease. Consult a qualified healthcare 
        professional for medical concerns.
    </div>
    """, unsafe_allow_html=True)

    try:
        df = get_cached_dataset()
        results = get_cached_results()
        
        # High-level Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients in Dataset", f"{len(df)} Cases", "Cleveland Cohort")
        with col2:
            st.metric("Clinical Features", f"{df.shape[1] - 1} Biomarkers", "Physiological + ECG")
        with col3:
            pos_ratio = (df['target'].sum() / len(df)) * 100
            st.metric("Disease Prevalence", f"{pos_ratio:.1f}%", f"{df['target'].sum()} Positive")
        with col4:
            healthy_ratio = ((len(df) - df['target'].sum()) / len(df)) * 100
            st.metric("Healthy Baseline", f"{healthy_ratio:.1f}%", f"{len(df) - df['target'].sum()} Negative")

        st.markdown("### 🎯 System Highlights & Academic Workflow")
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("""
            **HeartGuard AI** is engineered as an end-to-end, rigorous academic machine learning system:
            
            1. **Leakage-Free Preprocessing:** Features are preprocessed using Scikit-Learn `ColumnTransformer` with median/mode imputation, one-hot encoding, and standard scaling—strictly fitted on the 80% training set.
            2. **Five Comparative Baselines:** Implements and tunes **Logistic Regression**, **Support Vector Machine (SVM)**, **Random Forest**, **K-Nearest Neighbors (KNN)**, and **XGBoost Classifier**.
            3. **Clinical Metric Prioritization:** Rejects naïve accuracy optimization in favor of **Sensitivity (Recall)** and **ROC-AUC** to penalize lethal False Negatives.
            4. **Reproducible Experimentation:** Adheres to fixed random seeds, stratified cross-validation, and serialized joblib pipelines.
            """)
            
            st.info("👉 Use the sidebar to explore interactive risk prediction, comparative model curves, or dataset distributions.")

        with col_right:
            st.markdown("#### 🔬 Target Class Distribution")
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            counts = df['target'].value_counts()
            labels = ["0: No Heart Disease", "1: Heart Disease Present"]
            ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140, colors=["#0284c7", "#e11d48"], explode=(0.04, 0.04))
            ax.axis("equal")
            st.pyplot(fig)

        # Comparative Snapshot Table
        if results and "models" in results:
            st.markdown("### 🏆 Trained Models Quick Overview")
            res_df = pd.DataFrame(results["models"])
            display_cols = ["model_name", "accuracy", "recall", "specificity", "roc_auc", "f1", "clinical_score"]
            renamed = {
                "model_name": "Model",
                "accuracy": "Accuracy",
                "recall": "Recall (Sensitivity)",
                "specificity": "Specificity",
                "roc_auc": "ROC-AUC",
                "f1": "F1-Score",
                "clinical_score": "Clinical Safety Score"
            }
            st.dataframe(res_df[display_cols].rename(columns=renamed), use_container_width=True)
            best_model = results.get("best_model_name", "Random Forest")
            st.success(f"⭐ **Clinically Recommended Model:** **{best_model}** (Selected based on sensitivity to reduce false negatives).")

    except Exception as e:
        st.warning(f"Note: Dataset or models are currently being initialized. Details: {e}")

# ==========================================
# PAGE 2: RISK PREDICTION FORM
# ==========================================
elif menu == "🩺 Risk Prediction Form":
    st.markdown("## 🩺 Cardiovascular Clinical Risk Screening")
    st.caption("Input patient physiological and diagnostic parameters to evaluate predictive risk using trained models.")

    # Mandatory Medical Safety Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Clinical Notice:</strong> This interactive form provides a mathematical prediction 
        derived from a statistical machine learning model trained on historical data. 
        It does not constitute medical advice or a verified clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

    # Preset Profiles Helper
    col_preset1, col_preset2, col_preset3 = st.columns([1, 1, 2])
    with col_preset1:
        load_low = st.button("🔄 Preset: Low Risk Patient")
    with col_preset2:
        load_high = st.button("⚠️ Preset: Elevated Risk Patient")

    # Default values based on presets
    if load_low:
        default_age = 42
        default_sex = 0  # Female
        default_cp = 2   # Atypical Angina
        default_trestbps = 118
        default_chol = 195
        default_fbs = 0
        default_restecg = 0
        default_thalach = 168
        default_exang = 0
        default_oldpeak = 0.0
        default_slope = 1
        default_ca = 0
        default_thal = 3
    elif load_high:
        default_age = 63
        default_sex = 1  # Male
        default_cp = 4   # Asymptomatic
        default_trestbps = 155
        default_chol = 285
        default_fbs = 1
        default_restecg = 2
        default_thalach = 115
        default_exang = 1
        default_oldpeak = 2.8
        default_slope = 2
        default_ca = 2
        default_thal = 7
    else:
        default_age = 54
        default_sex = 1
        default_cp = 3
        default_trestbps = 130
        default_chol = 240
        default_fbs = 0
        default_restecg = 0
        default_thalach = 150
        default_exang = 0
        default_oldpeak = 1.0
        default_slope = 2
        default_ca = 0
        default_thal = 3

    # Model Selection Dropdown
    from src.predict import get_available_models, get_selected_best_model_name, predict_heart_disease_risk
    
    available_models = get_available_models()
    best_name = get_selected_best_model_name()
    model_options = list(available_models.keys()) if available_models else ["Random Forest"]
    default_idx = model_options.index(best_name) if best_name in model_options else 0

    st.markdown("#### 1. Select Prediction Engine")
    selected_model_name = st.selectbox(
        "Choose Model to Perform Inference:",
        options=model_options,
        index=default_idx,
        help="Select any of the 5 trained models to compare diagnostic predictions."
    )

    st.markdown("#### 2. Patient Clinical Parameters")
    with st.form("clinical_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 Demographic & Vitals")
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=default_age, help="Documented range: 29-77 years")
            sex = st.selectbox("Biological Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1], index=0 if default_sex == 1 else 1)
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=default_trestbps, help="Resting BP on admission. Normal < 120, Hypertensive >= 140")
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=default_chol, help="Desirable < 200, High >= 240")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[(0, "False (<= 120 mg/dl)"), (1, "True (> 120 mg/dl)")], format_func=lambda x: x[1], index=default_fbs)

        with col2:
            st.markdown("##### 🫀 Symptoms & Stress Test")
            cp_options = [
                (1, "1: Typical Angina (Chest Discomfort)"),
                (2, "2: Atypical Angina"),
                (3, "3: Non-Anginal Pain"),
                (4, "4: Asymptomatic (Silent Ischemia Risk)")
            ]
            cp = st.selectbox("Chest Pain Type", options=cp_options, format_func=lambda x: x[1], index=default_cp - 1)
            thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=default_thalach, help="Achieved during Bruce treadmill exercise protocol")
            exang = st.selectbox("Exercise-Induced Angina", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1], index=default_exang)
            oldpeak = st.number_input("ST Depression (oldpeak in mm)", min_value=0.0, max_value=7.0, value=float(default_oldpeak), step=0.1, help="ST depression induced by exercise relative to rest")

        with col3:
            st.markdown("##### 🔬 ECG & Advanced Diagnostics")
            restecg_options = [
                (0, "0: Normal"),
                (1, "1: ST-T Wave Abnormality"),
                (2, "2: Left Ventricular Hypertrophy (Estes)")
            ]
            restecg = st.selectbox("Resting Electrocardiographic Results", options=restecg_options, format_func=lambda x: x[1], index=default_restecg)
            
            slope_options = [
                (1, "1: Upsloping"),
                (2, "2: Flat (Ischemic indicator)"),
                (3, "3: Downsloping (Severe indicator)")
            ]
            slope = st.selectbox("Slope of Peak Exercise ST Segment", options=slope_options, format_func=lambda x: x[1], index=default_slope - 1)
            
            ca = st.slider("Major Vessels Colored by Fluoroscopy (0 - 3)", min_value=0, max_value=3, value=default_ca, help="Number of major coronary vessels showing contrast narrowing")
            
            thal_options = [
                (3, "3: Normal Blood Flow"),
                (6, "6: Fixed Defect (Previous Infarct)"),
                (7, "7: Reversible Defect (Active Ischemia)")
            ]
            thal_idx = 0 if default_thal == 3 else (1 if default_thal == 6 else 2)
            thal = st.selectbox("Thallium Heart Scintigraphy (thal)", options=thal_options, format_func=lambda x: x[1], index=thal_idx)

        submit_btn = st.form_submit_button("🚀 Run Predictive Risk Analysis", use_container_width=True)

    if submit_btn:
        input_payload = {
            "age": age,
            "sex": sex[0],
            "cp": cp[0],
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs[0],
            "restecg": restecg[0],
            "thalach": thalach,
            "exang": exang[0],
            "oldpeak": oldpeak,
            "slope": slope[0],
            "ca": ca,
            "thal": thal[0]
        }
        
        try:
            with st.spinner(f"Evaluating clinical parameters with {selected_model_name}..."):
                res = predict_heart_disease_risk(input_payload, model_name=selected_model_name)
            
            st.markdown("---")
            st.markdown("### 📋 Predictive Assessment Output")
            
            out_col1, out_col2 = st.columns([1, 1])
            
            with out_col1:
                if res["prediction_class"] == 1:
                    st.markdown("""
                    <div class="risk-badge-high">
                        ⚠️ PREDICTED STATUS: ELEVATED HEART DISEASE RISK
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-badge-low">
                        ✅ PREDICTED STATUS: LOW HEART DISEASE RISK
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("")
                st.metric("Model Risk Confidence Score", f"{res['risk_percentage']}%", f"Risk Tier: {res['risk_level']}")
                st.progress(float(res["risk_score"]))
                st.caption("ℹ️ Note: This score is a statistical model confidence output, not an individually calibrated biological probability.")

            with out_col2:
                st.markdown("#### 🔍 Notable Clinical Indicators Flagged")
                if res["clinical_flags"]:
                    for flag in res["clinical_flags"]:
                        st.warning(f"• {flag}")
                else:
                    st.success("• No extreme abnormal vitals detected in input features.")
                
                st.markdown(f"**Model Utilized:** `{res['model_used']}`")

        except Exception as err:
            st.error(f"Prediction failed: {err}")

# ==========================================
# PAGE 3: MODEL COMPARISON & ANALYTICS
# ==========================================
elif menu == "📊 Model Comparison & Analytics":
    st.markdown("## 📊 State-of-the-Art ML Model Comparison & Analytics")
    st.caption("Empirical evaluation of 5 comparative classifiers evaluated on the identical held-out test set (20%, N=61).")

    results = get_cached_results()
    
    if results and "models" in results:
        res_df = pd.DataFrame(results["models"])
        
        # Best model highlight
        best_name = results.get("best_model_name", "Random Forest")
        st.markdown(f"""
        <div style="background: #f0fdf4; border-left: 5px solid #22c55e; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <h4 style="margin: 0; color: #15803d;">🏆 Selected Benchmark Model: <strong>{best_name}</strong></h4>
            <p style="margin: 0.3rem 0 0 0; color: #166534; font-size: 0.95rem;">
                <strong>Clinical Decision Rule:</strong> Evaluated using multi-criteria weighted score: 
                <code>Score = 0.40 × Recall + 0.35 × ROC-AUC + 0.15 × Specificity + 0.10 × F1</code>.<br>
                This ensures the model prioritizes <em>Sensitivity (Recall)</em> to minimize dangerous False Negatives 
                while maintaining strong discrimination (ROC-AUC) and acceptable false alarm rates (Specificity).
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Tabbed Analytics View
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Metrics Comparison Table",
            "📈 Performance Charts & ROC",
            "🔲 Confusion Matrices",
            "💡 Feature Importance"
        ])

        with tab1:
            st.markdown("### Test-Set Performance Metrics (Held-Out N=61)")
            display_df = res_df[[
                "model_name", "accuracy", "recall", "specificity", "precision", "f1", "roc_auc", "fn", "fp", "inference_latency_ms", "clinical_score"
            ]].rename(columns={
                "model_name": "Classifier",
                "accuracy": "Accuracy",
                "recall": "Recall (Sens.)",
                "specificity": "Specificity",
                "precision": "Precision",
                "f1": "F1-Score",
                "roc_auc": "ROC-AUC",
                "fn": "False Neg (FN)",
                "fp": "False Pos (FP)",
                "inference_latency_ms": "Latency (ms)",
                "clinical_score": "Safety Score"
            })
            
            st.dataframe(
                display_df.style.highlight_max(
                    subset=["Accuracy", "Recall (Sens.)", "Specificity", "F1-Score", "ROC-AUC", "Safety Score"],
                    color="#bbf7d0"
                ).highlight_min(
                    subset=["False Neg (FN)"],
                    color="#fed7aa"
                ),
                use_container_width=True
            )
            
            st.markdown("""
            > **Why Accuracy Alone Is Misleading in Healthcare:**  
            > In cardiac diagnosis, a **False Negative** means a patient with severe coronary artery disease is sent home without treatment, risking heart failure or fatal myocardial infarction. Conversely, a **False Positive** leads to follow-up testing (e.g., angiography). Therefore, **Recall (Sensitivity)** and **ROC-AUC** are significantly more important than raw Accuracy.
            """)

        with tab2:
            st.markdown("### Metric Comparison Bar Chart & ROC Curves")
            col_bar, col_roc = st.columns(2)
            
            with col_bar:
                if os.path.exists("figures/model_comparison_bar.png"):
                    st.image("figures/model_comparison_bar.png", caption="Comparative Metrics Across Models", use_container_width=True)
                else:
                    st.info("Generating bar chart...")

            with col_roc:
                if os.path.exists("figures/roc_curves.png"):
                    st.image("figures/roc_curves.png", caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
                else:
                    st.info("Generating ROC curves...")

        with tab3:
            st.markdown("### Diagnostic Confusion Matrices Across All 5 Classifiers")
            if os.path.exists("figures/confusion_matrices.png"):
                st.image("figures/confusion_matrices.png", caption="Confusion Matrices (Note low FN values for top models)", use_container_width=True)
            else:
                st.info("Generating confusion matrices...")

        with tab4:
            st.markdown("### Clinical Feature Importance Analysis")
            st.markdown("Shows the most influential physiological markers driving predictions in tree-based ensembles:")
            if os.path.exists("figures/feature_importance.png"):
                st.image("figures/feature_importance.png", caption="Relative Feature Importance (Gini Impurity / Gain)", use_container_width=True)
            if os.path.exists("figures/pr_curves.png"):
                st.image("figures/pr_curves.png", caption="Precision-Recall Curves", use_container_width=True)

    else:
        st.warning("Model comparison metrics have not been generated yet. Please run the training pipeline.")

# ==========================================
# PAGE 4: DATASET EXPLORER & EDA
# ==========================================
elif menu == "🔍 Dataset Explorer & EDA":
    st.markdown("## 🔍 UCI Cleveland Dataset Explorer & EDA")
    st.caption("Interactive data profiling, summary statistics, and feature correlation analysis.")

    df = get_cached_dataset()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cohort Size", f"{df.shape[0]} Rows")
    col2.metric("Total Attributes", f"{df.shape[1]} Columns")
    col3.metric("Missing Values", f"{df.isnull().sum().sum()} Cleaned/Imputed")

    st.markdown("### 📋 Sample Data Preview (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)

    tab_stats, tab_corr, tab_dist = st.tabs(["📊 Summary Statistics", "🔥 Correlation Heatmap", "📈 Feature Distributions"])
    
    with tab_stats:
        st.markdown("#### Numerical Attributes Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with tab_corr:
        st.markdown("#### Pearson Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6.5))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5)
        plt.title("Inter-Feature Correlation Matrix", fontweight="bold", pad=12)
        st.pyplot(fig)
        st.caption("Key observations: 'target' correlates most positively with 'cp', 'oldpeak', 'ca', and 'thal', and negatively with 'thalach'.")

    with tab_dist:
        st.markdown("#### Feature Distribution vs Target Diagnosis")
        feat_choice = st.selectbox("Select Feature to Visualize:", options=["age", "trestbps", "chol", "thalach", "oldpeak", "cp", "ca"])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x=feat_choice, hue="target", kde=True, palette=["#0284c7", "#e11d48"], ax=ax, bins=20)
        ax.set_title(f"Distribution of {feat_choice} Grouped by Heart Disease Status", fontweight="bold")
        ax.set_xlabel(feat_choice)
        ax.set_ylabel("Patient Count")
        st.pyplot(fig)

# ==========================================
# PAGE 5: ABOUT & METHODOLOGY
# ==========================================
elif menu == "📚 About & Methodology":
    st.markdown("## 📚 Project Methodology, Literature Review & Ethics")
    
    st.markdown("""
    ### 1. Problem Statement & Motivation
    Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide, claiming an estimated 17.9 million lives each year according to the World Health Organization (WHO). Early detection of coronary artery disease enables timely lifestyle and pharmacological interventions. However, clinical diagnostic procedures such as coronary angiography are invasive, costly, and carry procedural risks. Machine learning offers non-invasive decision support to triage patients based on routine, cost-effective biomarkers.

    ### 2. Dataset Provenance
    - **Dataset:** UCI Cleveland Heart Disease Dataset
    - **Host:** UC Irvine Machine Learning Repository (Detrano et al., 1989)
    - **Cohort:** 303 patient records with 14 clinical attributes.
    - **Target Binarization:** In the original study, `target` ranges from 0 (absence) to 1-4 (progressive stages of arterial stenosis). For clinical binary triage, 0 denotes healthy and 1-4 denotes the presence of diagnosed disease.

    ### 3. Leakage-Free Preprocessing Pipeline
    To guarantee strict academic reproducibility and eliminate data leakage:
    - **Stratified Split:** 80% Training ($N=242$) and 20% Testing ($N=61$) preserving target class ratio.
    - **Imputation:** Median for continuous features and mode for categorical features, fitted solely on the training partition.
    - **Encoding & Scaling:** `StandardScaler` applied to continuous variables and `OneHotEncoder(drop='first')` applied to multi-category variables.

    ### 4. Recent Literature Review (2021–2023)
    """)

    literature_data = [
        {
            "Paper Title": "Efficient Prediction of Cardiovascular Disease Using ML with Relief & LASSO",
            "Authors": "Karim et al.",
            "Year": "2021",
            "Venue": "IEEE Access",
            "Algorithms": "RF, Bagging, DT, SVM",
            "Key Findings": "Feature selection via LASSO/Relief isolates critical subset; ensemble trees yield top accuracy."
        },
        {
            "Paper Title": "Heart Disease Risk Prediction Using ML Classifiers with Attribute Evaluators",
            "Authors": "Reddy et al.",
            "Year": "2021",
            "Venue": "Applied Sciences (MDPI)",
            "Algorithms": "SVM, RF, KNN, Naive Bayes",
            "Key Findings": "Attribute evaluation using Chi-squared and ReliefF stabilizes multi-classifier performance."
        },
        {
            "Paper Title": "Effective Heart Disease Prediction Using Machine Learning",
            "Authors": "Bhatt et al.",
            "Year": "2023",
            "Venue": "Biomedical Signal Processing & Control",
            "Algorithms": "XGBoost, RF, MLP",
            "Key Findings": "Gradient boosting outperforms single classifiers when continuous features are standardized."
        },
        {
            "Paper Title": "Impact of Categorical & Numerical Features in Ensemble ML Frameworks",
            "Authors": "Pan et al.",
            "Year": "2022",
            "Venue": "Biomedical Signal Processing & Control",
            "Algorithms": "Ensemble Stacking, RF, LR",
            "Key Findings": "Proper categorical one-hot encoding prevents tree splits from biasing toward high-cardinality features."
        },
        {
            "Paper Title": "Early Detection of Coronary Heart Disease Using Ensemble Techniques",
            "Authors": "Shorewala",
            "Year": "2021",
            "Venue": "Informatics in Medicine Unlocked",
            "Algorithms": "Ensemble Voting, AdaBoost, SVM",
            "Key Findings": "Soft-voting ensembles mitigate individual model variance and improve ROC-AUC on small cohorts."
        }
    ]
    st.table(pd.DataFrame(literature_data))

    st.markdown("""
    ### 5. Identified Research Gap & Contribution
    Many published works focus exclusively on maximizing accuracy without reporting sensitivity (recall) or addressing the lethal risk of false negatives. This project bridges that gap by implementing a **Clinical Safety Score** that rewards sensitivity and ROC-AUC, paired with an accessible, interactive clinical dashboard.

    ### 6. Limitations & Ethical AI Considerations
    - **Cohort Size:** 303 cases is modest; performance may vary when generalized to larger multi-center populations.
    - **Demographic Bias:** Historical datasets may exhibit gender or regional skew.
    - **No Clinical Replacement:** This system is an algorithmic screening aid, not a diagnostic instrument.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "HeartGuard AI • B.Tech Computer Science Mini Project • Designed for Academic Evaluation & Research Demonstration"
    "</div>",
    unsafe_allow_html=True
)
