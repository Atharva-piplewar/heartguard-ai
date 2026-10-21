# Heart Disease Dataset Documentation

## Dataset Provenance and Source
- **Official Source**: UCI Machine Learning Repository
- **Dataset**: Heart Disease Dataset (Cleveland Database)
- **URL**: [UCI Heart Disease Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Donor**: Cleveland Clinic Foundation (Robert Detrano, M.D., Ph.D.)
- **Format**: Originally `processed.cleveland.data` consisting of 303 instances and 14 clinical attributes.

## Attribute Information (14 Clinical Features)

| Feature | Name in Dataset | Description | Type | Valid Range / Categories |
|---|---|---|---|---|
| 1 | `age` | Age in years | Numerical | 29 - 77 |
| 2 | `sex` | Biological sex | Binary | 1 = Male, 0 = Female |
| 3 | `cp` | Chest pain type | Categorical | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic |
| 4 | `trestbps` | Resting blood pressure (mm Hg on hospital admission) | Numerical | 94 - 200 mm Hg |
| 5 | `chol` | Serum cholesterol in mg/dl | Numerical | 126 - 564 mg/dl |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | Binary | 1 = True (> 120 mg/dl), 0 = False |
| 7 | `restecg` | Resting electrocardiographic results | Categorical | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy |
| 8 | `thalach` | Maximum heart rate achieved | Numerical | 71 - 202 bpm |
| 9 | `exang` | Exercise induced angina | Binary | 1 = Yes, 0 = No |
| 10 | `oldpeak` | ST depression induced by exercise relative to rest | Numerical | 0.0 - 6.2 |
| 11 | `slope` | Slope of peak exercise ST segment | Categorical | 1: Upsloping, 2: Flat, 3: Downsloping |
| 12 | `ca` | Number of major vessels (0-3) colored by fluoroscopy | Numerical/Discrete | 0, 1, 2, 3 (contains 4 missing values marked as `?` in raw data) |
| 13 | `thal` | Thalassemia cardiac scan status | Categorical | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect (contains 2 missing values marked as `?` in raw data) |
| 14 | `target` (`num`) | Angiographic heart disease status | Target (Integer) | Original: 0 = Absence, 1-4 = Severity of narrowing; Binarized: 0 = No Disease, 1 = Disease Present |

## Target Binarization Rationale
In the clinical study, values 1 through 4 represent different grades of coronary artery stenosis (>50% diameter narrowing). Consistent with the machine learning literature (Detrano et al., 1989; UCI guidelines), classes 1, 2, 3, and 4 are aggregated into positive class `1` (Presence of Heart Disease), while class 0 remains `0` (Absence of Heart Disease).

## Missing Values Strategy
In the raw Cleveland data:
- `ca` has 4 missing entries (originally `?`).
- `thal` has 2 missing entries (originally `?`).
Missing values are handled systematically using median/mode imputation within the scikit-learn preprocessing pipeline, fitted strictly on training data to prevent data leakage.
