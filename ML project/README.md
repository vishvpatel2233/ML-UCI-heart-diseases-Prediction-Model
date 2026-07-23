# ❤️ Heart Disease Prediction — ML Data Preprocessing & Predictive Modeling

A complete, end-to-end **supervised machine learning pipeline** built on the **UCI Heart Disease
(Cleveland) dataset** — covering exploratory data analysis, data cleaning, feature engineering,
encoding, scaling, feature selection, multi-model training, evaluation, cross-validation, and model
comparison.

This project is designed to be **resume-worthy** and demonstrates practical, professional ML workflow
skills using pure Python + scikit-learn (no black-box AutoML).

---

## 📁 Project Structure

```
heart-disease-ml-project/
├── README.md
├── requirements.txt
├── build_notebook.py                 # Script that programmatically generates the notebook
├── data/
│   └── heart.csv                     # Raw UCI Heart Disease dataset (303 records, 14 columns)
├── notebooks/
│   └── ML_Data_Preprocessing_Predictive_Modeling.ipynb   # Main, fully-executed notebook
└── models/
    ├── best_model_Random_Forest.pkl  # Best-performing trained model (serialized)
    ├── standard_scaler.pkl           # Fitted StandardScaler for inference-time reuse
    └── selected_features.pkl         # List of the top-K selected feature names
```

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Heart Disease Data Set](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland subset)

- **303 patient records**, 13 clinical features + binary target
- **Target:** `1` = heart disease present, `0` = absent
- Features include age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood
  sugar, resting ECG results, max heart rate, exercise-induced angina, ST depression, slope, number
  of major vessels, and thalassemia test result.

> **Note:** To robustly demonstrate a real-world preprocessing pipeline (the source data has no
> organic missing values), a small amount of controlled missingness (~3–5%, MCAR) was intentionally
> injected into a few columns. This is disclosed transparently in the notebook, alongside genuine
> duplicate rows introduced/removed to demonstrate deduplication.

---

## 🔬 Pipeline Overview

| Stage | Techniques Used |
|---|---|
| **EDA** | `.info()`, dtypes, missing-value heatmap, duplicate detection, summary statistics, histograms, box plots, scatter plots, pair plots, correlation heatmap |
| **Cleaning** | Median imputation (numeric), mode imputation (categorical), duplicate removal |
| **Encoding** | `LabelEncoder` (chest pain type), `OneHotEncoder` (thalassemia) |
| **Scaling** | `StandardScaler` (used for modeling), `MinMaxScaler` (comparison) |
| **Feature Selection** | `SelectKBest` with ANOVA F-test (`f_classif`), top-10 features retained |
| **Split** | Stratified 80/20 train-test split |
| **Models** | KNN, Logistic Regression, Gaussian Naive Bayes, SVM (RBF), Decision Tree, Random Forest |
| **Evaluation** | Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC |
| **Validation** | Stratified 5-Fold Cross-Validation |
| **Interpretability** | Feature importance (Decision Tree & Random Forest) |

---

## 🏆 Model Comparison Results

| Model | Train Acc. | Test Acc. | CV Acc. (5-Fold) | Precision | Recall | F1 | Train Time (s) |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 0.9711 | **0.8525** | 0.8415 | 0.8000 | 0.9697 | 0.8767 | 0.327 |
| Logistic Regression | 0.8512 | 0.8361 | 0.8416 | 0.7805 | 0.9697 | 0.8649 | 0.008 |
| KNN | 0.8636 | 0.8197 | 0.8151 | 0.7619 | 0.9697 | 0.8533 | 0.005 |
| Gaussian Naive Bayes | 0.8388 | 0.8197 | 0.8120 | 0.8056 | 0.8788 | 0.8406 | 0.004 |
| SVM (RBF Kernel) | 0.8884 | 0.8197 | 0.8350 | 0.7619 | 0.9697 | 0.8533 | 0.012 |
| Decision Tree | 0.9091 | 0.7869 | 0.8216 | 0.7632 | 0.8788 | 0.8169 | 0.005 |

**Random Forest** delivered the best test accuracy and F1 score, and was persisted as the final model.
Full details, plots (confusion matrices, ROC curves, feature importances) and discussion are in the
notebook.

---

## 🚀 Getting Started

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd heart-disease-ml-project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the notebook
```bash
jupyter notebook notebooks/ML_Data_Preprocessing_Predictive_Modeling.ipynb
```

### 5. (Optional) Load the saved model for inference
```python
import joblib
model = joblib.load("models/best_model_Random_Forest.pkl")
scaler = joblib.load("models/standard_scaler.pkl")
features = joblib.load("models/selected_features.pkl")

# X_new must contain the same engineered/encoded columns as during training,
# then be scaled with `scaler` and subset to `features` before calling:
# model.predict(X_new[features])
```

---

## 🧠 Key Findings

- Chest pain type (`cp`), max heart rate (`thalach`), ST depression (`oldpeak`), and number of major
  vessels (`ca`) were consistently the strongest predictors across tree-based models — matching
  established clinical cardiology risk factors.
- Ensemble (Random Forest) and kernel-based (SVM) methods gave the most stable performance across
  the holdout test set and 5-fold cross-validation.
- Logistic Regression, despite being the simplest model, remained highly competitive — indicating the
  underlying relationship in this dataset is largely near-linear after scaling.

## ⚠️ Limitations

- Small dataset (303 records) limits statistical power and increases variance in performance estimates.
- Single-hospital (Cleveland Clinic) origin may limit generalizability to broader populations.
- Missing values were synthetically injected for pipeline demonstration, not organically occurring.
- Only light/default hyperparameter settings were used — no exhaustive tuning was performed.

## 🔭 Future Improvements

- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV` / Optuna.
- SHAP/LIME explainability for per-patient interpretability.
- Deployment as a REST API (FastAPI/Flask) for real-time inference.
- Validation against other UCI heart disease subsets (Hungarian, Switzerland, VA Long Beach) for
  cross-population generalization testing.

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `Jupyter Notebook`

---

## 📄 License

This project uses the publicly available UCI Heart Disease dataset for educational purposes.
Code in this repository is provided under the MIT License — feel free to use, modify, and extend it.
