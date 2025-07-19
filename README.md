# Hospreadmission
Prediction - Hospital Readmission
# Hospital Readmission Prediction and Clinical Entity Extraction
This project focuses on predicting hospital readmissions within 30 days and extracting clinical entities from discharge notes using machine learning and NLP techniques. The aim is to identify high-risk patients and derive meaningful insights from unstructured clinical text.

##  Files in This Repository
- `finalhosp.ipynb` – Contains all code (data prep, model training, evaluation, and NER)
- `finalreport.pdf` – Final report summarizing methodology and findings
- `requirements.txt` – List of Python dependencies
## 🔍 Project Overview
###  Objectives
- Predict 30-day hospital readmissions using structured and unstructured data
- Extract key clinical entities (diagnoses, medications, follow-up instructions) from discharge notes using Flan-T5

###  Dataset
- **File**: `Assignment_Data.csv`
- Contains patient age, gender, length of stay, diagnosis codes, medication types, and discharge notes
- Target variable: `readmitted_30_days` (1 = readmitted, 0 = not)
##  Approach
### Data Preprocessing
- Handled class imbalance with **SMOTE**
- Scaled numerical features (`age`, `length_of_stay`, `num_previous_admissions`)
- One-hot encoded categorical features
- Extracted flags from text (e.g. `has_followup`, `has_surgery`)
- Applied **TF-IDF** on discharge notes
###  Model Training
- **XGBoost** (tuned using GridSearchCV)
- **Logistic Regression** (with class balancing)
- Feature selection via XGBoost importance
### NER (Clinical Entity Extraction)
- Used **Flan-T5** for Named Entity Recognition
- Prompted to extract medications, diagnoses, follow-up notes
## Results
### Readmission Prediction
| Model                     | AUC-ROC | F1 Score |
|--------------------------|---------|----------|
| XGBoost (Full Features)  | 0.595   | 0.444    |
| Logistic Regression       | 0.35    | 0.333    |
| XGBoost (Top 30 Features) | 0.59    | 0.333    |
- XGBoost consistently outperformed Logistic Regression
- `age` and `length_of_stay` were key predictors
### Clinical Entity Extraction
- Flan-T5 extracted medications like "antibiotics" and conditions like "pneumonia"
- Occasionally missed nuanced phrases like unclear "follow-up" instructions

##How to Run
### Run in Google Colab:
1. Open the notebook file
2. Run all cells top to bottom
3. Dependencies install automatically

