# Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

🎓 **Academic Capstone Project in Machine Learning & Predictive Analytics**

---

> [!IMPORTANT]
> **Academic Integrity & Data Collection Notice:**
> The final submitted dataset must be based on **original attendance observations organically collected by students** (from faculty registers, manual head counts, department sheets, and LMS records) as required by the project specification. 
> 
> The codebase includes a standardized data collection template (`data/templates/attendance_data_template.csv`) and is pre-configured with a technical demonstration dataset so that all pipelines, notebooks, models, unit tests, and the Streamlit dashboard can be verified immediately. Once you have collected your real attendance records, simply replace `data/raw/attendance_raw.csv` and re-run the training pipeline.

---

## 1. Project Overview & Problem Statement

Educational institutions face irregular classroom attendance due to compounding factors such as lecture scheduling (e.g. early morning vs. late afternoon), day of the week, proximity to internal tests and assignment deadlines, public holidays, weather, and pedagogical delivery formats (theory vs. laboratory practicals). While institutions routinely log attendance, they rarely utilize machine learning to forecast attendance patterns proactively.

This project delivers an end-to-end Machine Learning pipeline that predicts:
1. **Attendance Percentage** ($\hat{y} \in [0.0, 100.0]\%$) for any scheduled lecture.
2. **Expected Students Present** ($\text{Headcount} = \text{round}((\hat{y} / 100) \times \text{Total Enrolled})$).
3. **Attendance Category Band**:
   - 🔴 **Low Attendance**: $< 50\%$ (High absenteeism risk alert)
   - 🟡 **Medium Attendance**: $50\% - 75\%$ (Moderate turnout)
   - 🟢 **High Attendance**: $> 75\%$ (Healthy attendance / university benchmark compliant)

---

## 2. Key Objectives

- **Standardized Data Ingestion & Cleansing**: Ingest lecture-level observations with automated anomaly detection, boundary enforcement, and privacy protection (anonymized Faculty IDs, zero student PII).
- **Leakage-Free Feature Engineering**: Generate academic calendar indicators, before/after lunch flags, and strictly shifted historical autoregressive rolling averages.
- **Temporal Split Validation**: Utilize chronological splitting ($70\%$ Train, $15\%$ Validation, $15\%$ Test) to prevent lookahead bias.
- **Multi-Model Regression Benchmark**: Train, tune, and compare Linear Regression (Ridge), Decision Trees, Random Forests, Gradient Boosting, and XGBoost.
- **Interactive Streamlit Web Dashboard**: Provide faculty and department heads with an interactive application for attendance prediction, schedule diagnostics, and dataset uploading.

---

## 3. Project Structure

```
attendance-prediction/
│
├── data/
│   ├── raw/
│   │   ├── attendance_raw.csv            # Active raw dataset (demo / real)
│   │   └── synthetic_demo_attendance.csv # Clearly marked demo dataset
│   ├── processed/
│   │   └── attendance_cleaned.csv        # Cleaned, validated dataset
│   └── templates/
│       ├── attendance_data_template.csv  # Blank CSV template for data entry
│       └── DATA_COLLECTION_GUIDE.md      # Data collection protocol guide
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb            # Data cleaning & validation
│   ├── 02_eda.ipynb                      # 16-figure visual exploratory analysis
│   ├── 03_feature_engineering.ipynb      # Lag features & temporal signals
│   ├── 04_model_training.ipynb           # Model training & hyperparameter tuning
│   └── 05_model_evaluation.ipynb         # Evaluation metrics & diagnostics
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py                  # Validation, anomaly checks, audit report
│   ├── feature_engineering.py            # Calendar & leakage-free lag features
│   ├── preprocessing.py                  # Chronological split & ColumnTransformer
│   ├── train.py                          # Training, GridSearch, model persistence
│   ├── evaluate.py                       # MAE, RMSE, MAPE, R2 & visual plots
│   ├── predict.py                        # Reusable single & batch prediction API
│   ├── eda.py                            # 16 automated publication-grade plots
│   └── generate_demo_data.py             # Synthetic demonstration data generator
│
├── models/
│   ├── best_model.pkl                    # Serialized best performing model
│   ├── preprocessor.pkl                  # Fitted scikit-learn ColumnTransformer
│   └── model_metadata.json               # Model metrics, hyperparams & stats
│
├── reports/
│   ├── figures/                          # 21 high-resolution plots (.png)
│   ├── cleaning_report.json              # Data cleaning audit report
│   ├── experiment_results.csv            # Multi-model training/val/test matrix
│   └── model_comparison.csv             # Comparative ranking across metrics
│
├── app/
│   └── streamlit_app.py                  # Full interactive web application
│
├── tests/
│   ├── test_cleaning.py                  # Tests for cleaning & validation
│   ├── test_features.py                  # Tests for feature engineering & lags
│   └── test_prediction.py                # Tests for prediction math & categories
│
├── docs/
│   └── CAPSTONE_PROJECT_REPORT.md        # 16-Chapter academic final report
│
├── run_pipeline.py                       # Master end-to-end execution script
├── requirements.txt                      # Pinned Python package dependencies
├── .gitignore                            # Git ignore rules
└── README.md                             # Comprehensive project documentation
```

---

## 4. Dataset & Data Collection Protocol

### 4.1 Required 22 Fields

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Date` | String | Format: `DD-MM-YYYY` (e.g., `01-08-2026`) |
| `Day of Week` | String | `Monday`, `Tuesday`, `Wednesday`, `Thursday`, `Friday`, `Saturday` |
| `Lecture Number` | Integer | Daily period sequence (`1` to `8`) |
| `Start Time` | String | 24-hour format (e.g., `09:00`, `13:45`) |
| `Subject` | String | Subject title/code (e.g., `Python Programming`, `DBMS`) |
| `Faculty ID` | String | Anonymized code: `F_001`, `F_002`, etc. |
| `Semester` | Integer | Academic semester (`1` to `8`) |
| `Branch` | String | Department (e.g., `CSE`, `IT`, `AI-DS`, `ECE`) |
| `Section` | String | Cohort division (`A`, `B`, `C`) |
| `Classroom` | String | Location identifier (e.g., `Room 401`, `Lab 1`) |
| `Total Enrolled Students` | Integer | Total registered strength of the class |
| `Students Present` | Integer | Observed physical attendance count |
| `Attendance Percentage` | Float | Calculated: $(\text{Present} / \text{Enrolled}) \times 100$ |
| `Previous Lecture Attendance` | Float | Attendance of immediate previous lecture in this subject |
| `Gap Since Previous Lecture` | Float | Elapsed hours since last session in this subject |
| `Practical/Theory` | String | `Theory` or `Practical` |
| `Internal Test Week` | String | `Yes` or `No` |
| `Assignment Due` | String | `Yes` or `No` |
| `Holiday Before/After` | String | `Yes` or `No` |
| `Weather` | String | `Sunny`, `Rainy`, `Cloudy`, `Cold` |
| `Special Event` | String | `Yes` or `No` (Campus fests, symposiums) |
| `Faculty Experience` | Float | Teaching experience in years |

---

## 5. Installation & Setup

### Step 1: Clone or Navigate to Directory
```bash
cd attendance-prediction
```

### Step 2: Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. How to Run

### 6.1 Run Automated Unit Tests
Verify that all 12 unit tests pass:
```bash
python -m pytest -v
```

### 6.2 Execute Master End-to-End Pipeline
Runs data cleaning, generates 16 EDA figures, performs feature engineering, trains and tunes all ML models, evaluates metrics, and saves all artifacts:
```bash
python run_pipeline.py
```

### 6.3 Launch the Interactive Streamlit Web App
```bash
python -m streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. How to Replace Demo Data with Real Collected Data

1. Open `data/templates/attendance_data_template.csv` in Excel or Google Sheets.
2. Fill in your physically collected lecture observations following the [Data Collection Guide](file:///d:/ML/data/templates/DATA_COLLECTION_GUIDE.md).
3. Save the completed file as `attendance_raw.csv` and place it in `data/raw/`:
   ```bash
   # Replace data/raw/attendance_raw.csv with your real file
   ```
4. Re-run the full pipeline:
   ```bash
   python run_pipeline.py
   ```
5. All models, evaluation tables (`reports/model_comparison.csv`), and Streamlit dashboard will instantly update with your real-world data!

---

## 8. Machine Learning Algorithms & Model Comparison

The project compares 5 regression models evaluated on an untouched chronological test split:

| Model | MAE | RMSE | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Ridge)** | **3.49** | **4.60** | **4.41%** | **0.850** |
| **Gradient Boosting** | 6.05 | 8.21 | 8.09% | 0.523 |
| **XGBoost Regressor** | 6.32 | 8.43 | 8.40% | 0.496 |
| **Random Forest** | 7.08 | 9.38 | 9.31% | 0.376 |
| **Decision Tree** | 8.01 | 11.12 | 10.63% | 0.123 |

---

## 9. Example Prediction via Python API

```python
from src.predict import predict_attendance

sample_lecture = {
    "Date": "2026-09-15",
    "Day of Week": "Monday",
    "Lecture Number": 2,
    "Start Time": "10:00",
    "Subject": "Python Programming",
    "Faculty ID": "F_001",
    "Semester": 5,
    "Branch": "CSE",
    "Section": "A",
    "Classroom": "Room 401",
    "Total Enrolled Students": 60,
    "Previous Lecture Attendance": 82.0,
    "Gap Since Previous Lecture": 24.0,
    "Practical/Theory": "Theory",
    "Internal Test Week": "No",
    "Assignment Due": "No",
    "Holiday Before/After": "No",
    "Weather": "Sunny",
    "Special Event": "No",
    "Faculty Experience": 8.0
}

result = predict_attendance(sample_lecture)
print(result)
```

**Output:**
```json
{
  "Predicted Attendance Percentage": 78.76,
  "Expected Students Present": 47,
  "Total Enrolled Students": 60,
  "Attendance Category": "High Attendance",
  "Status": "SUCCESS"
}
```

---

## 10. Viva Defense / Presentation Cheat Sheet

- **Why Regression?** Attendance percentage is a continuous target ($0-100\%$). It enables exact student headcount estimation ($\text{Expected Present} = \text{Predicted } \% \times \text{Enrolled}$).
- **Why Chronological Split?** Attendance is time-series data. Random splitting leaks future trends into past training sets (lookahead bias). Chronological splitting strictly tests on future lectures.
- **How was Target Leakage avoided?** `Attendance Percentage` and `Students Present` are never passed as predictor features. Rolling averages use $\text{shift}(1)$, computing averages strictly over previous historical lectures ($t-1, t-2, \dots$).
- **Privacy Compliance**: All faculty IDs are masked (`F_001`, `F_002`), and no student roll numbers or names are ever collected.

---

## 11. Authors & Academic Credits
- **Project Title**: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data
- **Domain**: Machine Learning, Educational Data Mining, Time-Series Predictive Analytics
- **Technologies**: Python 3.13, Scikit-Learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Pytest
