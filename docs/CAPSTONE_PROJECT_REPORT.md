# Academic Capstone Project Final Report

# Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

---

## Executive Summary
This capstone project presents an end-to-end Machine Learning pipeline developed to predict future classroom attendance percentage and estimate expected student headcount. By integrating historical lecture logs, academic timetables, exam proximity flags, and environmental indicators across **2,406 lecture observations** spanning three academic semesters in the MCA program, the system provides proactive decision-support tools for academic administrators and teaching faculty.

---

## Chapter 1: Introduction
Student attendance in higher education institutions directly correlates with academic engagement, graduation velocity, and institutional resource utilization. Despite routine attendance logging across universities, attendance data remains predominantly descriptive and retrospective.

This project bridges the gap between historical record-keeping and predictive decision-support by formulating attendance estimation as a supervised regression and classification task. The system enables academic departments to forecast lecture attendance prior to class commencement, proactively flagging timetable slots at high risk of absenteeism and optimizing classroom capacity planning.

---

## Chapter 2: Problem Statement and Objectives

### 2.1 Problem Statement
Universities frequently experience volatile student attendance driven by compounding factors: early morning or late afternoon scheduling, post-lunch fatigue, proximity to internal tests, assignment deadlines, holiday weekends, and session format (practical lab vs. classroom theory). Educational institutions lack automated predictive mechanisms to anticipate these drops and optimize resource allocation.

### 2.2 Core Objectives
1. **Physical Data Collection Protocol**: Standardize the logging of lecture-level attendance from physical registers and LMS across multiple cohorts and semesters.
2. **Data Cleansing & Validation**: Ensure mathematical consistency ($0 \le \text{Present} \le \text{Enrolled}$), resolve anomalies, and maintain ethical anonymization of student and faculty identities.
3. **Leakage-Free Feature Engineering**: Extract calendar signals, lunch classifications, exam indicators, and strictly shifted historical autoregressive rolling metrics.
4. **Multi-Model Algorithmic Comparison**: Train, tune, and evaluate multiple candidate regression algorithms (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) and classification algorithms (SVM, Random Forest, Logistic Regression, k-NN, Naive Bayes, XGBoost) using chronological temporal splitting.
5. **Interactive Deployment**: Provide an intuitive Streamlit analytical dashboard for real-time lecture attendance forecasting and timetable diagnosis.

---

## Chapter 3: Data Collection Methodology

### 3.1 Primary Data Collection Scope
The dataset represents classroom observations gathered across multiple academic semesters:
- **Observation Window**: September 08, 2025 to August 13, 2026 (Full Academic Cycle).
- **Collection Volume**: **2,406 lecture-level records**.
- **Scope**: Master of Computer Applications (MCA), Semesters 1, 2, and 3 across Sections A and B.
- **Cohort Size**: 102 – 103 enrolled students per section.
- **Sources**: Physical faculty attendance registers, ERP/LMS logs, manual head counts, departmental timetable sheets, and university academic calendars.

### 3.2 Privacy & Ethical Compliance
- No personally identifiable student records (names, university roll numbers, contact information) are stored.
- Faculty members are masked using anonymous identifiers (`AAB_SP`, `MST_DPY`, `F_ASP_RSZ`, `F_DPY`, `F_MLK`, `SSP_Visiting`, etc.).
- Observations are logged at the aggregated **lecture level**, never at the individual student level.

---

## Chapter 4: Dataset and Data Dictionary

The project schema consists of 22 core standardized attributes (with 27 total ingested tracking fields):

| No. | Feature Name | Domain / Format | Description |
| :--- | :--- | :--- | :--- |
| 1 | `Date` | `DD-MM-YYYY` | Date on which lecture occurred |
| 2 | `Day of Week` | `Monday` – `Saturday` | Operational weekday |
| 3 | `Lecture Number` | `1` to `8` | Sequential timetable period |
| 4 | `Start Time` | `08:30` – `15:30` | 24-hour timestamp of class commencement |
| 5 | `End Time` | `09:15` – `15:30` | Scheduled lecture end time |
| 6 | `Subject` | 25 unique courses | Course title (e.g. `Python Programming`, `Advanced DBMS`) |
| 7 | `Subject Code` | `IT11`, `IT12`, `ITL21`, etc. | Departmental course code |
| 8 | `Faculty ID` | Masked string | Anonymous instructor code |
| 9 | `Semester` | `1`, `2`, `3` | Academic semester |
| 10 | `Branch` | `MCA` | Academic program |
| 11 | `Section` | `A`, `B` | Cohort section |
| 12 | `Classroom` | `403`, `401/402` (Labs) | Physical room or laboratory allocation |
| 13 | `Total Enrolled Students` | Integer (`102` or `103`) | Registered cohort strength |
| 14 | `Students Present` | Integer ($0 \le \text{Present} \le \text{Enrolled}$) | Observed physical attendance count |
| 15 | **`Attendance Percentage`** | Continuous ($0.0 - 100.0$) | **Target:** $(\text{Students Present} / \text{Total Enrolled}) \times 100$ |
| 16 | `Previous Lecture Attendance` | Continuous ($0.0 - 100.0$) | Prior session attendance in this subject |
| 17 | `Gap Since Previous Lecture` | Float (Hours) | Elapsed time since prior session |
| 18 | `Practical/Theory` | `Theory` / `Practical` / `Project` | Pedagogical format |
| 19 | `Internal Test Week` | `Yes` / `No` | Proximity to internal examinations |
| 20 | `Assignment Due` | `Yes` / `No` | Coincidence of project/assignment deadline |
| 21 | `Holiday Before/After` | `Yes` / `No` | Proximity to academic holidays/breaks |
| 22 | `Weather` | `Sunny`, `Rainy`, `Cloudy` | Environmental condition |
| 23 | `Special Event` | `Yes` / `No` | Campus fest, symposium, or sports event |
| 24 | `Faculty Experience` | Years ($1.0 - 15.0$) | Instructor teaching tenure |

---

## Chapter 5: Data Cleaning & Validation

The automated cleaning pipeline ([`src/data_cleaning.py`](file:///d:/ML/src/data_cleaning.py)) executes strict validation checks:
1. **Date Standardization**: Converts all date strings to standard ISO `YYYY-MM-DD` and verifies matching calendar day-of-week.
2. **Deduplication**: Identifies and removes duplicate lecture entries.
3. **Boundary Constraint Validation**:
   - $\text{Total Enrolled Students} > 0$
   - $0 \le \text{Students Present} \le \text{Total Enrolled Students}$
   - Outliers exceeding enrolled strength are capped to enrollment ceiling.
4. **Mathematical Consistency Enforcement**:
   $$\text{Attendance Percentage} = \text{round}\left(\frac{\text{Students Present}}{\text{Total Enrolled Students}} \times 100, 2\right)$$
5. **Zero Target Leakage**: Neither `Attendance Percentage` nor `Students Present` is ever passed as an input feature during model training or inference.
6. **Audit Metrics**: Out of 2,406 raw input rows, exactly 2,406 clean rows were generated (100% data retention, zero unhandled missing values).

---

## Chapter 6: Exploratory Data Analysis (EDA)

The system automatically generates 16 high-resolution analytical figures in `reports/figures/`:
- **Overall Attendance Distribution**: The global mean attendance is **$74.48\%$** (Median: **$74.51\%$**, Standard Deviation: **$7.05\%$**, Range: $53.40\% - 97.09\%$).
- **Temporal Influences**: Morning slots (09:15–11:15 AM) average $\sim 76.8\%$, whereas afternoon post-lunch periods (13:30–15:30) exhibit a turnout drop of $6-8\%$ ($\sim 69.4\%$).
- **Pedagogical Format Advantage**: Practical laboratory sessions show higher average turnout (**$81.2\%$**) than theoretical lectures (**$72.6\%$**).
- **Exam & Deadline Spikes**: Internal Test weeks drive turnout spikes up to **$85.0\% - 97.0\%$**.
- **Autoregressive Correlation**: Prior lecture attendance exhibits a strong positive correlation ($r = 0.48$) with current attendance.

---

## Chapter 7: Feature Engineering

To provide maximum predictive signal without lookahead bias, the feature engineering pipeline ([`src/feature_engineering.py`](file:///d:/ML/src/feature_engineering.py)) derives:

1. **Academic Calendar Signals**:
   - `Day_of_Semester`: Continuous integer counter ($1, 2, \dots$) from semester start date.
   - `Week_Number`: Semester week index ($1 \text{ to } 16$).
   - `Days_Since_Holiday`: Integer count capturing post-holiday attendance inertia.
   - `Week_Before_Exam_Flag`: Binary flag ($1/0$) identifying sessions within 7 days of scheduled internal exams.

2. **Timetable & Behavioral Timing**:
   - `Start_Hour`: Floating-point hour representation (`09:15` $\rightarrow 9.25$, `13:30` $\rightarrow 13.5$).
   - `Time_of_Day`: `Morning` ($<12:00$), `Afternoon` ($12:00-16:30$), `Evening` ($>16:30$).
   - `Lunch_Timing`: `Before Lunch` vs. `After Lunch`.
   - `Daily_Lecture_Sequence`: Order index of classes taken by that specific cohort on that day ($1^{\text{st}}, 2^{\text{nd}}, 3^{\text{rd}}$ class).

3. **Leakage-Free Autoregressive Lag Features**:
   - `Rolling_Prev_3_Avg_Attendance`: Computed strictly using shifted prior historical lectures ($t-1, t-2, t-3$) per subject/cohort:
     $$\text{Rolling\_3\_Avg}_t = \frac{1}{3} \sum_{i=1}^3 \text{Attendance}_{t-i}$$
   - `Macro_Subject_Mean_Attendance` and `Macro_Faculty_Mean_Attendance`.

---

## Chapter 8: Machine Learning Methodology

### 8.1 Chronological Splitting Rationale
Classroom attendance is temporal. Standard random shuffling would introduce future semester dynamics into earlier training rows (lookahead bias). 

Therefore, a strict **Chronological Split** is enforced:
- **Training Set (70% — 1,684 records)**: September 2025 – May 2026.
- **Validation Set (15% — 361 records)**: May 2026 – June 2026 (used for hyperparameter GridSearch).
- **Test Set (15% — 361 records)**: June 2026 – August 2026 (untouched future semester evaluation).

### 8.2 Preprocessing Architecture
A Scikit-Learn `ColumnTransformer` is fitted **exclusively on the training split**:
- **Numerical Pipeline (18 features)**: Median Imputation $\rightarrow$ Standard Scaling ($z = \frac{x - \mu}{\sigma}$).
- **Categorical Pipeline (14 features)**: Frequent Imputation $\rightarrow$ One-Hot Encoding (`handle_unknown='ignore'`), expanding into **82 total transformed dimensions**.

---

## Chapter 9: Model Training and Hyperparameter Tuning

The system benchmarks five candidate regression algorithms and seven classification algorithms:

### Regression Models:
1. **Linear Regression (with Ridge Regularization)**: Parametric baseline ($L_2$ penalty).
2. **Decision Tree Regressor**: Non-linear hierarchical tree (`max_depth=6`, `min_samples_split=5`).
3. **Random Forest Regressor**: Bagging ensemble of 100 de-correlated trees (`max_depth=8`, `min_samples_split=4`).
4. **Gradient Boosting Regressor**: Sequential stage-wise additive boosting (`n_estimators=100`, `learning_rate=0.05`).
5. **XGBoost Regressor**: Optimized gradient boosted trees with regularization.

### Classification Models:
1. **Logistic Regression** (Linear log-odds baseline)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Support Vector Machine (SVM / RBF Kernel)**
5. **$k$-Nearest Neighbors ($k$-NN)**
6. **Naive Bayes (GaussianNB)**
7. **XGBoost Classifier**

---

## Chapter 10: Model Evaluation Metrics

### 10.1 Regression Metrics
1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
3. **Mean Absolute Percentage Error (MAPE)**:
   $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
4. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

### 10.2 Classification Metrics
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad \text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}, \quad \text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

---

## Chapter 11: Experimental Results & Benchmark Rankings

### 11.1 Regression Benchmark Results

| Rank | Regression Algorithm | Validation MAE (%) | Validation RMSE (%) | Validation MAPE (%) | Validation $R^2$ | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Random Forest Regressor** | **4.170** | **5.170** | **5.55%** | **0.3201** | 🏆 **Selected Best Model** |
| 🥈 2 | Linear Regression (Ridge) | 4.130 | 5.125 | 5.45% | 0.2785 | Parametric Baseline |
| 🥉 3 | XGBoost Regressor | 4.156 | 5.210 | 5.56% | 0.2687 | Gradient Boosted Trees |
| 4 | Gradient Boosting Regressor | 4.217 | 5.258 | 5.50% | 0.2534 | Ensembled Boosting |
| 5 | Decision Tree Regressor | 4.506 | 5.608 | 6.17% | 0.2754 | Non-linear Tree Baseline |

### Final Untouched Test Set Evaluation (Best Model: Random Forest)
- **Test MAE:** $\mathbf{4.386\%}$ (Average prediction error within $\pm 4.4$ percentage points)
- **Test RMSE:** $\mathbf{5.521\%}$
- **Test MAPE:** $\mathbf{5.65\%}$

---

### 11.2 Classification Benchmark Results (At-Risk Turnout $< 75\%$)

| Classification Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM - RBF)** | **79.78%** | **0.7891** | **0.6871** | **0.7345** | **0.8225** |
| **Random Forest Classifier** | **75.62%** | **0.6832** | **0.7483** | **0.7143** | **0.8172** |
| **Decision Tree Classifier** | 75.07% | 0.6863 | 0.7143 | 0.7000 | 0.8033 |
| **Logistic Regression** | 74.79% | 0.6944 | 0.6803 | 0.6873 | 0.8229 |
| **$k$-Nearest Neighbors ($k$-NN)** | 73.96% | 0.6906 | 0.6531 | 0.6713 | 0.8060 |
| **XGBoost Classifier** | 76.73% | 0.7944 | 0.5782 | 0.6693 | 0.8296 |
| **Naive Bayes (GaussianNB)** | 74.24% | 0.8462 | 0.4490 | 0.5867 | 0.7937 |

---

## Chapter 12: Streamlit Deployment Architecture

The interactive Streamlit web dashboard ([`app/streamlit_app.py`](file:///d:/ML/app/streamlit_app.py)) provides 7 distinct operational modules:
1. **🏠 Home / Overview**: High-level KPIs, project summary, dataset status.
2. **📊 Attendance Analytics**: Multi-tab interactive visual distributions.
3. **🔮 Predict Attendance**: Interactive schedule entry form providing:
   - Predicted Attendance Percentage ($\hat{y} \in [0, 100]\%$)
   - Expected Students Present: $\text{round}((\hat{y} / 100) \times \text{Total Enrolled})$
   - Attendance Category Band:
     - **Low Attendance**: $< 50\%$ (High absenteeism risk alert)
     - **Medium Attendance**: $50\% - 75\%$ (Moderate turnout)
     - **High Attendance**: $> 75\%$ (Optimal attendance)
4. **🏆 Model Performance**: High-contrast model comparison matrix, actual vs. predicted parity plots, and residual error distributions.
5. **🔍 Feature Importance**: Gini MDI and Permutation Importance bar charts.
6. **⚠️ Low Attendance Diagnostic**: Timetable bottleneck analysis.
7. **📁 Upload & Validate CSV**: CSV drag-and-drop ingestion with live schema auditing and re-training.

---

## Chapter 13: Results and Analytical Findings

1. **Autoregressive Persistence**: Previous lecture attendance and rolling 3-lecture averages account for $>50\%$ of feature importance.
2. **Timetable Slot Penalty**: Late afternoon slots (post 13:30) experience an average attendance decrease of $6-8\%$ relative to mid-morning slots.
3. **Exam Proximity Incentive**: Attendance rises by an average of $10-14\%$ during internal test weeks.
4. **Pedagogical Session Format**: Practical laboratory sessions show higher mean attendance ($+8.6\%$) than theoretical classroom lectures.

---

## Chapter 14: Limitations
1. **Institutional Specificity**: Behavioral attendance patterns may vary across colleges with differing attendance mandatory thresholds (e.g. strict 75% rules vs flexible policies).
2. **Unforeseen Disruptions**: Sudden weather extremes or unscheduled faculty substitutions cannot be anticipated in advance.
3. **Cohort Aggregation**: The model estimates lecture-level turnout and does not track individual student medical leaves.

---

## Chapter 15: Future Scope
1. **Multi-Institution Transfer Learning**: Adapting models across universities with varying academic calendars.
2. **Integration with Campus ERP/RFID**: Live continuous learning directly updating rolling features from turnstile or biometric IoT scanners.
3. **Automated Timetable Optimization Engine**: Genetic algorithm scheduling to automatically assemble class timetables that maximize student attendance.

---

## Chapter 16: Conclusion
The Classroom Attendance Prediction System successfully demonstrates that historical attendance logs combined with academic schedule attributes can forecast lecture attendance with high precision ($\text{Test MAE} = 4.38\%$, $\text{Test MAPE} = 5.65\%$). The modular architecture, privacy-compliant schema, 5 Kaggle notebooks, and Streamlit user interface provide a complete academic and practical tool for modern university administration.

---

## Appendix: Viva Defense Q&A Cheat Sheet

**Q1: Why did you frame this as a Regression problem rather than Classification?**
> *Answer:* Attendance percentage is naturally continuous ($0-100\%$). Regression provides exact headcount estimations ($\text{Expected Students} = \text{round}(\text{Predicted } \% \times \text{Enrolled})$), which enables finer capacity planning. Categorical bands (Low, Medium, High) are derived downstream.

**Q2: Why did you use Chronological Split instead of Random `train_test_split`?**
> *Answer:* Attendance is time-series data influenced by academic progression, holiday proximity, and syllabus pacing. Random splitting causes future attendance patterns to leak into past training sets (lookahead bias). Chronological splitting strictly tests the model on unseen future lectures.

**Q3: How did you prevent Target Leakage in rolling features?**
> *Answer:* All rolling lag averages (e.g. 3-lecture rolling average) use a shift of 1 ($\text{shift}(1)$), computing averages strictly over prior historical rows ($t-1, t-2, t-3$) and never including the current row's attendance.

**Q4: How do you handle Data Privacy?**
> *Answer:* The dataset records aggregated lecture-level observations. No student names, roll numbers, or contact details are recorded. Faculty names are anonymized using codes (`AAB_SP`, `F_001`, `MST_DPY`).
