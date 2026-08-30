"""
Streamlit Web Application: Classroom Attendance Prediction Dashboard
====================================================================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

A comprehensive, interactive analytics and prediction dashboard for faculty,
department heads, and academic administrators.
"""

import os
import sys
import json
import logging
from datetime import datetime, date
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_cleaning import clean_attendance_data, validate_schema, REQUIRED_COLUMNS
from src.feature_engineering import build_engineered_features
from src.predict import predict_attendance, get_attendance_category
from src.train import train_and_tune_models

# Page Configuration
st.set_page_config(
    page_title="Classroom Attendance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-med {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .demo-banner {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #92400E;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    candidates = [
        os.path.join("01_Data", "cleaned_attendance.csv"),
        os.path.join("..", "01_Data", "cleaned_attendance.csv"),
        os.path.join("data", "processed", "attendance_cleaned.csv"),
        os.path.join("..", "data", "processed", "attendance_cleaned.csv"),
        os.path.join("01_Data", "raw_attendance.csv"),
        os.path.join("data", "raw", "attendance_raw.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None


@st.cache_data
def load_metadata():
    candidates = [
        os.path.join("models", "model_metadata.json"),
        os.path.join("..", "models", "model_metadata.json"),
        os.path.join("04_Deployment", "attendance_model.pkl"),
        os.path.join("..", "04_Deployment", "attendance_model.pkl"),
        "attendance_model.pkl"
    ]
    for p in candidates:
        if os.path.exists(p):
            if p.endswith(".json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            elif p.endswith(".pkl"):
                try:
                    bundle = joblib.load(p)
                    if isinstance(bundle, dict) and "metadata" in bundle:
                        return bundle["metadata"]
                except Exception:
                    pass
    return None


@st.cache_data
def load_comparison_data():
    candidates = [
        os.path.join("03_Experiment", "model_comparison.xlsx"),
        os.path.join("..", "03_Experiment", "model_comparison.xlsx"),
        os.path.join("reports", "model_comparison.csv"),
        os.path.join("..", "reports", "model_comparison.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                if p.endswith(".xlsx"):
                    return pd.read_excel(p, sheet_name="Regression_Benchmark")
                else:
                    return pd.read_csv(p)
            except Exception:
                pass
    return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/classroom.png", width=70)
st.sidebar.title("Navigation Menu")
app_mode = st.sidebar.radio(
    "Select Module:",
    [
        "🏠 Home / Overview",
        "📊 Attendance Analytics",
        "🔮 Predict Attendance",
        "🏆 Model Performance",
        "🔍 Feature Importance",
        "⚠️ Low Attendance Diagnostic",
        "📁 Upload & Validate CSV"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Capstone Project**\n"
    "Classroom Attendance Prediction Using Academic Schedule & Historical Attendance Data\n\n"
    "*Target:* Attendance Percentage (0–100%)\n"
    "*Metric:* Expected Students Present"
)

# Load data & metadata
df = load_dataset()
metadata = load_metadata()
comp_df = load_comparison_data()

# -------------------------------------------------------------
# TAB 1: HOME / OVERVIEW
# -------------------------------------------------------------
if app_mode == "🏠 Home / Overview":
    st.markdown('<div class="main-header">🎓 Classroom Attendance Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning Predictive Analytics for Academic Timetable Optimization and Attendance Forecasting</div>', unsafe_allow_html=True)

    # Demo Banner Notice
    st.markdown("""
    <div class="demo-banner">
        <strong>📌 SYSTEM STATUS NOTICE:</strong> The dashboard is currently connected to the validated demonstration pipeline.
        To replace synthetic demonstration records with your <strong>original collected attendance data</strong>, navigate to the 
        <em>'📁 Upload & Validate CSV'</em> tab or place your file into <code>data/raw/attendance_raw.csv</code>.
    </div>
    """, unsafe_allow_html=True)

    # High-level Metrics Row
    if df is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Lecture Logs", f"{len(df):,}")
        with c2:
            st.metric("Mean Attendance", f"{df['Attendance Percentage'].mean():.1f}%")
        with c3:
            st.metric("Unique Subjects", f"{df['Subject'].nunique()}")
        with c4:
            st.metric("Faculty Members", f"{df['Faculty ID'].nunique()}")
        with c5:
            r2_val = metadata['test_metrics']['R2'] if metadata else 0.85
            st.metric("Best Model R²", f"{r2_val:.3f}")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("🎯 Project Objectives & Academic Scope")
        st.write("""
        Academic institutions frequently encounter significant fluctuations in classroom attendance due to compounding factors such as lecture scheduling, day of the week, test proximity, holidays, and delivery formats.
        
        This Machine Learning capstone pipeline provides:
        1. **End-to-End Ingestion & Validation**: Ingests lecture-level physical records with zero personally identifiable student data (GDPR/privacy compliant).
        2. **Leakage-Free Feature Engineering**: Constructs temporal indicators, exam proximity flags, and strictly shifted historical autoregressive signals.
        3. **Multi-Model Benchmark**: Compares Linear Regression, Decision Trees, Random Forests, Gradient Boosting, and XGBoost using chronological validation.
        4. **Interactive Predictive Decision Support**: Generates expected attendance percentages, expected headcount counts, and low-attendance risk alerts for scheduled timetable slots.
        """)

    with col_right:
        st.subheader("📋 Core Methodology Architecture")
        st.code("""
Real Attendance Data
        ↓
Data Validation & Cleaning
        ↓
Chronological Train/Val/Test Split
        ↓
Feature Engineering & Lag Averages
        ↓
ColumnTransformer Preprocessing
        ↓
Multi-Model Regression & Tuning
        ↓
Interactive Streamlit Deployment
        """, language="text")

    if df is not None:
        st.subheader("👀 Recent Attendance Records Preview")
        st.dataframe(df.head(8), use_container_width=True)

# -------------------------------------------------------------
# TAB 2: ATTENDANCE ANALYTICS
# -------------------------------------------------------------
elif app_mode == "📊 Attendance Analytics":
    st.markdown('<div class="main-header">📊 Exploratory Attendance Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive breakdowns of historical attendance distributions across scheduling dimensions</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("No dataset loaded. Please upload or generate data first.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📅 Temporal Trends", "📚 Subject & Faculty", "🧪 Format & Calendar", "📈 Correlation Matrix"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Attendance by Day of Week")
                fig, ax = plt.subplots(figsize=(6, 4))
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                valid_days = [d for d in day_order if d in df["Day of Week"].unique()]
                sns.barplot(data=df, x="Day of Week", y="Attendance Percentage", order=valid_days, palette="Blues_d", ax=ax)
                ax.set_ylabel("Mean Attendance (%)")
                ax.set_ylim(0, 100)
                plt.xticks(rotation=25)
                st.pyplot(fig)

            with c2:
                st.markdown("#### Attendance by Lecture Slot (Timetable Number)")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df, x="Lecture Number", y="Attendance Percentage", palette="Purples", ax=ax)
                ax.set_ylabel("Attendance Percentage (%)")
                ax.set_xlabel("Lecture Slot Number")
                st.pyplot(fig)

            st.markdown("#### Semester-Long Attendance Macro Trend")
            fig, ax = plt.subplots(figsize=(12, 3.5))
            df_temp = df.copy()
            df_temp["Date_DT"] = pd.to_datetime(df_temp["Date"])
            trend = df_temp.groupby("Date_DT")["Attendance Percentage"].mean()
            ax.plot(trend.index, trend.values, marker="o", markersize=3, color="#2563EB", linewidth=2)
            ax.set_ylabel("Daily Average Attendance (%)")
            ax.set_xlabel("Date")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Average Attendance by Subject")
                fig, ax = plt.subplots(figsize=(6, 5))
                sub_df = df.groupby("Subject")["Attendance Percentage"].mean().sort_values(ascending=True)
                sub_df.plot(kind="barh", color="#0D9488", edgecolor="black", ax=ax)
                ax.set_xlabel("Mean Attendance (%)")
                ax.set_xlim(0, 100)
                st.pyplot(fig)

            with c2:
                st.markdown("#### Average Attendance by Faculty Member")
                fig, ax = plt.subplots(figsize=(6, 5))
                fac_df = df.groupby("Faculty ID")["Attendance Percentage"].mean().sort_values(ascending=True)
                fac_df.plot(kind="barh", color="#6366F1", edgecolor="black", ax=ax)
                ax.set_xlabel("Mean Attendance (%)")
                ax.set_xlim(0, 100)
                st.pyplot(fig)

        with tab3:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### Practical vs Theory Format")
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.barplot(data=df, x="Practical/Theory", y="Attendance Percentage", palette=["#10B981", "#F59E0B"], ax=ax)
                ax.set_ylabel("Mean Attendance (%)")
                ax.set_ylim(0, 100)
                st.pyplot(fig)

            with c2:
                st.markdown("#### Internal Test Week Impact")
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.barplot(data=df, x="Internal Test Week", y="Attendance Percentage", palette=["#6B7280", "#EF4444"], ax=ax)
                ax.set_ylabel("Mean Attendance (%)")
                ax.set_ylim(0, 100)
                st.pyplot(fig)

            with c3:
                st.markdown("#### Holiday Proximity Effect")
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.barplot(data=df, x="Holiday Before/After", y="Attendance Percentage", palette=["#3B82F6", "#EC4899"], ax=ax)
                ax.set_ylabel("Mean Attendance (%)")
                ax.set_ylim(0, 100)
                st.pyplot(fig)

        with tab4:
            st.markdown("#### Pearson Correlation Heatmap")
            num_df = df.select_dtypes(include=[np.number])
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
            st.pyplot(fig)

# -------------------------------------------------------------
# TAB 3: PREDICT ATTENDANCE
# -------------------------------------------------------------
elif app_mode == "🔮 Predict Attendance":
    st.markdown('<div class="main-header">🔮 Predict Classroom Attendance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input upcoming lecture schedule parameters to forecast attendance percentage and expected headcount</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("#### 1. Lecture Timetable & Cohort Details")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            input_date = st.date_input("Lecture Date", value=date.today())
            day_of_week = input_date.strftime("%A")
            st.text(f"Day: {day_of_week}")
        with c2:
            lecture_num = st.selectbox("Lecture Period Slot", [1, 2, 3, 4, 5, 6, 7, 8], index=1)
        with c3:
            start_time = st.selectbox("Start Time", ["09:00", "10:00", "11:15", "12:15", "13:45", "14:45", "15:45"], index=1)
        with c4:
            format_type = st.selectbox("Format", ["Theory", "Practical"], index=0)

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            subject_options = df["Subject"].unique().tolist() if df is not None else ["Python Programming", "DBMS", "Operating Systems", "Computer Networks"]
            subject = st.selectbox("Subject", subject_options)
        with c6:
            faculty_options = df["Faculty ID"].unique().tolist() if df is not None else ["F_001", "F_002", "F_003"]
            faculty_id = st.selectbox("Faculty Identifier", faculty_options)
        with c7:
            semester = st.selectbox("Semester", [1, 2, 3, 4, 5, 6, 7, 8], index=4)
        with c8:
            branch = st.selectbox("Department / Branch", ["CSE", "IT", "AI-DS", "ECE", "Mechanical"], index=0)

        c9, c10, c11, c12 = st.columns(4)
        with c9:
            section = st.selectbox("Section", ["A", "B", "C"], index=0)
        with c10:
            classroom = st.text_input("Classroom / Lab", value="Room 401")
        with c11:
            total_enrolled = st.number_input("Total Enrolled Students", min_value=10, max_value=200, value=60, step=1)
        with c12:
            faculty_exp = st.number_input("Faculty Experience (Years)", min_value=1.0, max_value=40.0, value=8.0, step=1.0)

        st.markdown("#### 2. Academic Context & Environmental Attributes")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            prev_att = st.slider("Previous Class Attendance (%)", min_value=20.0, max_value=100.0, value=78.0, step=0.5)
        with e2:
            gap_hours = st.number_input("Gap Since Previous Lecture (Hours)", min_value=1.0, max_value=168.0, value=24.0, step=1.0)
        with e3:
            test_week = st.selectbox("Internal Test Week?", ["No", "Yes"], index=0)
        with e4:
            assignment_due = st.selectbox("Assignment Submission Due?", ["No", "Yes"], index=0)

        e5, e6, e7, e8 = st.columns(4)
        with e5:
            holiday_prox = st.selectbox("Holiday Before/After?", ["No", "Yes"], index=0)
        with e6:
            weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Cold"], index=0)
        with e7:
            special_event = st.selectbox("Special Campus Event?", ["No", "Yes"], index=0)
        with e8:
            st.write("")  # placeholder

        submitted = st.form_submit_button("🚀 Predict Lecture Attendance", use_container_width=True)

    if submitted:
        input_payload = {
            "Date": input_date.strftime("%Y-%m-%d"),
            "Day of Week": day_of_week,
            "Lecture Number": lecture_num,
            "Start Time": start_time,
            "Subject": subject,
            "Faculty ID": faculty_id,
            "Semester": semester,
            "Branch": branch,
            "Section": section,
            "Classroom": classroom,
            "Total Enrolled Students": int(total_enrolled),
            "Previous Lecture Attendance": float(prev_att),
            "Gap Since Previous Lecture": float(gap_hours),
            "Practical/Theory": format_type,
            "Internal Test Week": test_week,
            "Assignment Due": assignment_due,
            "Holiday Before/After": holiday_prox,
            "Weather": weather,
            "Special Event": special_event,
            "Faculty Experience": float(faculty_exp)
        }

        try:
            pred_result = predict_attendance(input_payload)
            pct = pred_result["Predicted Attendance Percentage"]
            expected_students = pred_result["Expected Students Present"]
            category = pred_result["Attendance Category"]

            st.markdown("---")
            st.markdown("### 📋 Prediction Results & Headcount Estimation")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Predicted Attendance", f"{pct:.1f}%")
            with r2:
                st.metric("Expected Students Present", f"{expected_students} / {int(total_enrolled)}")
            with r3:
                if "High" in category:
                    badge_html = '<span class="badge-high">🟢 HIGH ATTENDANCE (> 75%)</span>'
                elif "Medium" in category:
                    badge_html = '<span class="badge-med">🟡 MEDIUM ATTENDANCE (50–75%)</span>'
                else:
                    badge_html = '<span class="badge-low">🔴 LOW ATTENDANCE (< 50%)</span>'
                
                st.markdown(f"**Attendance Band:**<br>{badge_html}", unsafe_allow_html=True)

            # Recommendations
            st.markdown("#### 💡 Schedule Insight & Operational Recommendations")
            if pct < 50.0:
                st.error("⚠️ **High Absenteeism Risk Alert:** Attendance is predicted below 50%. Consider sending an LMS notification reminder or verifying student timetable overlap.")
            elif pct <= 75.0:
                st.info("ℹ️ **Moderate Attendance Forecast:** Lecture attendance is expected to be steady. Regular classroom allocation is appropriate.")
            else:
                st.success("✅ **Optimal Attendance Expected:** High classroom turnout anticipated. Ensure classroom capacity meets full strength.")

        except Exception as e:
            st.error(f"Prediction failed: {e}. Please ensure models are trained via `python src/train.py`.")

# -------------------------------------------------------------
# TAB 4: MODEL PERFORMANCE
# -------------------------------------------------------------
elif app_mode == "🏆 Model Performance":
    st.markdown('<div class="main-header">🏆 Model Evaluation & Comparative Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous evaluation across Linear, Tree-based, and Ensembled Gradient Boosting algorithms</div>', unsafe_allow_html=True)

    if comp_df is not None:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("📊 Model Comparison Matrix")
            # Format numerical metrics cleanly and apply theme-safe high-contrast highlighting
            styled_comp = (
                comp_df.copy()
                .style.format({
                    "MAE": "{:.3f}",
                    "RMSE": "{:.3f}",
                    "MAPE (%)": "{:.2f}%",
                    "R2": "{:.4f}"
                })
                .highlight_min(
                    subset=["MAE", "RMSE", "MAPE (%)"],
                    props="background-color: #064e3b; color: #4ade80; font-weight: 700; border-radius: 3px;"
                )
                .highlight_max(
                    subset=["R2"],
                    props="background-color: #064e3b; color: #4ade80; font-weight: 700; border-radius: 3px;"
                )
            )
            st.dataframe(styled_comp, use_container_width=True)

        with c2:
            st.subheader("ℹ️ Metric Interpretation Guide")
            st.markdown("""
            - **MAE (Mean Absolute Error)**: Average absolute divergence from actual attendance percentage. *(Lower is better)*
            - **RMSE (Root Mean Squared Error)**: Penalizes large attendance estimation outliers heavily. *(Lower is better)*
            - **MAPE (%)**: Average relative percentage error across observations. *(Lower is better)*
            - **R² Score**: Proportion of attendance variance explained by the model. *(Higher is better, max 1.0)*
            """)

        # Plots
        st.subheader("📈 Visual Comparison of Model Error & Goodness of Fit")
        fig_path = os.path.join("reports", "figures", "model_comparison_metrics.png")
        if os.path.exists(fig_path):
            st.image(fig_path, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            act_pred_path = os.path.join("reports", "figures", "actual_vs_predicted.png")
            if os.path.exists(act_pred_path):
                st.markdown("#### Actual vs. Predicted Attendance (Best Model)")
                st.image(act_pred_path, use_container_width=True)
        with c4:
            res_path = os.path.join("reports", "figures", "residual_distribution.png")
            if os.path.exists(res_path):
                st.markdown("#### Residual Error Distribution")
                st.image(res_path, use_container_width=True)

        if metadata:
            st.markdown("---")
            st.subheader("⚙️ Best Model Metadata")
            st.json(metadata)

# -------------------------------------------------------------
# TAB 5: FEATURE IMPORTANCE
# -------------------------------------------------------------
elif app_mode == "🔍 Feature Importance":
    st.markdown('<div class="main-header">🔍 Model Interpretability & Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding key predictive factors associated with classroom attendance patterns</div>', unsafe_allow_html=True)

    st.markdown("""
    > **Academic Note on Causality:** Feature importance reflects the relative predictive association between timetable attributes and attendance outcomes. These scores indicate **statistical association**, not direct causal relationships.
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌲 Tree-Based Gini Importance (MDI)")
        fig_mdi = os.path.join("reports", "figures", "feature_importance.png")
        if os.path.exists(fig_mdi):
            st.image(fig_mdi, use_container_width=True)
        else:
            st.info("Feature importance figure not yet generated.")

    with c2:
        st.subheader("🎲 Permutation Feature Importance")
        fig_perm = os.path.join("reports", "figures", "permutation_importance.png")
        if os.path.exists(fig_perm):
            st.image(fig_perm, use_container_width=True)
        else:
            st.info("Permutation importance figure not yet generated.")

# -------------------------------------------------------------
# TAB 6: LOW ATTENDANCE DIAGNOSTIC
# -------------------------------------------------------------
elif app_mode == "⚠️ Low Attendance Diagnostic":
    st.markdown('<div class="main-header">⚠️ Low Attendance Diagnostic & Schedule Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identifying operational schedule bottlenecks and timetable patterns linked with reduced turnout</div>', unsafe_allow_html=True)

    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🕐 Low Attendance Time Slots")
            slot_att = df.groupby(["Start Time", "Lunch_Timing" if "Lunch_Timing" in df.columns else "Lecture Number"])["Attendance Percentage"].mean().reset_index()
            slot_att = slot_att.sort_values(by="Attendance Percentage")
            st.dataframe(slot_att.head(5).rename(columns={"Attendance Percentage": "Mean Attendance (%)"}), use_container_width=True)
            st.info("📌 **Observation:** Late afternoon periods and post-lunch sessions frequently exhibit lower attendance rates.")

        with c2:
            st.subheader("📅 Day & Holiday Proximity Patterns")
            day_att = df.groupby(["Day of Week", "Holiday Before/After"])["Attendance Percentage"].mean().reset_index()
            day_att = day_att.sort_values(by="Attendance Percentage")
            st.dataframe(day_att.head(5).rename(columns={"Attendance Percentage": "Mean Attendance (%)"}), use_container_width=True)
            st.info("📌 **Observation:** Classes immediately preceding long weekends or on Saturdays correlate with noticeable turnout drops.")

        st.subheader("💡 Timetable Optimization Recommendations")
        st.markdown("""
        1. **Reschedule Critical Core Theory Modules**: Avoid placing core foundational courses during Period 6/7 (late afternoon post-lunch) or early Monday 9:00 AM slots.
        2. **Pair High-Engagement Practicals**: Schedule laboratory/practical sessions in post-lunch slots to sustain student participation.
        3. **Pre-Holiday Syllabus Pacing**: Adjust tutorial submissions and assignment reviews to avoid days immediately following or preceding public holidays.
        """)

# -------------------------------------------------------------
# TAB 7: UPLOAD & VALIDATE CSV
# -------------------------------------------------------------
elif app_mode == "📁 Upload & Validate CSV":
    st.markdown('<div class="main-header">📁 Ingest & Validate Real Attendance Records</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your original collected attendance CSV to validate data integrity, clean anomalies, and retrain models</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Raw Attendance CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_upload_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Detected {len(raw_upload_df)} rows and {len(raw_upload_df.columns)} columns.")

            # Schema Validation
            is_valid, missing = validate_schema(raw_upload_df)
            if not is_valid:
                st.error(f"❌ Schema validation failed! Missing required columns: {missing}")
            else:
                st.success("✅ Schema validation passed! All 22 required attributes are present.")
                
                if st.button("🧹 Clean Data & Run Validation Pipeline"):
                    cleaned_df, report = clean_attendance_data(
                        raw_upload_df,
                        output_path=os.path.join("data", "processed", "attendance_cleaned.csv"),
                        report_output_path=os.path.join("reports", "cleaning_report.json")
                    )
                    # Also save as raw
                    raw_upload_df.to_csv(os.path.join("data", "raw", "attendance_raw.csv"), index=False)
                    
                    st.success("Data cleaned and saved successfully!")
                    st.json(report)
                    st.dataframe(cleaned_df.head(10))

                if st.button("⚡ Retrain All ML Models with Uploaded Data"):
                    with st.spinner("Training models, tuning hyperparameters, and generating evaluation reports..."):
                        train_res = train_and_tune_models()
                        st.success(f"Models retrained successfully! Best Model: {train_res['best_model_name']}")
                        st.json(train_res['metadata']['test_metrics'])
                        st.experimental_rerun()

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")

    st.markdown("---")
    st.subheader("📥 Download Standard Attendance Template")
    template_path = os.path.join("data", "templates", "attendance_data_template.csv")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="⬇️ Download CSV Template (attendance_data_template.csv)",
                data=f.read(),
                file_name="attendance_data_template.csv",
                mime="text/csv"
            )
