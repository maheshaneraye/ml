# Data Collection Guide & Protocol

## Capstone Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

---

### 1. Overview and Objective
As mandated by the capstone specification, this machine learning project is designed to ingest **original, organically collected attendance logs** from real academic lectures. 

This guide details the exact data collection protocol, privacy regulations, column schema, and formatting standards required for successful dataset construction.

---

### 2. Collection Parameters & Scope

| Parameter | Recommended Specification |
| :--- | :--- |
| **Observation Window** | Minimum 1 Week up to a full academic semester (continuous tracking) |
| **Target Volume** | 500 to 3,000 lecture-level records across cohorts |
| **Cohort Diversity** | Multiple branches (e.g., CSE, IT, ECE, Mechanical, AI-DS), semesters (e.g., Sem 3, 5, 7), and divisions/sections (A, B, C) |
| **Daily Intensity** | 3 to 8 lectures logged per day across tracked classrooms |
| **Primary Sources** | Physical faculty attendance registers, ERP/LMS logs, manual classroom head counts, departmental timetable sheets, academic calendar |

---

### 3. Data Privacy & Ethical Compliance
1. **Strict Anonymization**: Individual student names, university roll numbers, registration IDs, or personal contact info **MUST NOT** be recorded.
2. **Faculty Anonymization**: Mask faculty names with standardized codes: `F_001`, `F_002`, `F_003`, etc.
3. **Institutional Clearance**: Ensure department permission before transcribing attendance records from official departmental registers.
4. **Lecture-Level Aggregation**: Each row must represent **one complete lecture/session**, not individual student records.

---

### 4. Data Dictionary & Accepted Value Ranges

| Column Name | Data Type | Expected Format / Domain | Description & Rules |
| :--- | :--- | :--- | :--- |
| `Date` | Date String | `DD-MM-YYYY` (e.g., `01-08-2026`) | Date on which the class was held. |
| `Day of Week` | String | `Monday`, `Tuesday`, `Wednesday`, `Thursday`, `Friday`, `Saturday` | Day of the week corresponding to the lecture date. |
| `Lecture Number` | Integer | `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8` | Period / slot sequence in the daily timetable. |
| `Start Time` | String / Time | `09:00`, `10:00`, `11:15`, `12:15`, `13:45`, `14:45`, `15:45` | 24-hour timestamp of lecture commencement. |
| `Subject` | String | e.g. `Python`, `DBMS`, `Operating Systems`, `Computer Networks`, `Machine Learning` | Course title or code. |
| `Faculty ID` | String | `F_001`, `F_002`, `F_003`, etc. | Masked instructor identifier. |
| `Semester` | Integer | `1` to `8` (e.g., `3`, `5`, `7`) | Academic semester of the class. |
| `Branch` | String | `CSE`, `IT`, `ECE`, `Mechanical`, `Civil`, `AI-DS` | Academic discipline/department. |
| `Section` | String | `A`, `B`, `C` | Division or cohort section. |
| `Classroom` | String | `Room 401`, `Room 402`, `Lab 1`, `Lab 2`, `Auditorium` | Physical room or lab where session was held. |
| `Total Enrolled Students` | Integer | Positive integer (e.g., `60`, `65`, `72`) | Total official student strength of the cohort. |
| `Students Present` | Integer | `0` to `Total Enrolled Students` | Count of physically present students. |
| `Attendance Percentage` | Float | `0.0` to `100.0` | Calculated as: `(Students Present / Total Enrolled) * 100`. |
| `Previous Lecture Attendance` | Float | `0.0` to `100.0` | Attendance percentage of the immediate prior lecture for this subject/cohort. |
| `Gap Since Previous Lecture` | Float | In hours or days (e.g., `1.0`, `24.0`, `48.0`) | Elapsed time since the previous class in this course. |
| `Practical/Theory` | String | `Theory` or `Practical` | Delivery format (classroom lecture vs laboratory). |
| `Internal Test Week` | String | `Yes` or `No` | `Yes` if lecture occurred within an exam/midterm test week. |
| `Assignment Due` | String | `Yes` or `No` | `Yes` if an assignment submission deadline coincided with this lecture. |
| `Holiday Before/After` | String | `Yes` or `No` | `Yes` if class was immediately preceded or succeeded by a public holiday / long weekend. |
| `Weather` | String | `Sunny`, `Rainy`, `Cloudy`, `Cold` | Prevailing weather condition during class time. |
| `Special Event` | String | `Yes` or `No` | `Yes` if cultural fest, sports day, technical symposium, or campus drive was scheduled. |
| `Faculty Experience` | Integer / Float | Integer in years (e.g., `2`, `5`, `12`) | Total teaching experience of the instructor. |

---

### 5. Step-by-Step Data Entry Workflow

1. Open `data/templates/attendance_data_template.csv` in Microsoft Excel, Google Sheets, or LibreOffice Calc.
2. For each completed lecture, record the schedule attributes (`Date`, `Day of Week`, `Lecture Number`, `Start Time`, `Subject`, `Faculty ID`, `Semester`, `Branch`, `Section`, `Classroom`).
3. Count and enter `Total Enrolled Students` and `Students Present`.
4. Enter contextual indicators (`Internal Test Week`, `Assignment Due`, `Holiday Before/After`, `Weather`, `Special Event`).
5. Save the file as a UTF-8 CSV named `attendance_raw.csv` and place it in the `data/raw/` directory.
6. Execute the validation pipeline:
   ```bash
   python src/data_cleaning.py
   ```

---

### 6. Common Data Pitfalls & How to Avoid Them
- **Target Leakage**: Do not enter future lecture information or end-of-semester summary statistics into predictor columns.
- **Inconsistent Capitalization**: Use standard casing (`Theory`, `Practical`, `Yes`, `No`, `Sunny`, `Rainy`, `Cloudy`).
- **Logical Inconsistencies**: Ensure `Students Present <= Total Enrolled Students`.
- **Date Inconsistencies**: Use strict `DD-MM-YYYY` format (e.g., `15-09-2026`).
