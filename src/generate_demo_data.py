"""
Demonstration Synthetic Dataset Generator
=========================================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

IMPORTANT NOTICE:
=================
This file generates a SYNTHETIC DEMONSTRATION DATASET used ONLY to verify the end-to-end
technical pipeline, unit tests, model training, and Streamlit dashboard before the student
inserts their organically collected physical classroom records.

DO NOT CLAIM THIS IS REAL COLLECTED DATA IN ACADEMIC SUBMISSIONS.
"""

import os
import random
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_attendance_dataset(
    num_days: int = 60,
    start_date_str: str = "2026-08-03",
    output_path: str = os.path.join("data", "raw", "attendance_raw.csv")
) -> pd.DataFrame:
    """
    Generates a realistic synthetic demonstration dataset reflecting real-world
    academic classroom attendance dynamics across multiple branches, subjects, and semesters.
    """
    np.random.seed(42)
    random.seed(42)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    # Define Academic Structure
    cohorts = [
        {"Branch": "CSE", "Semester": 5, "Section": "A", "Classroom": "Room 401", "Total_Enrolled": 60},
        {"Branch": "CSE", "Semester": 5, "Section": "B", "Classroom": "Room 402", "Total_Enrolled": 62},
        {"Branch": "IT", "Semester": 5, "Section": "A", "Classroom": "Room 301", "Total_Enrolled": 58},
        {"Branch": "CSE", "Semester": 7, "Section": "A", "Classroom": "Room 501", "Total_Enrolled": 55},
        {"Branch": "AI-DS", "Semester": 3, "Section": "A", "Classroom": "Room 201", "Total_Enrolled": 64},
    ]

    subjects_by_cohort = {
        ("CSE", 5): [
            ("Python Programming", "F_001", 8, "Theory"),
            ("Database Management Systems", "F_002", 12, "Theory"),
            ("Operating Systems", "F_003", 5, "Theory"),
            ("Computer Networks", "F_004", 7, "Theory"),
            ("Python Lab", "F_001", 8, "Practical"),
            ("DBMS Lab", "F_002", 12, "Practical")
        ],
        ("IT", 5): [
            ("Web Technologies", "F_005", 6, "Theory"),
            ("Database Management Systems", "F_002", 12, "Theory"),
            ("Software Engineering", "F_006", 10, "Theory"),
            ("Computer Networks", "F_004", 7, "Theory"),
            ("Web Tech Lab", "F_005", 6, "Practical")
        ],
        ("CSE", 7): [
            ("Machine Learning", "F_007", 14, "Theory"),
            ("Cloud Computing", "F_008", 9, "Theory"),
            ("Information Security", "F_009", 11, "Theory"),
            ("ML Project Lab", "F_007", 14, "Practical")
        ],
        ("AI-DS", 3): [
            ("Data Structures & Algorithms", "F_010", 6, "Theory"),
            ("Discrete Mathematics", "F_011", 15, "Theory"),
            ("Object Oriented Programming", "F_001", 8, "Theory"),
            ("DSA Lab", "F_010", 6, "Practical")
        ]
    }

    timetable_slots = [
        (1, "09:00"),
        (2, "10:00"),
        (3, "11:15"),
        (4, "12:15"),
        (5, "13:45"),  # After lunch
        (6, "14:45"),
        (7, "15:45")
    ]

    weather_choices = ["Sunny", "Sunny", "Sunny", "Cloudy", "Rainy"]
    
    records = []
    current_date = start_date
    days_generated = 0

    # Track historical subject attendance for rolling previous percentage
    last_known_attendance = {}

    while days_generated < num_days:
        # Skip Sundays
        if current_date.weekday() == 6:
            current_date += timedelta(days=1)
            continue

        day_of_week = current_date.strftime("%A")
        date_str = current_date.strftime("%d-%m-%Y")

        # Academic contextual conditions
        # Test week on weeks 4 and 8
        week_num = (days_generated // 6) + 1
        is_test_week = "Yes" if week_num in [4, 8] else "No"
        
        # Holiday proximity (Saturdays or special days)
        is_holiday_prox = "Yes" if current_date.weekday() in [4, 5] or days_generated in [15, 32, 45] else "No"
        
        # Assignment due on selected Wednesdays
        is_assignment_due = "Yes" if current_date.weekday() == 2 and week_num in [3, 5, 7] else "No"

        # Campus special event
        is_special_event = "Yes" if days_generated in [22, 48] else "No"

        # Daily weather
        weather = random.choice(weather_choices)

        for cohort in cohorts:
            cohort_key = (cohort["Branch"], cohort["Semester"])
            available_subs = subjects_by_cohort.get(cohort_key, subjects_by_cohort[("CSE", 5)])

            # Schedule 3 to 5 lectures per cohort per day
            daily_lecture_count = random.randint(3, 5)
            daily_slots = sorted(random.sample(timetable_slots, daily_lecture_count), key=lambda x: x[0])

            for slot_idx, (lec_num, start_time) in enumerate(daily_slots):
                sub_info = random.choice(available_subs)
                sub_name, faculty_id, fac_exp, format_type = sub_info
                
                enrolled = cohort["Total_Enrolled"]
                
                # Base attendance baseline influenced by realistic factors
                base_pct = 78.0

                # Day of week effect: Mondays & Fridays slightly lower, Midweek higher
                if day_of_week in ["Monday", "Friday"]:
                    base_pct -= random.uniform(3.0, 7.0)
                elif day_of_week in ["Tuesday", "Wednesday", "Thursday"]:
                    base_pct += random.uniform(2.0, 5.0)
                elif day_of_week == "Saturday":
                    base_pct -= random.uniform(8.0, 14.0)

                # Time of day / slot effect: 1st period and late periods slightly lower attendance
                if lec_num == 1:
                    base_pct -= random.uniform(2.0, 5.0)
                elif lec_num >= 5:  # Post lunch
                    base_pct -= random.uniform(4.0, 8.0)
                elif lec_num in [2, 3]:
                    base_pct += random.uniform(3.0, 6.0)

                # Practical sessions usually have higher attendance due to grading
                if format_type == "Practical":
                    base_pct += random.uniform(6.0, 10.0)

                # Test week increases attendance significantly
                if is_test_week == "Yes":
                    base_pct += random.uniform(8.0, 14.0)

                # Assignment due increases attendance
                if is_assignment_due == "Yes":
                    base_pct += random.uniform(4.0, 8.0)

                # Heavy Rain drops attendance
                if weather == "Rainy":
                    base_pct -= random.uniform(6.0, 12.0)

                # Special event / fest drops attendance
                if is_special_event == "Yes":
                    base_pct -= random.uniform(12.0, 20.0)

                # Faculty experience slight positive correlation
                base_pct += (fac_exp - 5) * 0.4

                # Add natural noise
                final_pct = np.clip(base_pct + np.random.normal(0, 3.5), 35.0, 98.0)
                
                # Calculate actual students present
                present_count = int(round((final_pct / 100.0) * enrolled))
                present_count = min(max(present_count, 15), enrolled)
                actual_pct = round((present_count / enrolled) * 100.0, 2)

                # Previous attendance retrieval
                history_key = (cohort["Branch"], cohort["Semester"], cohort["Section"], sub_name)
                prev_att = last_known_attendance.get(history_key, round(actual_pct + random.uniform(-4.0, 4.0), 2))
                prev_att = float(np.clip(prev_att, 40.0, 98.0))
                last_known_attendance[history_key] = actual_pct

                # Gap calculation
                gap_hours = random.choice([24.0, 48.0, 72.0]) if slot_idx == 0 else 1.0

                classroom = "Lab 1" if format_type == "Practical" else cohort["Classroom"]

                record = {
                    "Date": date_str,
                    "Day of Week": day_of_week,
                    "Lecture Number": lec_num,
                    "Start Time": start_time,
                    "Subject": sub_name,
                    "Faculty ID": faculty_id,
                    "Semester": cohort["Semester"],
                    "Branch": cohort["Branch"],
                    "Section": cohort["Section"],
                    "Classroom": classroom,
                    "Total Enrolled Students": enrolled,
                    "Students Present": present_count,
                    "Attendance Percentage": actual_pct,
                    "Previous Lecture Attendance": prev_att,
                    "Gap Since Previous Lecture": gap_hours,
                    "Practical/Theory": format_type,
                    "Internal Test Week": is_test_week,
                    "Assignment Due": is_assignment_due,
                    "Holiday Before/After": is_holiday_prox,
                    "Weather": weather,
                    "Special Event": is_special_event,
                    "Faculty Experience": fac_exp
                }
                records.append(record)

        days_generated += 1
        current_date += timedelta(days=1)

    df = pd.DataFrame(records)
    
    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Also save a dedicated copy as synthetic_demo_attendance.csv
    demo_copy_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "synthetic_demo_attendance.csv")
    df.to_csv(demo_copy_path, index=False)

    logger.info(f"Generated {len(df)} realistic demonstration records.")
    logger.info(f"Saved to: {output_path} and {demo_copy_path}")

    return df


if __name__ == "__main__":
    generate_synthetic_attendance_dataset()
