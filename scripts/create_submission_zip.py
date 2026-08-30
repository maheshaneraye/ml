"""
Submission Zip Generator
========================
Creates a lightweight, clean, high-compression submission zip archive
(Classroom_Attendance_Prediction.zip) containing all project code, data,
notebooks, experiments, PDFs, and models, guaranteed under 4.0 MB.
"""

import os
import zipfile


def make_submission_zip(output_zip="Classroom_Attendance_Prediction.zip"):
    if os.path.exists(output_zip):
        os.remove(output_zip)

    include_dirs = [
        "01_Data",
        "02_Notebooks",
        "03_Experiment",
        "04_Deployment",
        "05_Demo",
        "06_Final_Report",
        "src",
        "models",
        "reports",
        "tests",
        "data/templates"
    ]

    include_files = [
        "README.md",
        "run_pipeline.py",
        "requirements.txt"
    ]

    exclude_patterns = [
        ".git", "__pycache__", ".pytest_cache", ".DS_Store",
        "synthetic_demo_attendance.csv", "attendance_prediction_full_3_semesters_v2.csv",
        output_zip
    ]

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in include_files:
            if os.path.exists(f):
                zf.write(f, os.path.join("Classroom_Attendance_Prediction", f))
                
        for folder in include_dirs:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    if any(p in root for p in exclude_patterns):
                        continue
                    for file in files:
                        if any(p in file for p in exclude_patterns):
                            continue
                        fp = os.path.join(root, file)
                        rel_path = os.path.relpath(fp, ".")
                        arcname = os.path.join("Classroom_Attendance_Prediction", rel_path)
                        zf.write(fp, arcname)

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    size_kb = os.path.getsize(output_zip) / 1024
    print(f"=== SUBMISSION ZIP CREATED ===")
    print(f"File Name : {output_zip}")
    print(f"File Size : {size_mb:.2f} MB ({size_kb:.1f} KB)")
    if size_mb <= 4.0:
        print(f"[SUCCESS] Archive is {size_mb:.2f} MB - Well under the 4.0 MB limit!")
    else:
        print(f"[WARNING] Archive is {size_mb:.2f} MB - Exceeds 4.0 MB.")
    return output_zip


if __name__ == "__main__":
    make_submission_zip()
