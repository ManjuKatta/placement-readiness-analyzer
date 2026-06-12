import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv(
    "data/raw/college_student_placement_dataset.csv"
)

# Create figures folder if it doesn't exist
os.makedirs("reports/figures", exist_ok=True)

# --------------------------------------------------
# Placement vs Internship Experience
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    x="Internship_Experience",
    hue="Placement",
    data=df
)

plt.title("Placement vs Internship Experience")

plt.savefig(
    "reports/figures/internship_vs_placement.png"
)

plt.close()

# --------------------------------------------------
# CGPA vs Placement
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Placement",
    y="CGPA",
    data=df
)

plt.title("CGPA vs Placement")

plt.savefig(
    "reports/figures/cgpa_vs_placement.png"
)

plt.close()

# --------------------------------------------------
# Academic Performance vs Placement
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Placement",
    y="Academic_Performance",
    data=df
)

plt.title("Academic Performance vs Placement")

plt.savefig(
    "reports/figures/academic_performance_vs_placement.png"
)

plt.close()

# --------------------------------------------------
# Communication Skills vs Placement
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Placement",
    y="Communication_Skills",
    data=df
)

plt.title("Communication Skills vs Placement")

plt.savefig(
    "reports/figures/communication_skills_vs_placement.png"
)

plt.close()

# --------------------------------------------------
# Projects Completed vs Placement
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Placement",
    y="Projects_Completed",
    data=df
)

plt.title("Projects Completed vs Placement")

plt.savefig(
    "reports/figures/projects_completed_vs_placement.png"
)

plt.close()

# --------------------------------------------------
# IQ vs Placement
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Placement",
    y="IQ",
    data=df
)

plt.title("IQ vs Placement")

plt.savefig(
    "reports/figures/iq_vs_placement.png"
)

plt.close()

# --------------------------------------------------
# Previous Semester Result vs Placement
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Placement",
    y="Prev_Sem_Result",
    data=df
)

plt.title("Previous Semester Result vs Placement")

plt.savefig(
    "reports/figures/prev_sem_result_vs_placement.png"
)

plt.close()

# --------------------------------------------------
# Extra Curricular Score vs Placement
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Placement",
    y="Extra_Curricular_Score",
    data=df
)

plt.title("Extra Curricular Score vs Placement")

plt.savefig(
    "reports/figures/extra_curricular_vs_placement.png"
)

plt.close()

print("=" * 50)
print("ADVANCED EDA COMPLETED SUCCESSFULLY")
print("=" * 50)
print("Charts saved in reports/figures")