import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/raw/college_student_placement_dataset.csv"
)

# ----------------------------------
# Convert categorical variables
# ----------------------------------

df["Internship_Experience"] = (
    df["Internship_Experience"]
    .map({
        "No": 0,
        "Yes": 1
    })
)

df["Placement"] = (
    df["Placement"]
    .map({
        "No": 0,
        "Yes": 1
    })
)

# ----------------------------------
# Technical Score
# ----------------------------------

df["Technical_Score"] = (
    (df["IQ"] * 0.30)
    +
    (df["Academic_Performance"] * 0.40)
    +
    (df["Projects_Completed"] * 10 * 0.30)
)

# ----------------------------------
# Readiness Score
# ----------------------------------

df["Readiness_Score"] = (
    (df["CGPA"] * 10 * 0.40)
    +
    (df["Communication_Skills"] * 0.30)
    +
    (df["Extra_Curricular_Score"] * 0.10)
    +
    (df["Internship_Experience"] * 20 * 0.20)
)

# ----------------------------------
# Profile Strength
# ----------------------------------

df["Profile_Strength"] = (
    df["Technical_Score"]
    +
    df["Readiness_Score"]
)

# ----------------------------------
# Remove ID column
# ----------------------------------

df.drop(
    columns=["College_ID"],
    inplace=True
)

# ----------------------------------
# Save processed dataset
# ----------------------------------

df.to_csv(
    "data/processed/featured_data.csv",
    index=False
)

print("=" * 50)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 50)

print("\nNEW COLUMNS:")
print(
    [
        "Technical_Score",
        "Readiness_Score",
        "Profile_Strength"
    ]
)

print("\nFINAL SHAPE:")
print(df.shape)

print("\nFIRST 5 ROWS:")
print(df.head())