import pandas as pd

df = pd.read_csv(
    "data/raw/college_student_placement_dataset.csv"
)

print("\nDUPLICATES")
print(df.duplicated().sum())

print("\nPLACEMENT VALUES")
print(df["Placement"].value_counts())

print("\nINTERNSHIP VALUES")
print(df["Internship_Experience"].value_counts())

print("\nDATA TYPES")
print(df.dtypes)