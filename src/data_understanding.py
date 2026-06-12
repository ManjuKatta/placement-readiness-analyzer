import pandas as pd

df = pd.read_csv("data/raw/college_student_placement_dataset.csv")

print(df.head())
print(df.shape)
print(df.columns.tolist())
print(df.info())