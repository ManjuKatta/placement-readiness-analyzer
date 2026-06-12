import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(
    "data/raw/college_student_placement_dataset.csv"
)

# Create figure folder automatically
import os

os.makedirs("reports/figures", exist_ok=True)

# -----------------------------------
# Placement Distribution
# -----------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x="Placement",
    data=df
)

plt.title("Placement Distribution")

plt.savefig(
    "reports/figures/placement_distribution.png"
)

plt.show()

# -----------------------------------
# CGPA Distribution
# -----------------------------------

plt.figure(figsize=(6,4))

sns.histplot(
    df["CGPA"],
    bins=20,
    kde=True
)

plt.title("CGPA Distribution")

plt.savefig(
    "reports/figures/cgpa_distribution.png"
)

plt.show()

# -----------------------------------
# IQ Distribution
# -----------------------------------

plt.figure(figsize=(6,4))

sns.histplot(
    df["IQ"],
    bins=20,
    kde=True
)

plt.title("IQ Distribution")

plt.savefig(
    "reports/figures/iq_distribution.png"
)

plt.show()

# -----------------------------------
# Correlation Heatmap
# -----------------------------------

plt.figure(figsize=(10,6))

numeric_df = df.select_dtypes(include=["int64", "float64"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "reports/figures/correlation_heatmap.png"
)

plt.show()