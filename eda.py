"""
Quick EDA script for the capstone.
Run with: python src/eda.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "machine_failure_data.csv")
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nClass distribution:\n", df["failure"].value_counts())
print("\nSummary:\n", df.describe().round(2))

fig = df["failure"].value_counts().sort_index().plot(kind="bar")
fig.set_xlabel("Failure class (0=lower risk, 1=higher risk)")
fig.set_ylabel("Number of observations")
fig.figure.tight_layout()
fig.figure.savefig(OUT / "class_distribution.png", dpi=150)
plt.close(fig.figure)
