
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "output" / "final_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE DISTRIBUTION ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# NUMERIC FEATURE VALIDATION
# ============================================================

numeric_df = df.select_dtypes(include=[np.number])

print("\n" + "=" * 70)
print("NUMERIC FEATURE VALIDATION")
print("=" * 70)

print(f"Numeric columns: {len(numeric_df.columns)}")

print("\nMissing values:")
print(numeric_df.isna().sum().to_string())

print("\nInfinite values:")
print(
    pd.Series(
        np.isinf(numeric_df.to_numpy()).sum(axis=0),
        index=numeric_df.columns
    ).to_string()
)


# ============================================================
# CONSTANT AND NEAR-CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CONSTANT AND LOW-VARIANCE FEATURES")
print("=" * 70)

feature_columns = [
    column for column in numeric_df.columns
    if column != "execution_time"
]

constant_features = [
    column for column in feature_columns
    if numeric_df[column].nunique(dropna=False) <= 1
]

print("\nConstant predictor features:")
if constant_features:
    for column in constant_features:
        print(f"- {column}: {numeric_df[column].nunique()} unique value(s)")
else:
    print("None found.")

print("\nUnique values per numeric feature:")
print(
    numeric_df.nunique(dropna=False)
    .sort_values()
    .to_string()
)


# ============================================================
# NUMERIC SUMMARY AND SKEWNESS
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE SUMMARY")
print("=" * 70)

summary = numeric_df.describe().T

summary["skewness"] = numeric_df.skew(numeric_only=True)

summary = summary.round(4)

print(summary.to_string())

summary_file = OUTPUT_DIR / "feature_summary.csv"
summary.to_csv(summary_file)

print(f"\nFeature summary saved to: {summary_file}")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ANALYSIS COMPLETED")
print("=" * 70)