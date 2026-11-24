
import pandas as pd
from sklearn.model_selection import GroupKFold

# Load cleaned dataset
DATA_PATH = "data/output/cleaned_dataset.csv"

df = pd.read_csv(DATA_PATH)

y = df["execution_time"]
groups = df["query_id"]

# Create five query-grouped folds
cv = GroupKFold(n_splits=5)

print("=" * 65)
print("EXECUTION TIME DISTRIBUTION ACROSS CROSS-VALIDATION FOLDS")
print("=" * 65)

for fold_number, (_, test_idx) in enumerate(
    cv.split(df, y, groups=groups),
    start=1
):
    fold_times = y.iloc[test_idx]

    print(f"\nFOLD {fold_number}")
    print("-" * 40)

    print(
        fold_times.describe(
            percentiles=[0.50, 0.90, 0.95, 0.99]
        ).round(3).to_string()
    )

    print(f"Skewness: {fold_times.skew():.3f}")
    print(
        "Unique queries:",
        df.iloc[test_idx]["query_id"].nunique()
    )

print("\nAnalysis completed.")
