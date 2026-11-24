import pandas as pd
from sklearn.model_selection import GroupKFold

DATA_PATH = "data/output/cleaned_dataset.csv"

df = pd.read_csv(DATA_PATH)

y = df["execution_time"]
groups = df["query_id"]

cv = GroupKFold(n_splits=5)

print("=" * 80)
print("SLOWEST QUERY GROUPS IN FOLDS 3 AND 5")
print("=" * 80)

for fold_number, (_, test_idx) in enumerate(
    cv.split(df, y, groups=groups),
    start=1
):
    if fold_number not in [3, 5]:
        continue

    fold_df = df.iloc[test_idx].copy()

    slow_queries = (
        fold_df.groupby("query_id")
        .agg(
            execution_time_mean=("execution_time", "mean"),
            execution_time_max=("execution_time", "max"),
            execution_time_median=("execution_time", "median"),
            number_of_runs=("execution_time", "count"),
            machine_count=("machine_name", "nunique")
        )
        .sort_values("execution_time_max", ascending=False)
        .head(10)
    )

    print(f"\nFOLD {fold_number}")
    print("-" * 80)
    print(slow_queries.round(3).to_string())

print("\nAnalysis completed.")