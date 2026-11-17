import pandas as pd
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "output" / "final_dataset.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print(f"\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found at:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for number, column in enumerate(df.columns, start=1):
    print(f"{number:2}. {column}")


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

print(f"\nTotal missing values: {missing.sum()}")


# ============================================================
# DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")


# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

print(f"Unique query_id    : {df['query_id'].nunique()}")
print(f"Unique template_id : {df['template_id'].nunique()}")
print(f"Unique machines    : {df['machine_name'].nunique()}")
print(f"Unique collectors  : {df['collected_by'].nunique()}")
print(f"Unique runs        : {df['run_number'].nunique()}")


# ============================================================
# MACHINE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("MACHINE DISTRIBUTION")
print("=" * 70)

print(df["machine_name"].value_counts())


# ============================================================
# RUN DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RUN DISTRIBUTION")
print("=" * 70)

print(df["run_number"].value_counts().sort_index())


# ============================================================
# QUERY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("QUERY OBSERVATION DISTRIBUTION")
print("=" * 70)

query_counts = df.groupby("query_id").size()

print(f"Number of query variants : {len(query_counts)}")
print(f"Minimum observations     : {query_counts.min()}")
print(f"Maximum observations     : {query_counts.max()}")
print(f"Average observations     : {query_counts.mean():.2f}")

print("\nObservations per query:")
print(query_counts.value_counts().sort_index())


# ============================================================
# EXECUTION TIME
# ============================================================

print("\n" + "=" * 70)
print("EXECUTION TIME STATISTICS")
print("=" * 70)

print(df["execution_time"].describe())

print(f"\nMean     : {df['execution_time'].mean():.4f} ms")
print(f"Median   : {df['execution_time'].median():.4f} ms")
print(f"Std Dev  : {df['execution_time'].std():.4f} ms")
print(f"Skewness : {df['execution_time'].skew():.4f} ")


# ============================================================
# NUMERIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE SUMMARY")
print("=" * 70)

print(df.describe().T)


# ============================================================
# CHECK QUERY EXECUTION COMPLETENESS
# ============================================================

print("\n" + "=" * 70)
print("QUERY EXECUTION COMPLETENESS")
print("=" * 70)

expected_machines = df["machine_name"].nunique()
expected_runs = df["run_number"].nunique()

expected_observations = expected_machines * expected_runs

print(f"\nExpected observations per query:")
print(f"{expected_machines} machines × {expected_runs} runs")
print(f"= {expected_observations} observations")

query_counts = df.groupby("query_id").size()

incomplete_queries = query_counts[
    query_counts < expected_observations
]

if incomplete_queries.empty:

    print("\nAll queries have the expected number of observations.")

else:

    print(
        f"\nIncomplete queries: "
        f"{len(incomplete_queries)}"
    )

    print("\nObservation counts:")

    print(incomplete_queries)

    print("\nMissing machine/run combinations:")

    for query_id in incomplete_queries.index:

        query_data = df[
            df["query_id"] == query_id
        ]

        existing_combinations = set(
            zip(
                query_data["machine_name"],
                query_data["run_number"]
            )
        )

        all_combinations = {
            (machine, run)
            for machine in df["machine_name"].unique()
            for run in df["run_number"].unique()
        }

        missing_combinations = (
            all_combinations - existing_combinations
        )

        print(f"\nQuery: {query_id}")

        for machine, run in sorted(missing_combinations):

            print(
                f"  Missing → Machine: {machine}, "
                f"Run: {run}"
            )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DATASET OVERVIEW COMPLETED")
print("=" * 70)