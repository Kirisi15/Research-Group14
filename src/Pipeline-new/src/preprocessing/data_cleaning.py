
import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "output" / "final_dataset.csv"

OUTPUT_FILE = BASE_DIR / "data" / "output" / "cleaned_dataset.csv"

REPORT_FILE = BASE_DIR / "data" / "output" / "data_cleaning_report.txt"


# --------------------------------------------------
# 2. Columns to remove
# --------------------------------------------------

UNWANTED_COLUMNS = [
    "actual_rows",
    "parallel_aware",
    "collected_by"
]

TARGET_COLUMN = "execution_time"


# --------------------------------------------------
# 3. Required columns
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "query_id",
    "execution_time",
    "join_count",
    "where_count",
    "group_by_count",
    "order_by_count",
    "subquery_count",
    "query_length",
    "aggregate_count",
    "distinct_count",
    "condition_count",
    "table_count",
    "seq_scan_count",
    "index_scan_count",
    "hash_join_count",
    "nested_loop_count",
    "merge_join_count",
    "sort_count",
    "aggregate_node_count",
    "total_nodes",
    "plan_rows"
]


# --------------------------------------------------
# 4. Load dataset
# --------------------------------------------------

def load_dataset():

    print("\nLoading dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("Dataset loaded successfully!")
    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")

    return df


# --------------------------------------------------
# 5. Validate dataset structure
# --------------------------------------------------

def validate_columns(df):

    print("\nValidating dataset columns...")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    print("Required columns are available.")


# --------------------------------------------------
# 6. Remove unwanted columns
# --------------------------------------------------

def remove_unwanted_columns(df):

    print("\nRemoving unwanted columns...")

    existing_columns = [
        column for column in UNWANTED_COLUMNS
        if column in df.columns
    ]

    df = df.drop(columns=existing_columns)

    print(f"Removed columns: {existing_columns}")

    return df


# --------------------------------------------------
# 7. Check missing values
# --------------------------------------------------

def check_missing_values(df):

    print("\nChecking missing values...")

    missing_values = df.isnull().sum()

    missing_values = missing_values[
        missing_values > 0
    ]

    if missing_values.empty:
        print("No missing values found.")
    else:
        print("Missing values detected:")
        print(missing_values)

    return missing_values


# --------------------------------------------------
# 8. Check infinite values
# --------------------------------------------------

def check_infinite_values(df):

    print("\nChecking infinite values...")

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    infinite_counts = np.isinf(numeric_df).sum()

    infinite_counts = infinite_counts[
        infinite_counts > 0
    ]

    if infinite_counts.empty:
        print("No infinite values found.")
    else:
        print("Infinite values detected:")
        print(infinite_counts)

    return infinite_counts


# --------------------------------------------------
# 9. Check duplicate observations
# --------------------------------------------------

def check_duplicates(df):

    print("\nChecking duplicate observations...")

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")

    return duplicate_count


# --------------------------------------------------
# 10. Validate execution time
# --------------------------------------------------

def validate_execution_time(df):

    print("\nValidating execution time...")

    negative_count = (
        df[TARGET_COLUMN] < 0
    ).sum()

    zero_count = (
        df[TARGET_COLUMN] == 0
    ).sum()

    print(f"Negative execution times: {negative_count}")
    print(f"Zero execution times: {zero_count}")

    return negative_count, zero_count


# --------------------------------------------------
# 11. Main cleaning pipeline
# --------------------------------------------------

def run_cleaning_pipeline():

    print("=" * 55)
    print("SQL QUERY DATA CLEANING PIPELINE")
    print("=" * 55)

    df = load_dataset()

    original_rows = len(df)
    original_columns = len(df.columns)

    validate_columns(df)

    missing_values = check_missing_values(df)

    infinite_values = check_infinite_values(df)

    duplicate_count = check_duplicates(df)

    negative_count, zero_count = (
        validate_execution_time(df)
    )

    df = remove_unwanted_columns(df)

    # Do not automatically delete outliers or rows
    # with missing/infinite values in this first version.

    # Save cleaned dataset
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    # Generate cleaning report
    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        report.write("SQL QUERY DATA CLEANING REPORT\n")
        report.write("=" * 45 + "\n\n")

        report.write(f"Original rows: {original_rows}\n")
        report.write(f"Original columns: {original_columns}\n")

        report.write(f"Final rows: {len(df)}\n")
        report.write(f"Final columns: {len(df.columns)}\n")

        report.write(
            f"Removed columns: {UNWANTED_COLUMNS}\n"
        )

        report.write(
            f"Duplicate rows: {duplicate_count}\n"
        )

        report.write(
            f"Negative execution times: {negative_count}\n"
        )

        report.write(
            f"Zero execution times: {zero_count}\n"
        )

        report.write(
            f"Columns with missing values: "
            f"{missing_values.to_dict()}\n"
        )

        report.write(
            f"Columns with infinite values: "
            f"{infinite_values.to_dict()}\n"
        )

        report.write(
            "\nNo rows were automatically removed.\n"
        )

        report.write(
            "No outliers were automatically removed.\n"
        )

    print("\nCleaning completed successfully!")

    print(f"\nFinal rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")

    print(f"\nCleaned dataset: {OUTPUT_FILE}")
    print(f"Cleaning report: {REPORT_FILE}")


# --------------------------------------------------
# 12. Execute pipeline
# --------------------------------------------------

if __name__ == "__main__":
    run_cleaning_pipeline()
