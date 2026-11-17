
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "output" / "cleaned_dataset.csv"

def analyse_features():
    df = pd.read_csv(INPUT_FILE)
    inspect_incomplete_queries(df)
    analyse_query_grouping(df)

    print("=" * 65)
    print("FEATURE QUALITY ANALYSIS")
    print("=" * 65)

    print(f"\nDataset shape: {df.shape}")

    print("\n1. UNIQUE VALUE COUNT")
    print("-" * 65)

    for column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")

    print("\n2. CONSTANT FEATURES")
    print("-" * 65)

    constant_features = [
        column for column in df.columns
        if df[column].nunique() <= 1
    ]

    if constant_features:
        print(constant_features)
    else:
        print("No constant features found.")

    print("\n3. NUMERICAL FEATURE SUMMARY")
    print("-" * 65)
    print(df.describe().T.to_string())

    print("\n4. CATEGORICAL FEATURE DISTRIBUTION")
    print("-" * 65)

    for column in df.select_dtypes(include=["object"]).columns:
        print(f"\n{column}")
        print(df[column].value_counts().head(10).to_string())

    print("\nFeature analysis completed.")


def analyse_query_grouping(df):
    print("\n5. QUERY AND TEMPLATE RELATIONSHIP")
    print("-" * 65)

    query_to_template = df.groupby("query_id")["template_id"].nunique()
    template_to_query = df.groupby("template_id")["query_id"].nunique()

    print(
        "Query IDs mapped to more than one template:",
        (query_to_template > 1).sum()
    )

    print(
        "Template IDs mapped to more than one query:",
        (template_to_query > 1).sum()
    )

    print("\nObservations per query ID:")
    print(df.groupby("query_id").size().value_counts().sort_index())

    print("\nObservations per template ID:")
    print(df.groupby("template_id").size().value_counts().sort_index())


def inspect_incomplete_queries(df):
    print("\n6. INCOMPLETE QUERY INVESTIGATION")
    print("-" * 65)

    query_counts = df.groupby("query_id").size()
    incomplete_ids = query_counts[query_counts < 12].index

    for query_id in incomplete_ids:
        query_data = df[df["query_id"] == query_id]

        print(f"\nQuery ID: {query_id}")
        print(f"Template ID: {query_data['template_id'].iloc[0]}")
        print(f"Observations: {len(query_data)}")

        print("\nRecords by machine:")
        print(
            query_data.groupby("machine_name")["run_number"]
            .agg(["count", "unique"])
            .to_string()
        )

        print("\nAvailable runs:")
        print(
            query_data[["machine_name", "run_number"]]
            .sort_values(["machine_name", "run_number"])
            .to_string(index=False)
        )

if __name__ == "__main__":
    analyse_features()
