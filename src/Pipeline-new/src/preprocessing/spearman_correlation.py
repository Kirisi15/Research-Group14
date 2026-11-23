
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "output" / "cleaned_dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "output" / "spearman_correlation.csv"
HIGH_CORR_FILE = BASE_DIR / "data" / "output" / "spearman_high_correlation_pairs.csv"


def analyse_spearman_correlation():

    df = pd.read_csv(INPUT_FILE)

    excluded_columns = [
        "machine_name",
        "timestamp",
        "query_id",
        "template_id",
        "run_number",
        "node_type"
    ]

    numerical_df = df.drop(columns=excluded_columns)

    correlation = numerical_df.corr(method="spearman")

    # Save complete correlation matrix
    correlation.to_csv(OUTPUT_FILE)

    # Extract highly correlated feature pairs
    feature_correlation = correlation.drop(
        index=["execution_time"],
        columns=["execution_time"]
    )

    pairs = []

    for i in range(len(feature_correlation.columns)):
        for j in range(i + 1, len(feature_correlation.columns)):

            feature_1 = feature_correlation.columns[i]
            feature_2 = feature_correlation.columns[j]

            value = feature_correlation.iloc[i, j]

            if abs(value) >= 0.85:
                pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": value,
                    "absolute_correlation": abs(value)
                })

    result = pd.DataFrame(pairs)

    if not result.empty:
        result = result.sort_values(
            "absolute_correlation",
            ascending=False
        )

    print("\nSPEARMAN CORRELATION WITH EXECUTION TIME")
    print("=" * 65)

    target_corr = correlation["execution_time"].drop("execution_time")
    print(target_corr.sort_values(
        key=abs,
        ascending=False
    ).round(4).to_string())

    print("\nHIGHLY CORRELATED FEATURE PAIRS")
    print("=" * 65)

    if result.empty:
        print("No highly correlated pairs found.")
    else:
        print(result.round(4).to_string(index=False))

    correlation.to_csv(OUTPUT_FILE)
    result.to_csv(HIGH_CORR_FILE, index=False)

    print("\nFiles saved:")
    print(OUTPUT_FILE)
    print(HIGH_CORR_FILE)


if __name__ == "__main__":
    analyse_spearman_correlation()
