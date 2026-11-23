
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "output" / "cleaned_dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "output" / "high_correlation_pairs.csv"

def analyse_high_correlations():

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

    correlation = numerical_df.drop(
        columns=["execution_time"]
    ).corr(method="pearson")

    pairs = []

    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):

            feature_1 = correlation.columns[i]
            feature_2 = correlation.columns[j]

            value = correlation.iloc[i, j]

            if abs(value) >= 0.85:

                pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": value
                })

    result = pd.DataFrame(pairs)

    if not result.empty:
        result["absolute_correlation"] = result["correlation"].abs()
        result = result.sort_values(
            "absolute_correlation",
            ascending=False
        )

    print("\nHIGHLY CORRELATED FEATURE PAIRS")
    print("=" * 70)

    if result.empty:
        print("No highly correlated pairs found.")
    else:
        print(result.round(4).to_string(index=False))

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nReport saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    analyse_high_correlations()
