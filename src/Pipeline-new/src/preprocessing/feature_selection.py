
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "output" / "cleaned_dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "output" / "feature_correlation.csv"

def analyse_correlations():

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

    correlation = numerical_df.corr(method="pearson")

    target_correlation = (
        correlation["execution_time"]
        .drop("execution_time")
        .sort_values(key=abs, ascending=False)
    )

    print("\nFEATURE CORRELATION WITH EXECUTION TIME")
    print("=" * 65)
    print(target_correlation.round(4).to_string())

    correlation.to_csv(OUTPUT_FILE)

    print("\nCorrelation matrix saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    analyse_correlations()
