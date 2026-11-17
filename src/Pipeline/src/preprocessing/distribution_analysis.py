
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "output" / "cleaned_dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "output" / "distribution_analysis.csv"

def analyse_distributions():
    df = pd.read_csv(INPUT_FILE)

    numerical_df = df.select_dtypes(include=["number"])

    results = []

    for column in numerical_df.columns:
        results.append({
            "feature": column,
            "skewness": numerical_df[column].skew(),
            "minimum": numerical_df[column].min(),
            "median": numerical_df[column].median(),
            "mean": numerical_df[column].mean(),
            "maximum": numerical_df[column].max()
        })

    result_df = pd.DataFrame(results)
    result_df["absolute_skewness"] = result_df["skewness"].abs()

    result_df = result_df.sort_values(
        by="absolute_skewness",
        ascending=False
    )

    print("=" * 75)
    print("NUMERICAL FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 75)

    print(result_df.round(4).to_string(index=False))

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nAnalysis saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    analyse_distributions()
