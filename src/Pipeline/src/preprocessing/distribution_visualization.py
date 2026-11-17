
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR / "data" / "output" / "cleaned_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "output" / "distribution_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_distributions():

    df = pd.read_csv(INPUT_FILE)

    features = [
        "execution_time",
        "plan_rows",
        "merge_join_count",
        "condition_count",
        "nested_loop_count",
        "distinct_count",
        "index_scan_count",
        "join_count"
    ]

    for feature in features:

        plt.figure(figsize=(9, 5))

        plt.hist(df[feature], bins=50, edgecolor="black")

        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.tight_layout()

        output_file = OUTPUT_DIR / f"{feature}_distribution.png"

        plt.savefig(output_file, dpi=300)
        plt.close()

        print("Saved:", output_file)


if __name__ == "__main__":
    visualize_distributions()
