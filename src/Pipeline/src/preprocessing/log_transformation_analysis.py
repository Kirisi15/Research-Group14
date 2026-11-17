
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "output" / "cleaned_dataset.csv"

OUTPUT_DIR = BASE_DIR / "data" / "output" / "log_transformation_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyse_log_transformations():

    df = pd.read_csv(INPUT_FILE)

    features = [
        "execution_time",
        "plan_rows"
    ]

    results = []

    for feature in features:

        original = df[feature]
        transformed = np.log1p(original)

        results.append({
            "feature": feature,
            "original_skewness": original.skew(),
            "transformed_skewness": transformed.skew(),
            "original_median": original.median(),
            "transformed_median": transformed.median()
        })

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].hist(original, bins=50, edgecolor="black")
        axes[0].set_title(f"Original {feature}")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("Frequency")

        axes[1].hist(transformed, bins=50, edgecolor="black")
        axes[1].set_title(f"Log1p Transformed {feature}")
        axes[1].set_xlabel(f"log1p({feature})")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()

        output_file = OUTPUT_DIR / f"{feature}_log_comparison.png"

        plt.savefig(output_file, dpi=300)
        plt.close()

        print("Saved:", output_file)

    result_df = pd.DataFrame(results)

    print("\nLOG TRANSFORMATION COMPARISON")
    print("=" * 75)
    print(result_df.round(4).to_string(index=False))

    result_df.to_csv(
        OUTPUT_DIR / "log_transformation_comparison.csv",
        index=False
    )


if __name__ == "__main__":
    analyse_log_transformations()
