
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "output" / "final_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
PLOT_DIR = OUTPUT_DIR / "eda_plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("FEATURE RELATIONSHIP ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

# Numeric columns only; exclude constant and metadata columns
# from this initial correlation analysis.
numeric_df = df.select_dtypes(include=[np.number]).copy()

excluded_columns = [
    "parallel_aware",  # Constant feature
    "run_number",      # Collection run metadata
]

numeric_df = numeric_df.drop(
    columns=excluded_columns,
    errors="ignore"
)

# Pearson correlation measures linear relationships.
correlation = numeric_df.corr(method="pearson")

target_correlation = (
    correlation["execution_time"]
    .drop("execution_time")
    .sort_values(key=lambda values: values.abs(), ascending=False)
)

print("\n" + "=" * 70)
print("CORRELATION WITH EXECUTION TIME")
print("=" * 70)
print(target_correlation.round(4).to_string())

target_correlation.to_csv(
    OUTPUT_DIR / "execution_time_correlations.csv",
    header=["pearson_correlation"]
)

# Plot the correlations with the target.
plt.figure(figsize=(10, 8))

target_correlation.sort_values().plot(
    kind="barh"
)

plt.title("Feature Correlation with Execution Time")
plt.xlabel("Pearson Correlation")
plt.ylabel("Feature")
plt.tight_layout()

target_plot = PLOT_DIR / "execution_time_correlations.png"
plt.savefig(target_plot, dpi=300, bbox_inches="tight")
plt.close()

# Correlation heatmap for numeric features.
fig, ax = plt.subplots(figsize=(16, 13))

image = ax.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.columns)))

ax.set_xticklabels(
    correlation.columns,
    rotation=90,
    fontsize=8
)

ax.set_yticklabels(
    correlation.columns,
    fontsize=8
)

ax.set_title("Numeric Feature Correlation Heatmap")

fig.colorbar(
    image,
    ax=ax,
    label="Pearson Correlation",
    shrink=0.8
)

fig.tight_layout()

heatmap_file = PLOT_DIR / "feature_correlation_heatmap.png"
fig.savefig(heatmap_file, dpi=300, bbox_inches="tight")
plt.close(fig)

# Find strongly correlated predictor pairs.
# These are candidates for investigation, not automatic removals.
predictors = correlation.drop(
    columns=["execution_time"],
    errors="ignore"
).drop(
    index=["execution_time"],
    errors="ignore"
)

upper_triangle = predictors.where(
    np.triu(
        np.ones(predictors.shape),
        k=1
    ).astype(bool)
)

strong_pairs = (
    upper_triangle
    .stack()
    .reset_index()
)

strong_pairs.columns = [
    "feature_1",
    "feature_2",
    "correlation"
]

strong_pairs = strong_pairs[
    strong_pairs["correlation"].abs() >= 0.80
].sort_values(
    "correlation",
    key=lambda values: values.abs(),
    ascending=False
)

print("\n" + "=" * 70)
print("STRONGLY CORRELATED PREDICTOR PAIRS (|r| >= 0.80)")
print("=" * 70)

if strong_pairs.empty:
    print("No strongly correlated predictor pairs found.")
else:
    print(strong_pairs.round(4).to_string(index=False))

strong_pairs.to_csv(
    OUTPUT_DIR / "strongly_correlated_features.csv",
    index=False
)

print("\nSaved files:")
print(f"- {OUTPUT_DIR / 'execution_time_correlations.csv'}")
print(f"- {OUTPUT_DIR / 'strongly_correlated_features.csv'}")
print(f"- {target_plot}")
print(f"- {heatmap_file}")

print("\nFEATURE RELATIONSHIP ANALYSIS COMPLETED")