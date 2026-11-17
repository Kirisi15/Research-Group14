from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "output" / "final_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
PLOT_DIR = PROJECT_ROOT / "data" / "output" / "eda_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EXECUTION TIME DISTRIBUTION ANALYSIS")
print("=" * 70)

print("\nLoading final execution dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# BASIC STATISTICS
# ============================================================

execution_time = df["execution_time"]

print("\n" + "=" * 70)
print("EXECUTION TIME STATISTICS")
print("=" * 70)

print(f"Count       : {execution_time.count():,}")
print(f"Mean        : {execution_time.mean():.4f} ms")
print(f"Median      : {execution_time.median():.4f} ms")
print(f"Std Dev     : {execution_time.std():.4f} ms")
print(f"Minimum     : {execution_time.min():.4f} ms")
print(f"Maximum     : {execution_time.max():.4f} ms")

print("\nPercentiles:")

percentiles = execution_time.quantile(
    [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
)

for percentile, value in percentiles.items():
    print(f"{percentile * 100:>5.0f}%       : {value:.4f} ms")


# ============================================================
# SKEWNESS
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUTION SHAPE")
print("=" * 70)

print(f"Skewness    : {execution_time.skew():.4f}")

if execution_time.skew() > 1:
    print("Interpretation: Strong right-skew detected.")
elif execution_time.skew() > 0.5:
    print("Interpretation: Moderate right-skew detected.")
else:
    print("Interpretation: Distribution is approximately symmetric.")


# ============================================================
# EXTREME VALUES
# ============================================================

print("\n" + "=" * 70)
print("EXTREME VALUE ANALYSIS")
print("=" * 70)

q1 = execution_time.quantile(0.25)
q3 = execution_time.quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = execution_time[
    (execution_time < lower_bound) |
    (execution_time > upper_bound)
]

print(f"Q1          : {q1:.4f} ms")
print(f"Q3          : {q3:.4f} ms")
print(f"IQR         : {iqr:.4f} ms")
print(f"Lower bound : {lower_bound:.4f} ms")
print(f"Upper bound : {upper_bound:.4f} ms")
print(f"Outliers    : {len(outliers):,}")
print(f"Outlier %   : {len(outliers) / len(df) * 100:.2f}%")


# ============================================================
# HISTOGRAM - ORIGINAL SCALE
# ============================================================

print("\nCreating original-scale histogram...")

plt.figure(figsize=(10, 6))

plt.hist(
    execution_time,
    bins=100
)

plt.xlabel("Execution Time (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of SQL Query Execution Time")

plt.tight_layout()

original_plot = PLOT_DIR / "execution_time_distribution.png"

plt.savefig(original_plot, dpi=300)
plt.close()

print(f"Saved: {original_plot}")


# ============================================================
# HISTOGRAM - LOG SCALE
# ============================================================

print("\nCreating log-scale histogram...")

plt.figure(figsize=(10, 6))

plt.hist(
    execution_time,
    bins=100
)

plt.xscale("log")

plt.xlabel("Execution Time (ms, logarithmic scale)")
plt.ylabel("Frequency")
plt.title("SQL Query Execution Time Distribution - Log Scale")

plt.tight_layout()

log_plot = PLOT_DIR / "execution_time_log_distribution.png"

plt.savefig(log_plot, dpi=300)
plt.close()

print(f"Saved: {log_plot}")


# ============================================================
# BOX PLOT
# ============================================================

print("\nCreating box plot...")

plt.figure(figsize=(8, 6))

plt.boxplot(execution_time)

plt.ylabel("Execution Time (ms)")
plt.title("SQL Query Execution Time Box Plot")

plt.tight_layout()

box_plot = PLOT_DIR / "execution_time_boxplot.png"

plt.savefig(box_plot, dpi=300)
plt.close()

print(f"Saved: {box_plot}")


# ============================================================
# EXTREME EXECUTION TIME INVESTIGATION
# ============================================================

print("\n" + "=" * 70)
print("EXTREME EXECUTION TIME INVESTIGATION")
print("=" * 70)


# ------------------------------------------------------------
# TOP 20 HIGHEST EXECUTION TIMES
# ------------------------------------------------------------

print("\nTop 20 highest execution times:")

top_extreme = df.nlargest(
    20,
    "execution_time"
)[
    [
        "query_id",
        "template_id",
        "machine_name",
        "run_number",
        "execution_time"
    ]
]

print(top_extreme.to_string(index=False))


# ------------------------------------------------------------
# OUTLIERS BY MACHINE
# ------------------------------------------------------------

outlier_df = df[
    df["execution_time"] > upper_bound
].copy()

print("\n" + "-" * 70)
print("OUTLIERS BY MACHINE")
print("-" * 70)

machine_outliers = (
    outlier_df["machine_name"]
    .value_counts()
    .sort_index()
)

print(machine_outliers.to_string())

print("\nOutlier percentage by machine:")

machine_total = df["machine_name"].value_counts().sort_index()

machine_outlier_percentage = (
    machine_outliers / machine_total * 100
).round(2)

print(machine_outlier_percentage.to_string())


# ------------------------------------------------------------
# OUTLIERS BY RUN
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("OUTLIERS BY RUN")
print("-" * 70)

run_outliers = (
    outlier_df["run_number"]
    .value_counts()
    .sort_index()
)

print(run_outliers.to_string())


# ------------------------------------------------------------
# EXTREME VALUES BY QUERY
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("QUERIES WITH MULTIPLE EXTREME OBSERVATIONS")
print("-" * 70)

query_outlier_counts = (
    outlier_df["query_id"]
    .value_counts()
)

repeated_extreme_queries = query_outlier_counts[
    query_outlier_counts >= 2
]

print(
    f"Queries with 2 or more outlier observations: "
    f"{len(repeated_extreme_queries):,}"
)

if len(repeated_extreme_queries) > 0:
    print("\nTop queries by number of outlier observations:")
    print(
        repeated_extreme_queries
        .head(20)
        .to_string()
    )


# ------------------------------------------------------------
# COMPLEXITY OF EXTREME OBSERVATIONS
# ------------------------------------------------------------

complexity_file = OUTPUT_DIR / "query_complexity.csv"

if complexity_file.exists():

    complexity_df = pd.read_csv(complexity_file)

    complexity_mapping = complexity_df[
        ["query_id", "complexity"]
    ].drop_duplicates()

    outlier_with_complexity = outlier_df.merge(
        complexity_mapping,
        on="query_id",
        how="left"
    )

    print("\n" + "-" * 70)
    print("OUTLIERS BY QUERY COMPLEXITY")
    print("-" * 70)

    print(
        outlier_with_complexity["complexity"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nOutlier percentage within each complexity category:")

    complexity_total = (
        df.merge(
            complexity_mapping,
            on="query_id",
            how="left"
        )
        .groupby("complexity")
        .size()
    )

    complexity_outliers = (
        outlier_with_complexity
        .groupby("complexity")
        .size()
    )

    complexity_outlier_percentage = (
        complexity_outliers
        / complexity_total
        * 100
    ).round(2)

    print(complexity_outlier_percentage.to_string())


# ------------------------------------------------------------
# EXTREME VALUE THRESHOLDS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("VERY HIGH EXECUTION TIMES")
print("-" * 70)

thresholds = [100, 500, 1000, 5000, 10000]

for threshold in thresholds:

    count = (execution_time > threshold).sum()

    percentage = count / len(df) * 100

    print(
        f"> {threshold:>5} ms : "
        f"{count:>5,} observations "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SAME QUERY PERFORMANCE ACROSS MACHINES
# ============================================================

print("\n" + "=" * 70)
print("SAME QUERY PERFORMANCE ACROSS MACHINES")
print("=" * 70)

# Summarise repeated runs for each query on each machine.
query_machine_stats = (
    df.groupby(["query_id", "machine_name"])["execution_time"]
    .agg(["count", "median", "mean", "max"])
    .reset_index()
)

# Keep query variants observed on all four machines.
machine_counts = (
    query_machine_stats.groupby("query_id")["machine_name"]
    .nunique()
)

complete_queries = machine_counts[
    machine_counts == df["machine_name"].nunique()
].index

complete_stats = query_machine_stats[
    query_machine_stats["query_id"].isin(complete_queries)
].copy()

print(f"\nQuery variants observed on all machines: "
      f"{len(complete_queries):,}")

# Rank queries by their median execution time across machines.
query_ranking = (
    complete_stats.groupby("query_id")["median"]
    .median()
    .sort_values(ascending=False)
)

top_query_ids = query_ranking.head(10).index

print("\nTop 10 slowest query variants by median machine-level time:")

top_queries = complete_stats[
    complete_stats["query_id"].isin(top_query_ids)
].sort_values(["query_id", "machine_name"])

print(
    top_queries[
        ["query_id", "machine_name", "count", "median", "max"]
    ].to_string(index=False)
)

# Compare overall execution times between machines.
print("\n" + "-" * 70)
print("OVERALL EXECUTION TIME BY MACHINE")
print("-" * 70)

machine_summary = (
    df.groupby("machine_name")["execution_time"]
    .agg(["count", "median", "mean", "max"])
    .round(3)
)

print(machine_summary.to_string())

# For each query, calculate the ratio between the slowest
# and fastest machine-level median.
machine_medians = complete_stats.pivot(
    index="query_id",
    columns="machine_name",
    values="median"
)

machine_medians = machine_medians.dropna()

# Avoid division by zero.
positive_medians = machine_medians.where(machine_medians > 0)

query_ratios = (
    positive_medians.max(axis=1)
    / positive_medians.min(axis=1)
).dropna()

print("\n" + "-" * 70)
print("CROSS-MACHINE MEDIAN VARIABILITY")
print("-" * 70)

print(f"Queries compared: {len(query_ratios):,}")
print(f"Median slowest/fastest machine ratio: "
      f"{query_ratios.median():.2f}x")
print(f"Queries with ratio >= 2x: "
      f"{(query_ratios >= 2).sum():,}")
print(f"Queries with ratio >= 5x: "
      f"{(query_ratios >= 5).sum():,}")
print(f"Queries with ratio >= 10x: "
      f"{(query_ratios >= 10).sum():,}")
# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)