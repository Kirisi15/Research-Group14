from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUERIES_FILE = PROJECT_ROOT / "data" / "input" / "queries.csv"
DATASET_FILE = PROJECT_ROOT / "data" / "output" / "final_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_FILE = OUTPUT_DIR / "query_complexity.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("QUERY COMPLEXITY ANALYSIS")
print("=" * 70)

print("\nLoading query definitions...")
queries_df = pd.read_csv(QUERIES_FILE)

print(f"Query definitions loaded: {len(queries_df):,}")

print("\nLoading final execution dataset...")
dataset_df = pd.read_csv(DATASET_FILE)

print(f"Execution observations loaded: {len(dataset_df):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_query_columns = {"query_id", "sql_text"}

required_dataset_columns = {"query_id"}

missing_query_columns = required_query_columns - set(queries_df.columns)
missing_dataset_columns = required_dataset_columns - set(dataset_df.columns)

if missing_query_columns:
    raise ValueError(
        f"Missing columns in queries.csv: {missing_query_columns}"
    )

if missing_dataset_columns:
    raise ValueError(
        f"Missing columns in final_dataset.csv: {missing_dataset_columns}"
    )


# ============================================================
# NORMALIZE SQL
# ============================================================

def normalize_sql(sql):
    if pd.isna(sql):
        return ""

    sql = str(sql).lower()

    # Remove SQL comments
    sql = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)

    # Replace multiple spaces/newlines with one space
    sql = re.sub(r"\s+", " ", sql)

    return sql.strip()


queries_df["normalized_query"] = queries_df["sql_text"].apply(normalize_sql)


# ============================================================
# COMPLEXITY FEATURE FUNCTIONS
# ============================================================

def count_joins(sql):
    return len(re.findall(r"\bjoin\b", sql))


def count_subqueries(sql):
    select_count = len(re.findall(r"\bselect\b", sql))
    return max(0, select_count - 1)


def count_ctes(sql):
    return len(re.findall(r"\bwith\b", sql))


def count_group_by(sql):
    return len(re.findall(r"\bgroup\s+by\b", sql))


def count_having(sql):
    return len(re.findall(r"\bhaving\b", sql))


def count_window_functions(sql):
    return len(re.findall(r"\bover\s*\(", sql))


def count_set_operations(sql):
    return len(
        re.findall(
            r"\bunion\b|\bintersect\b|\bexcept\b",
            sql
        )
    )


def count_exists(sql):
    return len(re.findall(r"\bexists\s*\(", sql))


def count_distinct(sql):
    return len(re.findall(r"\bdistinct\b", sql))


def count_order_by(sql):
    return len(re.findall(r"\border\s+by\b", sql))


def count_case(sql):
    return len(re.findall(r"\bcase\b", sql))


def count_aggregations(sql):
    return len(
        re.findall(
            r"\b(sum|avg|count|min|max)\s*\(",
            sql
        )
    )


def count_tables(sql):
    tables = set()

    # Tables appearing after FROM
    from_matches = re.findall(
        r"\bfrom\s+([a-zA-Z_][\w.]*)",
        sql
    )

    # Tables appearing after JOIN
    join_matches = re.findall(
        r"\bjoin\s+([a-zA-Z_][\w.]*)",
        sql
    )

    for table in from_matches + join_matches:
        tables.add(table.lower())

    return len(tables)


def calculate_nesting_depth(sql):
    current_depth = 0
    max_depth = 0

    for char in sql:
        if char == "(":
            current_depth += 1
            max_depth = max(max_depth, current_depth)

        elif char == ")":
            current_depth = max(0, current_depth - 1)

    return max_depth


# ============================================================
# EXTRACT COMPLEXITY FEATURES
# ============================================================

print("\nExtracting structural complexity features...")

queries_df["join_count"] = queries_df["normalized_query"].apply(count_joins)

queries_df["subquery_count"] = queries_df["normalized_query"].apply(
    count_subqueries
)

queries_df["cte_count"] = queries_df["normalized_query"].apply(
    count_ctes
)

queries_df["group_by_count"] = queries_df["normalized_query"].apply(
    count_group_by
)

queries_df["having_count"] = queries_df["normalized_query"].apply(
    count_having
)

queries_df["window_function_count"] = queries_df["normalized_query"].apply(
    count_window_functions
)

queries_df["set_operation_count"] = queries_df["normalized_query"].apply(
    count_set_operations
)

queries_df["exists_count"] = queries_df["normalized_query"].apply(
    count_exists
)

queries_df["distinct_count"] = queries_df["normalized_query"].apply(
    count_distinct
)

queries_df["order_by_count"] = queries_df["normalized_query"].apply(
    count_order_by
)

queries_df["case_count"] = queries_df["normalized_query"].apply(
    count_case
)

queries_df["aggregate_count"] = queries_df["normalized_query"].apply(
    count_aggregations
)

queries_df["table_count"] = queries_df["normalized_query"].apply(
    count_tables
)

queries_df["max_nesting_depth"] = queries_df["normalized_query"].apply(
    calculate_nesting_depth
)

queries_df["query_length"] = queries_df["normalized_query"].str.len()


# ============================================================
# STRUCTURAL COMPLEXITY SCORE
# ============================================================

queries_df["complexity_score"] = (
    queries_df["table_count"].sub(1).clip(lower=0)
    + 1.5 * queries_df["join_count"]
    + 2.0 * queries_df["subquery_count"]
    + 2.0 * queries_df["cte_count"]
    + 0.5 * queries_df["aggregate_count"]
    + queries_df["group_by_count"]
    + queries_df["having_count"]
    + 2.5 * queries_df["window_function_count"]
    + 2.5 * queries_df["set_operation_count"]
    + 2.0 * queries_df["exists_count"]
    + 0.5 * queries_df["distinct_count"]
    + 0.5 * queries_df["order_by_count"]
    + 0.5 * queries_df["case_count"]
    + (queries_df["max_nesting_depth"] - 2).clip(lower=0)
)


# ============================================================
# COMPLEXITY CLASSIFICATION
# ============================================================

def classify_complexity(score):
    if score <= 4:
        return "Simple"

    elif score <= 9:
        return "Intermediate"

    else:
        return "Complex"


queries_df["complexity"] = queries_df["complexity_score"].apply(
    classify_complexity
)


# ============================================================
# BASE QUERY ID
# ============================================================

queries_df["base_query_id"] = queries_df["query_id"].str.extract(
    r"^(Q\d+)_V\d+$"
)[0]


# ============================================================
# COMPLEXITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("COMPLEXITY DISTRIBUTION")
print("=" * 70)

complexity_counts = (
    queries_df["complexity"]
    .value_counts()
    .reindex(["Simple", "Intermediate", "Complex"])
    .fillna(0)
    .astype(int)
)

complexity_percentages = (
    complexity_counts / len(queries_df) * 100
)

for complexity in complexity_counts.index:
    print(
        f"{complexity:<15}"
        f"{complexity_counts[complexity]:>8,}"
        f" query variants "
        f"({complexity_percentages[complexity]:.2f}%)"
    )


# ============================================================
# MAP COMPLEXITY TO FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("MAPPING COMPLEXITY TO EXECUTION OBSERVATIONS")
print("=" * 70)

complexity_mapping = queries_df[
    ["query_id", "base_query_id", "complexity_score", "complexity"]
].copy()

dataset_df = dataset_df.merge(
    complexity_mapping,
    on="query_id",
    how="left",
    validate="many_to_one"
)


# ============================================================
# CHECK FOR UNMATCHED QUERIES
# ============================================================

unmatched = dataset_df["complexity"].isna().sum()

print(f"\nUnmatched execution observations: {unmatched:,}")

if unmatched > 0:
    unmatched_ids = (
        dataset_df.loc[
            dataset_df["complexity"].isna(),
            "query_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    print("\nUnmatched query IDs:")
    for query_id in unmatched_ids[:20]:
        print(f"  {query_id}")

    raise ValueError(
        "Some execution observations could not be mapped "
        "to a complexity category."
    )


# ============================================================
# EXECUTION OBSERVATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("EXECUTION OBSERVATION DISTRIBUTION")
print("=" * 70)

observation_counts = (
    dataset_df["complexity"]
    .value_counts()
    .reindex(["Simple", "Intermediate", "Complex"])
    .fillna(0)
    .astype(int)
)

observation_percentages = (
    observation_counts / len(dataset_df) * 100
)

for complexity in observation_counts.index:
    print(
        f"{complexity:<15}"
        f"{observation_counts[complexity]:>8,}"
        f" observations "
        f"({observation_percentages[complexity]:.2f}%)"
    )


# ============================================================
# QUERY VARIANT + OBSERVATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMPLEXITY SUMMARY")
print("=" * 70)

summary_rows = []

for complexity in ["Simple", "Intermediate", "Complex"]:

    variant_count = (
        queries_df.loc[
            queries_df["complexity"] == complexity,
            "query_id"
        ]
        .nunique()
    )

    observation_count = (
        dataset_df.loc[
            dataset_df["complexity"] == complexity,
            "query_id"
        ]
        .count()
    )

    variant_percentage = (
        variant_count / queries_df["query_id"].nunique() * 100
    )

    observation_percentage = (
        observation_count / len(dataset_df) * 100
    )

    summary_rows.append({
        "complexity": complexity,
        "query_variants": variant_count,
        "variant_percentage": variant_percentage,
        "execution_observations": observation_count,
        "observation_percentage": observation_percentage
    })

summary_df = pd.DataFrame(summary_rows)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "variant_percentage": "{:.2f}%".format,
            "observation_percentage": "{:.2f}%".format
        }
    )
)


# ============================================================
# EXECUTION TIME BY COMPLEXITY
# ============================================================

if "execution_time" in dataset_df.columns:

    print("\n" + "=" * 70)
    print("EXECUTION TIME BY COMPLEXITY")
    print("=" * 70)

    execution_summary = (
        dataset_df
        .groupby("complexity")["execution_time"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max"
        )
        .reindex(["Simple", "Intermediate", "Complex"])
    )

    print(
        execution_summary.to_string(
            formatters={
                "mean": "{:.3f}".format,
                "median": "{:.3f}".format,
                "std": "{:.3f}".format,
                "min": "{:.3f}".format,
                "max": "{:.3f}".format
            }
        )
    )


# ============================================================
# SAVE QUERY COMPLEXITY FILE
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save query-level complexity information.
queries_output = queries_df.drop(
    columns=["normalized_query"]
)

queries_output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(f"Query complexity file saved to:")
print(OUTPUT_FILE)

print("\nAnalysis completed successfully.")