import pandas as pd

DATA_PATH = "data/output/cleaned_dataset.csv"

df = pd.read_csv(DATA_PATH)

query_ids = [
    "Q019_V013",
    "Q013_V008",
    "Q143_V005",
    "Q143_V003"
]

features = [
    "query_id",
    "machine_name",
    "run_number",
    "execution_time",
    "query_length",
    "table_count",
    "join_count",
    "condition_count",
    "seq_scan_count",
    "index_scan_count",
    "hash_join_count",
    "nested_loop_count",
    "merge_join_count",
    "sort_count",
    "aggregate_node_count",
    "total_nodes",
    "plan_rows",
    "node_type"
]

for query_id in query_ids:

    query_df = df[df["query_id"] == query_id]

    print("\n" + "=" * 100)
    print(f"QUERY GROUP: {query_id}")
    print("=" * 100)

    print(
        query_df[features]
        .sort_values(["machine_name", "run_number"])
        .to_string(index=False)
    )

print("\nAnalysis completed.")