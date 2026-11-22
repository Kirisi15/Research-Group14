
# ============================================================
# SQL QUERY PERFORMANCE PREDICTION
# Model Training Pipeline
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = "data/output/cleaned_dataset.csv"

OUTPUT_DIR = "data/output/model_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("SQL Query Performance Prediction")
print("=" * 60)
print("Configuration loaded successfully.")
print("Dataset:", DATA_PATH)
print("Output directory:", OUTPUT_DIR)



# ============================================================
# FEATURE GROUP DEFINITIONS
# ============================================================

# Group A: SQL Semantic Features
semantic_features = [
    "join_count",
    "where_count",
    "group_by_count",
    "order_by_count",
    "subquery_count",
    "query_length",
    "aggregate_count",
    "distinct_count",
    "condition_count",
    "table_count"
]

# Group B: Query Plan Features
plan_features = [
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

# Group C: System Metrics
system_features = [
    "cpu_percent",
    "memory_used_mb",
    "disk_read_bytes",
    "disk_write_bytes"
]

# Five experimental feature groups
feature_groups = {
    "A_Semantic": semantic_features,

    "B_Query_Plan": plan_features,

    "C_System_Metrics": system_features,

    "D_Semantic_Query_Plan":
        semantic_features + plan_features,

    "E_All_Features":
        semantic_features + plan_features + system_features
}

print("\nFeature Groups")
print("=" * 60)

for group_name, features in feature_groups.items():
    print(f"{group_name}: {len(features)} features")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

# Separate prediction target
y = df["execution_time"]

# Group identifiers are used only for splitting
groups = df["query_id"]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(df, y, groups=groups)
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Verify no query leakage
overlap = set(train_df["query_id"]) & set(test_df["query_id"])

print("\nTrain/Test Split")
print("=" * 60)
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training query groups:", train_df["query_id"].nunique())
print("Testing query groups:", test_df["query_id"].nunique())
print("Overlapping query IDs:", len(overlap))



# ============================================================
# DATA PREPROCESSING
# ============================================================

def create_preprocessor(features, scale_numeric=False):

    categorical_features = [
        feature for feature in features
        if feature == "node_type"
    ]

    numerical_features = [
        feature for feature in features
        if feature not in categorical_features
    ]

    numerical_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numerical_steps.append(
            ("scaler", StandardScaler())
        )

    numerical_transformer = Pipeline(
        steps=numerical_steps
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    transformers = [
        ("num", numerical_transformer, numerical_features)
    ]

    if categorical_features:
        transformers.append(
            ("cat", categorical_transformer, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return preprocessor


print("\nPreprocessing configuration")
print("=" * 60)

for group_name, features in feature_groups.items():

    categorical_count = sum(
        feature == "node_type" for feature in features
    )

    numerical_count = len(features) - categorical_count

    print(f"\n{group_name}")
    print("Numerical features:", numerical_count)
    print("Categorical features:", categorical_count)



# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {
    "Linear_Regression": LinearRegression(),

    "Random_Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="reg:squarederror"
    )
}

print("\nMachine Learning Models")
print("=" * 60)

for model_name in models:
    print(model_name)



# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

results = []

print("\nStarting Model Training")
print("=" * 70)

for group_name, features in feature_groups.items():

    print(f"\nFeature Group: {group_name}")
    print("-" * 70)

    X_train = train_df[features]
    X_test = test_df[features]

    for target_type in ["Original", "Log1p"]:

        if target_type == "Original":
            y_train_target = y_train
        else:
            y_train_target = np.log1p(y_train)

        for model_name, model in models.items():

            scale_numeric = model_name == "Linear_Regression"

            preprocessor = create_preprocessor(
                features,
                scale_numeric=scale_numeric
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ]
            )

            pipeline.fit(X_train, y_train_target)

            predictions = pipeline.predict(X_test)

            # Convert log predictions back to milliseconds
            if target_type == "Log1p":
                predictions = np.expm1(predictions)

            # Prevent negative execution-time predictions
            predictions = np.maximum(predictions, 0)

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(y_test, predictions)
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            results.append({
                "Feature_Group": group_name,
                "Target_Transformation": target_type,
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            print(
                f"{target_type:8} | "
                f"{model_name:18} | "
                f"MAE: {mae:.4f} | "
                f"RMSE: {rmse:.4f} | "
                f"R2: {r2:.4f}"
            )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison_results.csv"
)

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)

print("Total experiments:", len(results_df))
print("Results saved to:", results_path)
