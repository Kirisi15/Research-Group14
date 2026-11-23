
# ============================================================
# SQL QUERY PERFORMANCE PREDICTION
# 5-Fold GroupKFold Cross-Validation
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
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
N_SPLITS = 5

DATA_PATH = "data/output/cleaned_dataset.csv"
OUTPUT_DIR = "data/output/model_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("SQL Query Performance Prediction")
print("5-Fold GroupKFold Cross-Validation")
print("=" * 65)

# ============================================================
# FEATURE GROUPS
# ============================================================

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

system_features = [
    "cpu_percent",
    "memory_used_mb",
    "disk_read_bytes",
    "disk_write_bytes"
]

feature_groups = {
    "A_Semantic": semantic_features,
    "B_Query_Plan": plan_features,
    "C_System_Metrics": system_features,
    "D_Semantic_Query_Plan": semantic_features + plan_features,
    "E_All_Features": semantic_features + plan_features + system_features
}

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

X_groups = df["query_id"]
y = df["execution_time"]

print("\nDataset loaded")
print("Rows:", len(df))
print("Unique query groups:", X_groups.nunique())

# ============================================================
# PREPROCESSING
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

    return ColumnTransformer(transformers=transformers)


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

# ============================================================
# GROUP K-FOLD SPLITTING
# ============================================================

group_kfold = GroupKFold(n_splits=N_SPLITS)

fold_results = []

print("\nStarting 5-Fold GroupKFold")
print("=" * 65)

# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

for group_name, features in feature_groups.items():

    print(f"\nFeature Group: {group_name}")
    print("-" * 65)

    X = df[features]

    for target_type in ["Original", "Log1p"]:

        if target_type == "Original":
            target = y.copy()
        else:
            target = np.log1p(y)

        for model_name, base_model in models.items():

            print(f"\n{target_type} | {model_name}")

            for fold_number, (train_idx, test_idx) in enumerate(
                group_kfold.split(X, target, groups=X_groups),
                start=1
            ):

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]

                y_train = target.iloc[train_idx]
                y_test = y.iloc[test_idx]

                train_groups = set(X_groups.iloc[train_idx])
                test_groups = set(X_groups.iloc[test_idx])

                overlap = train_groups.intersection(test_groups)

                if overlap:
                    raise ValueError(
                        f"Query leakage detected in fold {fold_number}"
                    )

                scale_numeric = model_name == "Linear_Regression"

                preprocessor = create_preprocessor(
                    features,
                    scale_numeric=scale_numeric
                )

                # Create a fresh model for every fold
                model = clone(base_model)

                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", model)
                    ]
                )

                pipeline.fit(X_train, y_train)

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

                fold_results.append({
                    "Feature_Group": group_name,
                    "Target_Transformation": target_type,
                    "Model": model_name,
                    "Fold": fold_number,
                    "Training_Rows": len(train_idx),
                    "Testing_Rows": len(test_idx),
                    "Training_Query_Groups": len(train_groups),
                    "Testing_Query_Groups": len(test_groups),
                    "Query_Group_Overlap": len(overlap),
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                })

                print(
                    f"Fold {fold_number} | "
                    f"MAE: {mae:.4f} | "
                    f"RMSE: {rmse:.4f} | "
                    f"R2: {r2:.4f}"
                )

# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_results_df = pd.DataFrame(fold_results)

fold_results_path = os.path.join(
    OUTPUT_DIR,
    "group_cross_validation_fold_results.csv"
)

fold_results_df.to_csv(fold_results_path, index=False)

# ============================================================
# CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

summary_df = (
    fold_results_df
    .groupby(
        ["Feature_Group", "Target_Transformation", "Model"]
    )
    .agg(
        MAE_Mean=("MAE", "mean"),
        MAE_Std=("MAE", "std"),
        RMSE_Mean=("RMSE", "mean"),
        RMSE_Std=("RMSE", "std"),
        R2_Mean=("R2", "mean"),
        R2_Std=("R2", "std")
    )
    .reset_index()
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "group_cross_validation_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("GROUP CROSS-VALIDATION COMPLETED")
print("=" * 65)

print("Total experiments:", summary_df.shape[0])
print("Total fold evaluations:", fold_results_df.shape[0])

print("\nResults saved:")
print(fold_results_path)
print(summary_path)

print("\nSummary:")
print(
    summary_df.sort_values(
        by="MAE_Mean",
        ascending=True
    ).to_string(index=False)
)
