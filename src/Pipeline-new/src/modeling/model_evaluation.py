import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = "data/output/cleaned_dataset.csv"
OUTPUT_DIR = "data/output/model_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# Define features used by the best-performing model
features = [
    "join_count",
    "where_count",
    "group_by_count",
    "order_by_count",
    "subquery_count",
    "query_length",
    "aggregate_count",
    "distinct_count",
    "condition_count",
    "table_count",
    "seq_scan_count",
    "index_scan_count",
    "hash_join_count",
    "nested_loop_count",
    "merge_join_count",
    "sort_count",
    "aggregate_node_count",
    "total_nodes",
    "plan_rows",
    "node_type",
    "cpu_percent",
    "memory_used_mb",
    "disk_read_bytes",
    "disk_write_bytes"
]

# Separate features, target, and query groups
X = df[features]
y = df["execution_time"]
groups = df["query_id"]

# Recreate the original grouped split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTrain/Test Split")
print("=" * 50)
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(
    f"Overlapping query IDs: "
    f"{len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))}"
)

# Identify numerical and categorical features
categorical_features = ["node_type"]

numerical_features = [
    feature for feature in features
    if feature not in categorical_features
]

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing configuration")
print("=" * 50)
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Configure Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Create complete machine learning pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nBest Model Configuration")
print("=" * 50)
print("Model: Random Forest")
print("Number of trees: 200")
print("Target transformation: Original")
print(f"Number of input features: {len(features)}")

# Train the model
print("\nTraining Random Forest Model...")
pipeline.fit(X_train, y_train)

print("Model training completed successfully.")

# Generate predictions
y_pred = pipeline.predict(X_test)

# Prevent negative execution-time predictions
y_pred = np.maximum(y_pred, 0)

# Calculate residuals
residuals = y_test.to_numpy() - y_pred

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
median_ae = median_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
print("\nModel Evaluation Results")
print("=" * 50)
print(f"MAE:             {mae:.4f} ms")
print(f"MSE:             {mse:.4f}")
print(f"RMSE:            {rmse:.4f} ms")
print(f"Median AE:       {median_ae:.4f} ms")
print(f"R2 Score:        {r2:.4f}")

# Create detailed prediction results
prediction_results = pd.DataFrame({
    "Query_ID": df.iloc[test_idx]["query_id"].to_numpy(),
    "Template_ID": df.iloc[test_idx]["template_id"].to_numpy(),
    "Machine_Name": df.iloc[test_idx]["machine_name"].to_numpy(),
    "Run_Number": df.iloc[test_idx]["run_number"].to_numpy(),

    "Actual_Execution_Time": y_test.to_numpy(),
    "Predicted_Execution_Time": y_pred,
    "Residual": residuals,
    "Absolute_Error": np.abs(residuals)
})

# Add original input features
for feature in features:
    prediction_results[feature] = X_test[feature].to_numpy()

# Save prediction results
prediction_path = os.path.join(
    OUTPUT_DIR,
    "random_forest_predictions.csv"
)

prediction_results.to_csv(prediction_path, index=False)

print("\nPrediction results saved successfully.")
print(f"File: {prediction_path}")

# Actual vs Predicted Execution Time
plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

# Perfect prediction reference line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Execution Time (ms)")
plt.ylabel("Predicted Execution Time (ms)")
plt.title("Random Forest: Actual vs Predicted Execution Time")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Actual vs Predicted plot saved: {plot_path}")

# Residual Distribution
plt.figure(figsize=(10, 6))

plt.hist(
    residuals,
    bins=50,
    edgecolor="black",
    alpha=0.7
)

plt.axvline(
    x=0,
    linestyle="--",
    label="Zero Residual"
)

plt.xlabel("Residual (ms)")
plt.ylabel("Frequency")
plt.title("Random Forest: Residual Distribution")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

residual_plot_path = os.path.join(
    OUTPUT_DIR,
    "residual_distribution.png"
)

plt.savefig(residual_plot_path, dpi=300)
plt.close()

print(f"Residual distribution plot saved: {residual_plot_path}")

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

print(f"MAE: {mae:.4f} ms")
print(f"RMSE: {rmse:.4f} ms")
print(f"Median Absolute Error: {median_ae:.4f} ms")
print(f"R2 Score: {r2:.4f}")

print("\nGenerated Files:")
print("1. random_forest_predictions.csv")
print("2. actual_vs_predicted.png")
print("3. residual_distribution.png")

print(f"\nOutput directory: {OUTPUT_DIR}")