# Select which batch to process
BATCH_NUMBER = 1

import csv
import json
import socket
import sys
import time

import pandas as pd

from datetime import datetime
from pathlib import Path


# --------------------------------------------------
# PROJECT PATH CONFIGURATION
# --------------------------------------------------

PIPELINE_ROOT = Path(__file__).resolve().parents[2]

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------

from src.database.connection import get_connection
from src.features.semantic_features import extract as sqlf
from src.features.system_features import extract as sysf
from src.features.plan_features import extract as planf

from src.utils.csv_manager import append_row
from src.utils.duplicate_checker import DuplicateTracker
from src.utils.logger import log


# --------------------------------------------------
# DIRECTORY CONFIGURATION
# --------------------------------------------------

CONFIG_FILE = PIPELINE_ROOT / "config" / "config.json"

INPUT_DIR = PIPELINE_ROOT / "data" / "input"
OUTPUT_DIR = PIPELINE_ROOT / "data" / "output"
FAILED_DIR = PIPELINE_ROOT / "data" / "failed"

PLANS_DIR = PIPELINE_ROOT / "plans"
LOGS_DIR = PIPELINE_ROOT / "logs"

OUTPUT_FILE = OUTPUT_DIR / "final_dataset.csv"
FAILED_FILE = FAILED_DIR / "failed_queries.csv"


# Create required directories
for directory in [
    OUTPUT_DIR,
    FAILED_DIR,
    PLANS_DIR,
    LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

def load_config():

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# GET INPUT BATCH FILES
# --------------------------------------------------

def get_batch_files():

    batch_file = INPUT_DIR / f"Batch{BATCH_NUMBER:02d}.csv"

    if not batch_file.exists():
        raise FileNotFoundError(
            f"Batch file not found: {batch_file}"
        )

    return [batch_file]
# --------------------------------------------------
# RECORD FAILED QUERIES
# --------------------------------------------------

def record_failure(query_id, run, batch_name, error):

    file_exists = FAILED_FILE.exists()

    with open(
        FAILED_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "query_id",
                "run_number",
                "batch_file",
                "error",
                "timestamp"
            ])

        writer.writerow([
            query_id,
            run,
            batch_name,
            str(error),
            datetime.now().isoformat()
        ])


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_pipeline():

    cfg = load_config()

    batch_files = get_batch_files()

    print("=" * 60)
    print("NEW PIPELINE - DATA COLLECTION")
    print("=" * 60)

    print(f"Pipeline directory : {PIPELINE_ROOT}")
    print(f"Input directory    : {INPUT_DIR}")
    print(f"Output file        : {OUTPUT_FILE}")

    print(f"\nFound {len(batch_files)} batch files")

    # Database connection
    conn = get_connection(cfg["database"])

    # Machine information
    machine = socket.gethostname()

    # Load duplicate tracker once
    tracker = DuplicateTracker(str(OUTPUT_FILE))

    # Counters
    total_done = 0
    total_skip = 0
    total_fail = 0

    try:

        # Process every batch
        for batch_file in batch_files:

            batch_name = batch_file.name

            qs = pd.read_csv(batch_file)

            required_columns = {
                "query_id",
                "template_id",
                "sql_text"
            }

            missing_columns = (
                required_columns - set(qs.columns)
            )

            if missing_columns:
                raise ValueError(
                    f"{batch_name} is missing columns: "
                    f"{missing_columns}"
                )

            print(
                f"\nProcessing {batch_name} "
                f"({len(qs)} queries)..."
            )

            batch_done = 0

            # Process individual queries
            for _, q in qs.iterrows():

                query_id = q["query_id"]
                template_id = q["template_id"]
                sql_text = q["sql_text"]

                # Repeat each query according to configuration
                for run in range(
                    1,
                    int(cfg["runs_per_query"]) + 1
                ):

                    # Check previously collected queries
                    if tracker.exists(
                        query_id,
                        run,
                        machine
                    ):

                        total_skip += 1
                        continue

                    cur = None

                    try:

                        cur = conn.cursor()

                        # ----------------------------------
                        # STEP 1: SEMANTIC FEATURES
                        # BEFORE QUERY EXECUTION
                        # ----------------------------------

                        semantic_features = sqlf(
                            sql_text
                        )

                        # ----------------------------------
                        # STEP 2: ESTIMATED EXECUTION PLAN
                        # DOES NOT EXECUTE THE QUERY
                        # ----------------------------------

                        cur.execute(
                            "EXPLAIN (FORMAT JSON) "
                            + sql_text
                        )

                        estimated_plan = cur.fetchone()[0][0]

                        # Extract estimated plan features
                        plan_features = planf(
                            estimated_plan
                        )

                        # Remove actual execution information
                        # It is unavailable before execution.
                        plan_features.pop(
                            "actual_rows",
                            None
                        )

                        # Save estimated execution plan
                        plan_file = (
                            PLANS_DIR
                            / f"{query_id}_{run}.json"
                        )

                        with open(
                            plan_file,
                            "w",
                            encoding="utf-8"
                        ) as f:

                            json.dump(
                                estimated_plan,
                                f,
                                indent=2
                            )

                        # ----------------------------------
                        # STEP 3: SYSTEM METRICS
                        # BEFORE ACTUAL QUERY EXECUTION
                        # ----------------------------------

                        system_features = sysf()

                        # ----------------------------------
                        # STEP 4: ACTUAL QUERY EXECUTION
                        # COLLECT TARGET EXECUTION TIME
                        # ----------------------------------

                        cur.execute(
                            "EXPLAIN "
                            "(ANALYZE, BUFFERS, FORMAT JSON) "
                            + sql_text
                        )

                        actual_plan = cur.fetchone()[0][0]

                        execution_time = (
                            actual_plan["Plan"].get(
                                "Actual Total Time",
                                None
                            )
                        )

                        if execution_time is None:
                            raise ValueError(
                                "PostgreSQL did not return "
                                "Actual Total Time"
                            )

                        # ----------------------------------
                        # STEP 5: CONSTRUCT DATASET ROW
                        # ----------------------------------

                        row = {

                            "machine_name": machine,

                            "collected_by": cfg["collected_by"],

                            "timestamp": datetime.now().isoformat(),

                            "query_id": query_id,

                            "template_id": template_id,

                            "run_number": run,

                            # Semantic features (10)
                            **semantic_features,

                            # System features (4)
                            **system_features,

                            # Estimated plan features (11)
                            **plan_features,

                            # Target variable (milliseconds)
                            "execution_time": execution_time
                        }

                        # ----------------------------------
                        # STEP 6: SAVE DATASET
                        # ----------------------------------

                        append_row(
                            row,
                            str(OUTPUT_FILE)
                        )

                        # Update duplicate tracker
                        tracker.add(
                            query_id,
                            run,
                            machine
                        )

                        batch_done += 1
                        total_done += 1

                        log(
                            f"DONE {query_id} run {run}"
                        )

                        print(
                            f"  DONE {query_id} "
                            f"- run {run} "
                            f"- {execution_time:.3f} ms"
                        )

                    except Exception as e:

                        # Reset failed PostgreSQL transaction
                        conn.rollback()

                        total_fail += 1

                        record_failure(
                            query_id,
                            run,
                            batch_name,
                            e
                        )

                        log(
                            f"FAIL {query_id} "
                            f"run {run}: {e}"
                        )

                        print(
                            f"  FAIL {query_id} "
                            f"- run {run}: {e}"
                        )

                    finally:

                        if cur is not None:
                            cur.close()

                        # End the current transaction
                        # before processing the next run.
                        if not conn.closed:
                            conn.rollback()

                    # Delay between query runs
                    time.sleep(
                        float(cfg["sleep_between_runs"])
                    )

            print(
                f"\nCompleted {batch_name}: "
                f"{batch_done} new rows collected"
            )

    finally:

        conn.close()

    # ----------------------------------
    # FINAL SUMMARY
    # ----------------------------------

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETED")
    print("=" * 60)

    print(f"Successfully collected : {total_done}")
    print(f"Skipped duplicates     : {total_skip}")
    print(f"Failed executions      : {total_fail}")

    print(f"\nDataset saved to: {OUTPUT_FILE}")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    run_pipeline()
