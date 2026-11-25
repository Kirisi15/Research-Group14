"""
TPC-H SF-10 Data Generator
===========================
Uses DuckDB's built-in TPC-H extension to generate SF-10 data,
exports to CSV, and loads into PostgreSQL.

Usage:
    cd Pipeline-new
    python generate_tpch_data.py

Requirements:
    pip install duckdb psycopg2-binary
"""

import duckdb
import json
import os
import sys
import time
import psycopg2

# ============================================================
# CONFIGURATION
# ============================================================

SCALE_FACTOR = 3  # SF-10 = ~10GB database, ~60M lineitem rows

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# PostgreSQL config (reads from your existing config.json)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "config.json")

# Temporary directory for CSV export
CSV_DIR = os.path.join(SCRIPT_DIR, "data", "tpch_sf10_csv")

# TPC-H tables in dependency order (load order matters for FK constraints)
TABLES = [
    ("region",   "r_regionkey, r_name, r_comment"),
    ("nation",   "n_nationkey, n_name, n_regionkey, n_comment"),
    ("supplier", "s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment"),
    ("customer", "c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment"),
    ("part",     "p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment"),
    ("partsupp", "ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment"),
    ("orders",   "o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment"),
    ("lineitem", "l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment"),
]

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS public.region (
    r_regionkey integer NOT NULL,
    r_name character(25),
    r_comment character varying(152)
);

CREATE TABLE IF NOT EXISTS public.nation (
    n_nationkey integer NOT NULL,
    n_name character(25),
    n_regionkey integer,
    n_comment character varying(152)
);

CREATE TABLE IF NOT EXISTS public.supplier (
    s_suppkey integer NOT NULL,
    s_name character(25),
    s_address character varying(40),
    s_nationkey integer,
    s_phone character(15),
    s_acctbal numeric(15,2),
    s_comment character varying(101)
);

CREATE TABLE IF NOT EXISTS public.customer (
    c_custkey integer NOT NULL,
    c_name character varying(25),
    c_address character varying(40),
    c_nationkey integer,
    c_phone character(15),
    c_acctbal numeric(15,2),
    c_mktsegment character(10),
    c_comment character varying(117)
);

CREATE TABLE IF NOT EXISTS public.part (
    p_partkey integer NOT NULL,
    p_name character varying(55),
    p_mfgr character(25),
    p_brand character(10),
    p_type character varying(25),
    p_size integer,
    p_container character(10),
    p_retailprice numeric(15,2),
    p_comment character varying(23)
);

CREATE TABLE IF NOT EXISTS public.partsupp (
    ps_partkey integer NOT NULL,
    ps_suppkey integer NOT NULL,
    ps_availqty integer,
    ps_supplycost numeric(15,2),
    ps_comment character varying(199)
);

CREATE TABLE IF NOT EXISTS public.orders (
    o_orderkey integer NOT NULL,
    o_custkey integer,
    o_orderstatus character(1),
    o_totalprice numeric(15,2),
    o_orderdate date,
    o_orderpriority character(15),
    o_clerk character(15),
    o_shippriority integer,
    o_comment character varying(79)
);

CREATE TABLE IF NOT EXISTS public.lineitem (
    l_orderkey integer NOT NULL,
    l_partkey integer,
    l_suppkey integer,
    l_linenumber integer NOT NULL,
    l_quantity numeric(15,2),
    l_extendedprice numeric(15,2),
    l_discount numeric(15,2),
    l_tax numeric(15,2),
    l_returnflag character(1),
    l_linestatus character(1),
    l_shipdate date,
    l_commitdate date,
    l_receiptdate date,
    l_shipinstruct character(25),
    l_shipmode character(10),
    l_comment character varying(44)
);
"""


def load_pg_config():
    """Load PostgreSQL connection settings from config.json."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return json.load(f)["database"]


def test_pg_connection():
    """Test connection to PostgreSQL and ensure the database exists."""
    cfg = load_pg_config()
    dbname = cfg.get("dbname", "tpch")

    # Connect to the default 'postgres' database first to verify credentials
    admin_cfg = dict(cfg)
    admin_cfg["dbname"] = "postgres"
    try:
        conn = psycopg2.connect(**admin_cfg)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        if not cur.fetchone():
            print(f"Database '{dbname}' does not exist. Creating it automatically...")
            cur.execute(f'CREATE DATABASE "{dbname}"')
            print(f"Database '{dbname}' created successfully.")
        cur.close()
        conn.close()
    except Exception as e:
        return False, e

    # Now verify direct connection to the target database
    try:
        conn = psycopg2.connect(**cfg)
        conn.close()
        return True, None
    except Exception as e:
        return False, e


def step1_generate_csv():
    """Generate TPC-H data using DuckDB and export to CSV files."""
    os.makedirs(CSV_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STEP 1: Generate TPC-H SF-{SCALE_FACTOR} data with DuckDB")
    print(f"{'='*60}")

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")

    print(f"Generating SF-{SCALE_FACTOR} data (this may take several minutes)...")
    t0 = time.time()
    con.execute(f"CALL dbgen(sf={SCALE_FACTOR});")
    elapsed = time.time() - t0
    print(f"Data generated in {elapsed:.1f}s")

    # Show row counts
    print("\nRow counts:")
    for table, _ in TABLES:
        count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table:12s}: {count:>15,}")

    # Export each table to CSV
    print("\nExporting to CSV...")
    for table, _ in TABLES:
        csv_path = os.path.join(CSV_DIR, f"{table}.csv").replace("\\", "/")
        print(f"  Exporting {table}...", end=" ", flush=True)
        t0 = time.time()
        con.execute(f"""
            COPY {table} TO '{csv_path}'
            (FORMAT CSV, HEADER FALSE, DELIMITER '|');
        """)
        size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"{size_mb:.1f} MB ({time.time()-t0:.1f}s)")

    con.close()
    print("\nCSV export complete!")


def step2_load_into_postgresql():
    """Truncate existing data and load CSV files into PostgreSQL."""
    cfg = load_pg_config()

    print(f"\n{'='*60}")
    print(f"STEP 2: Load into PostgreSQL ({cfg['host']}:{cfg['port']}/{cfg['dbname']})")
    print(f"{'='*60}")

    conn = psycopg2.connect(**cfg)
    conn.autocommit = False
    cur = conn.cursor()

    # Ensure tables exist before truncating/loading
    cur.execute(CREATE_TABLES_SQL)
    conn.commit()

    # Disable constraints for faster loading
    print("\nPreparing database...")
    cur.execute("SET session_replication_role = 'replica';")  # Disables FK checks

    # Truncate all tables in reverse order (respect FK deps)
    print("Truncating existing data...")
    for table, _ in reversed(TABLES):
        cur.execute(f"TRUNCATE TABLE {table} CASCADE;")
        print(f"  Truncated {table}")
    conn.commit()

    # Load each table from CSV
    print("\nLoading data (this will take a while for lineitem ~60M rows)...")
    for table, columns in TABLES:
        csv_path = os.path.abspath(os.path.join(CSV_DIR, f"{table}.csv"))

        if not os.path.exists(csv_path):
            print(f"  SKIP {table} — CSV not found")
            continue

        print(f"  Loading {table}...", end=" ", flush=True)
        t0 = time.time()

        col_list = columns.replace(" ", "")
        with open(csv_path, "r", encoding="utf-8") as f:
            cur.copy_expert(
                f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT CSV, DELIMITER '|')",
                f
            )

        conn.commit()

        cur.execute(f"SELECT count(*) FROM {table}")
        count = cur.fetchone()[0]
        elapsed = time.time() - t0
        print(f"{count:>15,} rows ({elapsed:.1f}s)")

    # Re-enable constraints
    cur.execute("SET session_replication_role = 'origin';")
    conn.commit()

    # Analyze tables for query optimizer
    print("\nRunning ANALYZE on all tables...")
    conn.autocommit = True
    for table, _ in TABLES:
        print(f"  ANALYZE {table}...", end=" ", flush=True)
        cur.execute(f"ANALYZE {table};")
        print("done")

    cur.close()
    conn.close()
    print("\nPostgreSQL load complete!")


def step3_verify():
    """Verify the loaded data."""
    cfg = load_pg_config()

    print(f"\n{'='*60}")
    print("STEP 3: Verification")
    print(f"{'='*60}")

    conn = psycopg2.connect(**cfg)
    cur = conn.cursor()

    print("\nFinal row counts in PostgreSQL:")
    for table, _ in TABLES:
        cur.execute(f"SELECT count(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table:12s}: {count:>15,}")

    # Check total database size
    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
    db_size = cur.fetchone()[0]
    print(f"\nTotal database size: {db_size}")

    cur.close()
    conn.close()


def step4_cleanup_csv():
    """Optionally remove the temporary CSV files to free disk space."""
    print(f"\n{'='*60}")
    print("STEP 4: Cleanup")
    print(f"{'='*60}")

    total_size = 0
    for table, _ in TABLES:
        csv_path = os.path.join(CSV_DIR, f"{table}.csv")
        if os.path.exists(csv_path):
            total_size += os.path.getsize(csv_path)
            os.remove(csv_path)
            print(f"  Deleted {csv_path}")

    if os.path.isdir(CSV_DIR):
        os.rmdir(CSV_DIR)

    print(f"\nFreed {total_size / (1024**3):.2f} GB of disk space")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"TPC-H SF-{SCALE_FACTOR} DATA GENERATION & LOADING")
    print("=" * 60)
    print(f"Scale Factor: {SCALE_FACTOR}")
    print(f"Target: PostgreSQL (tpch database)")
    print(f"Estimated lineitem rows: ~{SCALE_FACTOR * 6_000_000:,}")
    print()

    # Pre-flight check: Verify PostgreSQL connection before anything else!
    print("Testing PostgreSQL connection...")
    ok, err = test_pg_connection()
    if not ok:
        cfg = load_pg_config()
        print(f"\n[ERROR] Could not connect to PostgreSQL at {cfg.get('host')}:{cfg.get('port')}/{cfg.get('dbname')}")
        print(f"Reason: {err}")
        print("\nPlease update your PostgreSQL credentials in:")
        print(f"  {CONFIG_PATH}")
        sys.exit(1)
    print("PostgreSQL connection successful!\n")

    answer = input("This will REPLACE all existing data in the 'tpch' database. Continue? [y/N]: ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)

    # Check if CSV files already exist
    all_csvs_exist = all(os.path.exists(os.path.join(CSV_DIR, f"{table}.csv")) for table, _ in TABLES)
    if all_csvs_exist:
        ans = input(f"\nExisting CSV files found in '{CSV_DIR}'. Reuse existing CSVs? [Y/n]: ")
        if ans.strip().lower() != "n":
            print("Skipping Step 1 (reusing existing CSVs).")
        else:
            step1_generate_csv()
    else:
        step1_generate_csv()

    step2_load_into_postgresql()
    step3_verify()

    answer = input("\nDelete temporary CSV files to free disk space? [Y/n]: ")
    if answer.strip().lower() != "n":
        step4_cleanup_csv()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"Your PostgreSQL 'tpch' database now contains SF-{SCALE_FACTOR} data.")
    print("You can now run the pipeline to collect query execution data.")
