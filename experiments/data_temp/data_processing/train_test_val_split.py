# Takes folder of paraquet files by station and makes test train and val split files for a given lat / long split and time split
import duckdb, os, datetime, math
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import time


START_TIME = 1931
EPOCH0_SQL = f"TIMESTAMP '{START_TIME}-01-01 00:00:00'"


# Writes summary file
def write_summary(path, d):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")


def process_set_to_test_train_val(SRC_ROOT, DST_ROOT, LAT_BOUNDS, LON_BOUNDS, SPLITS):
    start_t = time.time()
    dst_root = Path(DST_ROOT)
    dst_root.mkdir(parents=True, exist_ok=True)
    PARQUET_GLOB = f"{SRC_ROOT}/station_id=*/part*.parquet"

    # Selects all stations that are in lat + long bounds
    meta = ds.dataset(f"{SRC_ROOT}/station_meta.parquet")
    lat_lo, lat_hi = LAT_BOUNDS
    lon_lo, lon_hi = LON_BOUNDS
    wanted_ids = (
        meta.scanner(
            filter=(
                (pc.field("latitude") >= lat_lo) & (pc.field("latitude") <= lat_hi) & (pc.field("longitude") >= lon_lo) & (pc.field("longitude") <= lon_hi)
            ),
            columns=["station_id"]
        ).to_table()["station_id"].to_pylist()
    )
    if not wanted_ids:
        raise SystemExit("No stations inside specified lat and long bounds")

    # DuckDB used
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={os.cpu_count()}") # May also want to limit memory

    # register wanted_ids as DuckDB table
    con.register("wanted_ids", pa.table({"station_id": pa.array(wanted_ids)}))

    # Loops over train test and val - storing stats for each summary
    train_stats = {}
    i = 0
    for split, (y_min, y_max) in SPLITS.items():
        assert (i == 0 and split == "train") or i > 0, "Training must be first split for algo to work"
        print(f"\n--- {split.upper()} ---")

        split_dir = dst_root / split / "data"
        split_dir.mkdir(parents=True, exist_ok=True)
        dst_file = split_dir / "data.parquet"

        conds = ["station_id IN (SELECT station_id FROM wanted_ids)"]
        if y_min is not None: conds.append(f"time >= TIMESTAMP '{y_min}-01-01 00:00:00'")
        if y_max is not None: conds.append(f"time <  TIMESTAMP '{y_max+1}-01-01 00:00:00'")
        where_clause = " AND ".join(conds)

        # sql query for valid data
        sql = f"""
        COPY (
          SELECT
            station_id,
            latitude,
            longitude,
            elevation,
            date_diff('hour', {EPOCH0_SQL}, time) AS time,
            temperature
          FROM read_parquet('{PARQUET_GLOB}')
          WHERE {where_clause}
          ORDER BY time
        )
        TO '{dst_file}' (FORMAT PARQUET, COMPRESSION 'snappy')
        """
        con.execute(sql)
        print("Wrote file ", dst_file)

        # Tracks min and max for normalised fields and no items
        stats = con.execute(f"""
            SELECT
               min(temperature) AS t_min,
               max(temperature) AS t_max,
               avg(temperature) AS t_av,
               stddev_pop(temperature) AS t_std,
               min(elevation) AS e_min,
               max(elevation) AS e_max,
               avg(elevation) AS e_av,
               stddev_pop(elevation) AS e_std,
               count(*) AS n_rows
            FROM read_parquet('{dst_file}')
        """).fetchone()

        t_min, t_max, t_av, t_std, e_min, e_max, e_av, e_std, n_rows = stats

        summary = {
            "lat_range": LAT_BOUNDS,
            "lon_range": LON_BOUNDS,
            "years_covered": f"{y_min or '-infinity'}-{y_max or '+infinity'}",
            "no_stations": len(wanted_ids),
            "no_readings": int(n_rows),
        }

        # Adds train min and max to all results to all for normalisation across the different files regardless of true ranges
        if split == "train":
            summary.update({
                "min_temp_train": float(t_min),
                "max_temp_train": float(t_max),
                "mean_temp_train": float(t_av),
                "std_temp_train": float(t_std),
                "min_elev_train": float(e_min),
                "max_elev_train": float(e_max),
                "mean_elev_train": float(e_av),
                "std_elev_train": float(e_std),
            })
            train_stats = {k: v for k, v in summary.items() if k.endswith("_train")}
        else:
            summary.update(train_stats)
        write_summary(dst_root / split / "summary.txt", summary)

        i += 1

    # Overall high level summary
    split_info = f'train=({SPLITS["train"][0]}, {SPLITS["train"][1]}) val=({SPLITS["val"][0]}, {SPLITS["val"][1]}) test=({SPLITS["test"][0]}, {SPLITS["test"][1]})'
    over = {
        "lat_range": LAT_BOUNDS,
        "lon_range": LON_BOUNDS,
        "split_rule": split_info,
        "src_dataset": SRC_ROOT,
    }
    write_summary(dst_root / "over_summary.txt", over)
    print("Written all data")
    print(f"Total time {time.time() - start_t:.2f} (s)")


if __name__ == "__main__":
    SRC_ROOT = "/scratch/pm846/TNP/data/weather_parquet_valid"
    DST_ROOT = "/scratch/pm846/TNP/data/data_processed"
    LON_BOUNDS = (-10.0, 52.0) # Long is x, lat is y (-, - is bottom left of map)
    LAT_BOUNDS = (-20.0, 60.0)

    SPLITS = {
        "train": (START_TIME, 2016),
        "val": (2017, 2017),
        "test": (2018, 2019),
    }
    
    process_set_to_test_train_val(SRC_ROOT, DST_ROOT, LAT_BOUNDS, LON_BOUNDS, SPLITS)