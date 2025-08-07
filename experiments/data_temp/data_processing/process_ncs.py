# Reads folder of nc files and processes into effecient format with only relevant dataitems
import argparse, glob, os, sys, warnings, re, csv
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Converts netCDF char array to string in python for dataset
def char2str(da):
    bytes_arr = da.astype("U")
    if bytes_arr.ndim == 0:
        return bytes_arr.item().strip()
    elif bytes_arr.ndim == 1:
        return "".join(bytes_arr).strip()
    else:
        return ["".join(row).strip() for row in bytes_arr]

# Handles a single nc file
def load_nc_file_rows(path):
    ds = xr.open_dataset(path, engine="netcdf4")

    stat_id = char2str(ds["station_id"].values)
    lat = float(ds["latitude"].values)
    lon = float(ds["longitude"].values)
    # Temp data
    temp = ds["temperatures"]
    fillv = temp.encoding.get("_FillValue")
    flagged = temp.attrs["flagged_value"]
    t_min, t_max = temp.attrs["valid_min"], temp.attrs["valid_max"]
    temp_vals = temp.values

    times = xr.decode_cf(ds[["time"]])["time"].values

    # Filters invalid tempts
    invalid_temps_mask = np.isnan(temp_vals)
    invalid_temps_mask |= ((temp_vals == fillv) | (temp_vals == flagged) |
                    (temp_vals < t_min) |
                    (temp_vals > t_max))
    temp_vals = temp_vals.astype('float32')
    temp_vals[invalid_temps_mask] = np.nan


    # Returns the rows
    rows = []
    for time_row, tmp, val_temp in zip(times, temp_vals, ~invalid_temps_mask):
        new_row = [
            stat_id, # station id
            lat, # latitude
            lon, # longitude
            np.datetime_as_string(time_row, timezone='UTC'), # Date in a formatted string
            f"{tmp:.3f}" if np.isfinite(tmp) else "", # Temperature (blank if invalid)
            "1" if val_temp else "0", # Is temperature valid
            ]
        rows.append(new_row)
    return rows

# Combines all nc files in a directory and writes to the chosen output directory
def combine_nc_files(in_dir, out_dir):
    in_paths = sorted(glob.glob(os.path.join(in_dir, "*.nc*")))
    if not in_paths: sys.exit("No nc files found")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    header = ["station_id", "latitude", "longitude", "time", "temperature", "temp_valid"]

    # Writes data
    out_file = out_dir + "/combined.txt"
    with open(out_file, 'w') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        for i, p in enumerate(in_paths):
            for row in load_nc_file_rows(p):
                writer.writerow(row)
            print(f'{i}/{len(in_paths)} writes')
    print(f"Writen all to file {out_file}")


if __name__ == "__main__":
    inp_folder = "/scratch/pm846/TNP/data/nc_files"
    out_folder = "/scratch/pm846/TNP/data/para_processed"
    combine_nc_files(inp_folder, out_folder)