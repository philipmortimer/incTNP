# Handles data loading for HadISD dataset
from .base import Batch, DataGenerator
from .count_obs import cache_n_rows
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np
from pathlib import Path
import random
from bisect import bisect_left
import torch
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa
import ast


# HadISD Batch - used to recover real values for things like plotting / correct scale
@dataclass
class HadISDBatch(Batch):
    mean_temp: float
    std_temp: float
    mean_elev: float
    std_elev: float
    unnormalised_time: torch.Tensor
    lat_range: Tuple[float, float]
    long_range: Tuple[float, float]
    ordering: str

def normalise_time(x):
    # Encodes time
    return x % (365 * 24) # Modulo to make year definitely not input  - though mostly unneeded 

# Gets true temp for given y
def get_true_temp(batch: HadISDBatch, y_in: torch.tensor):
    return y_in * batch.std_temp + batch.mean_temp

# Converts a temp pred dist to correct scale
def scale_pred_temp_dist(batch: HadISDBatch, pred_dist: torch.distributions.Normal):
    mean_scaled = get_true_temp(batch, pred_dist.mean)
    std_scaled = pred_dist.stddev * batch.std_temp
    return torch.distributions.Normal(loc=mean_scaled, scale=std_scaled)

# Gets correct scale elevation
def get_true_elev(batch: HadISDBatch, y_in: torch.tensor):
    return y_in * batch.std_elev + batch.mean_elev


# Loads the array of counts per timestep for the dataloader
def load_counts_array(data_dir: str, min_n: int, max_n_practical: int):
    base_dir = Path(data_dir)
    counts_dir = base_dir / f"minn_{min_n}_maxn_{max_n_practical}"
    tbl = ds.dataset(counts_dir / "counts.parquet").to_table()
    hours = tbl["hour_int"].to_numpy() # This is already sorted
    counts = tbl["n_obs"].to_numpy()
    weights = counts / counts.sum()
    sort_idx_counts = np.argsort(counts, kind="stable") # This is needed for effecient search
    return hours, counts, sort_idx_counts


# Data loader for the HadISD dataset
class HadISDDataGenerator(DataGenerator):
    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        nt: int,
        data_directory: str,
        ordering_strategy: Literal["random", "stationctx-g", "stationctx-b"],
        **kwargs, 
    ):
        super().__init__(**kwargs)

        self.min_nc = min_nc
        self.max_nc = max_nc
        self.nt = nt
        self.data_directory = data_directory
        self.ordering_strategy = ordering_strategy
        # Dervived useful shorthands
        self.min_n = min_nc + nt
        self.max_n = max_nc + nt

        # Creates pre cached timestamp and number of observations list (if it does not already exist). Then loads it as array
        max_practical = cache_n_rows(data_directory, self.min_n, self.max_n, show=False)
        assert max_practical >= self.max_n, f"Max nt + max nc must be a valid tight upper bound at most. {max_nc} + {nt} = {self.max_n} > {max_practical}"
        self.hours, self.counts, self.sort_idx_counts = load_counts_array(data_directory, self.min_n, self.max_n)

        self.ds = ds.dataset(Path(data_directory) / "data" / "data.parquet") # Loads dataset

        # Gets max and mins for normalised variables from file
        summary_file = Path(data_directory) / "summary.txt"
        self.min_temp, self.max_temp, self.mean_temp, self.std_temp, self.min_elev, self.max_elev, self.mean_elev, self.std_elev, self.lat_range, self.long_range = None, None, None, None, None, None, None, None, None, None
        with open(summary_file, 'r') as f:
            for line in f:
                attr, val = line.split(": ")
                if attr == "min_temp_train": self.min_temp = float(val)
                elif attr == "max_temp_train": self.max_temp = float(val)
                elif attr == "mean_temp_train": self.mean_temp = float(val)
                elif attr == "std_temp_train": self.std_temp = float(val)
                elif attr == "min_elev_train": self.min_elev = float(val)
                elif attr == "max_elev_train": self.max_elev = float(val)
                elif attr == "mean_elev_train": self.mean_elev = float(val)
                elif attr == "std_elev_train": self.std_elev = float(val)
                elif attr == "lat_range": self.lat_range = ast.literal_eval(val)
                elif attr == "lon_range": self.long_range = ast.literal_eval(val)
        assert self.min_temp != None and self.max_temp != None and self.mean_temp != None and self.std_temp != None and self.min_elev != None and self.max_elev != None and self.mean_elev != None and self.std_elev != None and self.lat_range != None and self.long_range != None, "Not found values for normalising from dataset"
        self.lat_range = (float(self.lat_range[0]), float(self.lat_range[1]))
        self.long_range = (float(self.long_range[0]), float(self.long_range[1]))
        self.min_lat, self.max_lat = self.lat_range
        self.min_long, self.max_long = self.long_range

    # Gets a timestamp randomly from all timestamps with at least n observations. Weights sampling by number of samples per timestep
    def sample_timestep(self, n_obs: int):
        # Calculates eligible indices (at least n obs)
        start = bisect_left(self.counts[self.sort_idx_counts], n_obs)
        elig_indices = self.sort_idx_counts[start:]

        # Weights sampling for elible points - times with more samples are more likely to be sampled to prevent bias
        probs = self.counts[elig_indices] / self.counts[elig_indices].sum()
        idx = np.random.choice(elig_indices, p=probs)
        return self.hours[idx]

    
    # Gets n random points at a time step
    def get_data(self, time: int, n: int):
        # Filters out all other timesteps - quick because dataset is sorted as pre processing step by time
        tbl = (
            self.ds.scanner(
                filter=pc.field("time") == time,
                columns=["latitude", "longitude", "elevation", "temperature"]
            ).to_table()
        )
        n_obs = len(tbl)
        take = np.random.choice(n_obs, n, replace=False)
        tbl = tbl.take(pa.array(take)) # Picks random subset of samples

        # builds feature matrix of [lat, long, elev, hour]
        x = np.stack([
            tbl["latitude"].to_numpy(),
            tbl["longitude"].to_numpy(),
            tbl["elevation"].to_numpy(),
            np.full(n, time, dtype=np.int32)
        ], 1)
        y = tbl["temperature"].to_numpy()[:, None] # y is just the temperatures
        return x, y

    # Helper that calculates distance between points with harvestine formula
    def _harvestine_dist_between_points(self, lat_long):
        EARTH_RADIUS_KM = 6_378.1370

        long_rads = np.radians(lat_long[:, 1])[:, None]
        d_long = long_rads - long_rads.T
        lat_rads = np.radians(lat_long[:, 0])[:, None]
        d_lat = lat_rads - lat_rads.T

        # Harvestine calculation (broken down bit for simp)
        root_eq = (np.sin(d_lat / 2.0) ** 2) + (np.cos(lat_rads) * np.cos(lat_rads.T) * np.sin(d_long / 2.0) ** 2)
        dist = 2.0 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(root_eq), np.sqrt(1.0 - root_eq))
        return dist

    def station_order_gonzalez(self, lat_long):
        N = lat_long.shape[0]
        dists = self._harvestine_dist_between_points(lat_long)
        first_point = np.random.randint(N) # random first point - can probably improve this in future say by picking the known sparsest point or a border point
        order = [first_point]
        min_dist = dists[first_point].copy() # dist between point #1
        for _ in range(1, N):
            next_idx = np.argmax(min_dist)
            order.append(next_idx)
            min_dist = np.minimum(min_dist, dists[next_idx])
        return np.asarray(order, dtype=np.int64)

    # Shuffles the data according to the ordering strategy
    def order_data(self, x, y, nc):
        if self.ordering_strategy == "random":
            return x, y # get_data actually picks a random ordering for us already - no need to shuffle again
        elif self.ordering_strategy.startswith("stationctx"):
            # Shuffles the context set ordering by station location
            xc = x[:nc, :]
            yc = y[:nc, :]
            xt, yt = x[nc:, :], y[nc:, :]
            is_good = self.ordering_strategy == "stationctx-g"
            latlon = xc[:,:2].astype(np.float64)
            # Uses Gonzalez with Harvestine distance to sort data in approximate cluster order
            order = self.station_order_gonzalez(latlon)
            if not is_good: 
                bad_order = order[::-1] # Reverses order
                order=bad_order
            xc, yc = xc[order], yc[order]
            x = np.concatenate([xc, xt], axis=0)
            y = np.concatenate([yc, yt], axis=0)
            return x, y
        else:
            raise ValueError("Invalid ordering strategy")


    # Normalises the data
    def normalise_data(self, x, y):
        # Latitude and longitude to range [-1, 1]
        x[:, 0] = (2.0 * (x[:, 0] - self.min_lat) / (self.max_lat - self.min_lat)) - 1.0
        x[:, 1] = (2.0 * (x[:, 1] - self.min_long) / (self.max_long - self.min_long)) - 1.0

        # Z nomralises elevation
        x[:, 2] = (x[:, 2] - self.mean_elev) / self.std_elev # Z normalises elevation

        # Encodes time
        x[:, 3] = normalise_time(x[:, 3])
        # Z normalises temperature
        y = (y - self.mean_temp) / self.std_temp # Z normalises temp

        return x, y

    # Generates a batch
    def generate_batch(self) -> HadISDBatch:
        n = np.random.randint(self.min_n, self.max_n)

        nt = self.nt
        nc = n - nt
        DX = 4 # Lat, Long, Elev, Time
        DY = 1 # Temp
        xs, ys, raw_times = [], [], []
        for i in range(self.batch_size):
            timestep_i = self.sample_timestep(n)
            x_i, y_i = self.get_data(timestep_i, n)
            x_i_order, y_i_order = self.order_data(x_i, y_i, nc)
            unnorm_time = torch.Tensor(x_i_order[:, 3])[0] # [1] (all values are same)
            raw_times.append(unnorm_time)
            x_i_normalised, y_i_normalised = self.normalise_data(x_i_order, y_i_order)
            xs.append(torch.Tensor(x_i_normalised).to(dtype=torch.float32))
            ys.append(torch.Tensor(y_i_normalised).to(dtype=torch.float32))
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)
        times = torch.stack(raw_times, dim=0) # [m]

        xc = x[:,:nc, :]
        yc = y[:, :nc, :]
        xt = x[:, nc:n, :]
        yt = y[:, nc:n, :]

        batch_had = HadISDBatch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt,
            mean_temp=self.mean_temp, std_temp=self.std_temp, mean_elev=self.mean_elev, std_elev=self.std_elev,
            lat_range=self.lat_range, long_range=self.long_range, unnormalised_time=times,
            ordering=self.ordering_strategy)
        return batch_had