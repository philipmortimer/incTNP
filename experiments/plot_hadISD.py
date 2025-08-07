# Plotting code for hadISD during validation
import copy
import os
from typing import Callable, List, Tuple, Union
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from tnp.data.hadISD import HadISDBatch, normalise_time, scale_pred_temp_dist, get_true_temp
from tnp.utils.np_functions import np_pred_fn
import wandb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import numpy as np


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# Following key plots
# 1) Show context and target stations with dots on the map
# 2) Show context station ordering (order of points)
# 3) Extrapolate predictions onto whole grid (i.e. not just stations)
# 4) Prediction at target stations
# 5) True station readings
# 6) error at target stations
# 7) absolute error at target stations
# 8) Predictions and true readings side by side for easy comparison
# 9) Gridded predictions with the context points
def plot_hadISD(
    model: nn.Module,
    batches: List[HadISDBatch],
    lat_mesh: np.ndarray,
    lon_mesh: np.ndarray,
    elev_np: np.ndarray,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
    model_lbl: str="Model",
    pred_fn: Callable = np_pred_fn,
    huge_grid_plots: bool = True, # Whether to plot huge grid plots
    device=None
):
    for i in range(num_fig):
        batch = batches[i]
        BATCH_IDX = 0 # This is implicit from the original code - we only take the first item per batch to plot
        assert BATCH_IDX == 0, "Check logic for combined kernel with non zero plot batch index"
        xc = batch.xc[:BATCH_IDX+1] # same as batch.xc[:1] (i.e. first batch item)
        yc = batch.yc[:BATCH_IDX+1]
        xt = batch.xt[:BATCH_IDX+1]
        yt = batch.yt[:BATCH_IDX+1]
        if device is not None:
            xc, yc, xt, yt =xc.to(device), yc.to(device), xt.to(device), yt.to(device)
        unnorm_time = batch.unnormalised_time[BATCH_IDX]
        x = torch.cat((xc, xt), dim=1)
        y = torch.cat((yc, yt), dim=1)

        # Batch that can be used for the stations (i.e. predict on a given set of stations given context only m = 1)
        batch_pred = HadISDBatch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt,
            mean_temp=batch.mean_temp, std_temp=batch.std_temp, mean_elev=batch.mean_elev, std_elev=batch.std_elev,
            lat_range=batch.lat_range, long_range=batch.long_range, unnormalised_time=unnorm_time, ordering=batch.ordering)
        
        # Makes a batch within the lat and long range of all points (i.e. gridded). context is same just targets different
        # Uses cached DEM file data for 2m temperature
        N_POINTS = lat_mesh.shape[0] # defines N x N grid
        #Normalise lat and lon
        lat_norm = 2.0 * (lat_mesh - batch.lat_range[0]) / (batch.lat_range[1] - batch.lat_range[0]) - 1.0
        lon_norm = 2.0 * (lon_mesh - batch.long_range[0]) / (batch.long_range[1] - batch.long_range[0]) - 1.0
        time = np.full(shape=(N_POINTS*N_POINTS), fill_value=unnorm_time.cpu())
        time = normalise_time(time)
        # Z constaint hack: TODO precache with true values
        elev_norm = (elev_np - batch.mean_elev) / batch.std_elev
        elevation = torch.tensor(elev_norm.flatten(), device=xc.device, dtype=xc.dtype) 
        # Convert stuff to tensors
        time = torch.tensor(time, device=xc.device, dtype=xc.dtype) # [N]
        lat = torch.tensor(lat_norm.flatten(), device=xc.device, dtype=xc.dtype)
        long = torch.tensor(lon_norm.flatten(), device=xc.device, dtype=xc.dtype)
        xt_grid = torch.stack((lat, long, time, elevation), dim=-1) # [N, 4]
        # Shuffles data
        if batch.ordering == "random":
            indices = torch.randperm(N_POINTS * N_POINTS)
            xt_grid = xt_grid[indices]
        else:
            raise ValueError("Unspoorted plotting ordering type")
        xt_grid = xt_grid.unsqueeze(0) # adds batch dim of 1
        # Creates batch to predict all temp readings on grid
        batch_grid = HadISDBatch(x=None, y=None, xc=xc, yc=yc, xt=xt_grid, yt=None,
            mean_temp=batch.mean_temp, std_temp=batch.std_temp, mean_elev=batch.mean_elev, std_elev=batch.std_elev,
            lat_range=batch.lat_range, long_range=batch.long_range, unnormalised_time=unnorm_time, ordering=batch.ordering)

        # Gets predictive distributions and scales to correct units
        with torch.no_grad():
            yt_pred_dist = pred_fn(model, batch_pred)
            if huge_grid_plots: y_gridded_pred_dist = pred_fn(model, batch_grid, predict_without_yt_tnpa=True)
        yt_pred_dist = scale_pred_temp_dist(batch_pred, yt_pred_dist)
        if huge_grid_plots: 
            y_gridded_pred_dist = scale_pred_temp_dist(batch_grid, y_gridded_pred_dist)
            #y_gridded_pred_dist.mean.shape = [1, N_POINTS * N_POINTS, 1]
            predicted_grid_points_shuffled = y_gridded_pred_dist.mean.squeeze(0)
            # Unshuffles predicted grid distribution points as appropriate
            if batch.ordering == "random":
                predicted_grid_points = torch.empty_like(predicted_grid_points_shuffled)
                predicted_grid_points[indices] = predicted_grid_points_shuffled
            else:
                raise ValueError("Unspoorted plotting ordering type")
            predicted_grid_points = predicted_grid_points.view(N_POINTS, N_POINTS).cpu()
        pred_tgt_points = yt_pred_dist.mean.cpu()
        # Computes NLL
        yt_correct_units = get_true_temp(batch_pred, batch_pred.yt)
        true_tgt_points = yt_correct_units.squeeze(0, -1).cpu()
        nll = -yt_pred_dist.log_prob(yt_correct_units).sum() / yt_correct_units[..., 0].numel()
        rmse = nn.functional.mse_loss(yt_pred_dist.mean, yt_correct_units).sqrt().cpu().mean() 
        _, nc, _ = batch_pred.xc.shape
        _, nt, _ = batch_pred.xt.shape
        # Converts points to true long / lat value and to cpu for for plotting
        long_ctx = (((xc[:, :,1].cpu() + 1.0) / 2.0) * (batch_pred.long_range[1] - batch_pred.long_range[0])) + batch_pred.long_range[0]
        lat_ctx = (((xc[:, :,0].cpu() + 1.0) / 2.0) * (batch_pred.lat_range[1] - batch_pred.lat_range[0])) + batch_pred.lat_range[0]
        long_tgt = (((xt[:, :,1].cpu() + 1.0) / 2.0) * (batch_pred.long_range[1] - batch_pred.long_range[0])) + batch_pred.long_range[0]
        lat_tgt = (((xt[:, :,0].cpu() + 1.0) / 2.0) * (batch_pred.lat_range[1] - batch_pred.lat_range[0])) + batch_pred.lat_range[0]
        if huge_grid_plots: long_grid = ((((batch_grid.xt[:, :,1].view(N_POINTS, N_POINTS).cpu() + 1.0) / 2.0) * (batch_pred.long_range[1] - batch_pred.long_range[0])) + batch_pred.long_range[0])
        if huge_grid_plots: lat_grid = (((batch_grid.xt[:, :,0].view(N_POINTS, N_POINTS).cpu() + 1.0) / 2.0) * (batch_pred.lat_range[1] - batch_pred.lat_range[0])) + batch_pred.lat_range[0]

        proj = ccrs.PlateCarree()
        batch_time_str = convert_time_to_str(unnorm_time.cpu().item())
        height_data = lon_mesh, lat_mesh, elev_np
        # 1) Show context and target stations
        title_a = f"NC={nc} NT={nt} - {batch_time_str}"
        fig_a, ax_a = init_earth_fig(title_a, figsize, proj, batch_pred.lat_range, batch_pred.long_range, height_data)
        ax_a.scatter(long_ctx, lat_ctx, c="k", s=10, label="Context")
        ax_a.scatter(long_tgt, lat_tgt, c="r", s=10, label="Target")
        ax_a.legend()
        save_plot(fig_a, name, i, "A", logging, savefig)

        # 2) Shows ordering of context points
        title_b = f"Context Ordering NC={nc}"
        fig_b, ax_b = init_earth_fig(title_b, figsize, proj, batch_pred.lat_range, batch_pred.long_range, height_data)
        context_order = np.arange(1, nc + 1)
        sc = ax_b.scatter(long_ctx, lat_ctx, c=context_order, cmap='plasma', s=10)
        cbar = fig_b.colorbar(sc, ax=ax_b)
        cbar.set_label(f"Context Point Order (1-{nc})")
        save_plot(fig_b, name, i, "B", logging, savefig)

        # 3) Predictions at wide range of points within box
        if huge_grid_plots:
            title_c = f"Gridded Predictions NC={nc} P={N_POINTS * N_POINTS:,} - {batch_time_str}"
            fig_c, ax_c = init_earth_fig(title_c, figsize, proj, batch_pred.lat_range, batch_pred.long_range, height_data)
            pcm = ax_c.pcolormesh(lon_mesh, lat_mesh, predicted_grid_points, cmap="coolwarm", shading="auto")
            cbar = fig_c.colorbar(pcm, ax=ax_c, orientation="vertical", pad=0.05)
            cbar.set_label("Temperature (°C)")
            save_plot(fig_c, name, i, "C", logging, savefig)

        # 4) Shows predictions at target stations
        title_d = f"Predicted Temperature RMSE={rmse:.2f} NLL={nll:.3f} NC={nc} - {batch_time_str}"
        fig_d, ax_d = init_earth_fig(title_d, figsize, proj, batch_pred.lat_range, batch_pred.long_range, height_data)
        # Consistent colour scheme between true and predicted points range
        vmin = min(pred_tgt_points.min(), true_tgt_points.min())
        vmax = max(pred_tgt_points.max(), true_tgt_points.max())
        ax_d.scatter(long_ctx, lat_ctx, c="k", s=10, label="Context")
        sc = ax_d.scatter(long_tgt, lat_tgt, c=pred_tgt_points, s=20, cmap="coolwarm", vmin=vmin, vmax=vmax)
        cbar = fig_d.colorbar(sc, ax=ax_d, orientation="vertical", pad=0.05)
        cbar.set_label("Predicted Temperature (°C)")
        ax_d.legend()
        save_plot(fig_d, name, i, "D", logging, savefig)

        # 5) Show true target station readings
        title_e = f"Recorded Temperature NC={nc} - {batch_time_str}"
        fig_e, ax_e = init_earth_fig(title_e, figsize, proj, batch_pred.lat_range, batch_pred.long_range, height_data)
        # Consistent colour scheme between true and predicted points range
        vmin = min(pred_tgt_points.min(), true_tgt_points.min())
        vmax = max(pred_tgt_points.max(), true_tgt_points.max())
        ax_e.scatter(long_ctx, lat_ctx, c="k", s=10, label="Context")
        sc = ax_e.scatter(long_tgt, lat_tgt, c=true_tgt_points, s=20, cmap="coolwarm", vmin=vmin, vmax=vmax)
        cbar = fig_e.colorbar(sc, ax=ax_e, orientation="vertical", pad=0.05)
        cbar.set_label("Measured Temperature (°C)")
        ax_e.legend()
        save_plot(fig_e, name, i, "E", logging, savefig)

        # 6) Error
        title_f = f"Prediction Error RMSE={rmse:.2f} NLL={nll:.3f} NC={nc} - {batch_time_str}"
        fig_f, ax_f = init_earth_fig(title_f, figsize, proj, batch_pred.lat_range, batch_pred.long_range, height_data)
        error_pred = (true_tgt_points - pred_tgt_points.squeeze(0, -1)).numpy()
        ax_f.scatter(long_ctx, lat_ctx, c="k", s=10, label="Context")
        max_abs_error = np.max(np.abs(error_pred))
        error_norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-max_abs_error, vmax=max_abs_error)
        sc = ax_f.scatter(long_tgt, lat_tgt, c=error_pred, s=20, cmap="seismic", norm=error_norm)
        cbar = fig_f.colorbar(sc, ax=ax_f, orientation="vertical", pad=0.05)
        cbar.set_label("Prediction Error (°C) [True - Predicted]")
        ax_f.legend()
        save_plot(fig_f, name, i, "F", logging, savefig)

        # 7) Absolute Error
        title_g = f"Absolute Prediction Error RMSE={rmse:.2f} NLL={nll:.3f} NC={nc} - {batch_time_str}"
        fig_g, ax_g = init_earth_fig(title_g, figsize, proj, batch_pred.lat_range, batch_pred.long_range, height_data)
        error_pred = torch.abs(true_tgt_points - pred_tgt_points.squeeze(0, -1))
        ax_g.scatter(long_ctx, lat_ctx, c="k", s=10, label="Context")
        sc = ax_g.scatter(long_tgt, lat_tgt, c=error_pred, s=20, cmap="viridis", vmin=error_pred.min(), vmax=error_pred.max())
        cbar = fig_g.colorbar(sc, ax=ax_g, orientation="vertical", pad=0.05)
        cbar.set_label("Absolute Prediction Error (°C)")
        ax_g.legend()
        save_plot(fig_g, name, i, "G", logging, savefig)

        # 8) Side by Side of predicted vs true temps
        title_h = f"Predicted vs Recorded Temperature RMSE={rmse:.2f} NLL={nll:.3f} NC={nc} - {batch_time_str}"
        # Wider figure this time
        fig_h, (ax_pred, ax_true) = plt.subplots(
            1, 2,
            figsize=(figsize[0], figsize[1]), # Adjust width by hand basically
            subplot_kw={"projection": proj},
            gridspec_kw={"wspace": 0.05}
        )
        for ax in (ax_pred, ax_true):
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.set_extent([*batch_pred.long_range, *batch_pred.lat_range], crs=proj)
            pcm_elev = ax.pcolormesh(lon_mesh, lat_mesh, elev_np, cmap="terrain", shading="auto")
        vmin = min(pred_tgt_points.min(), true_tgt_points.min())
        vmax = max(pred_tgt_points.max(), true_tgt_points.max())
        # Left: Predicted Temps
        ax_pred.set_title("Predicted")
        ax_pred.scatter(long_ctx, lat_ctx, c="k", s=10)
        sc_pred = ax_pred.scatter(long_tgt, lat_tgt, c=pred_tgt_points, cmap="coolwarm", vmin=vmin, vmax=vmax, s=20)
        # Right: recorded temperature
        ax_true.set_title("Recorded",)
        ax_true.scatter(long_ctx, lat_ctx, c="k", s=10)
        sc_true = ax_true.scatter(long_tgt, lat_tgt, c=true_tgt_points, cmap="coolwarm", vmin=vmin, vmax=vmax,s=20)
        cbar_ax = fig_h.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig_h.colorbar(sc_true, cax=cbar_ax)
        cbar.set_label("Temperature (°C)")
        fig_h.suptitle(title_h)
        save_plot(fig_h, name, i, "H", logging, savefig)

        if huge_grid_plots:
            # 9) Gridded predictions with context points
            title_i = f"Gridded Predictions NC={nc} P={N_POINTS * N_POINTS:,} - {batch_time_str}"
            fig_i, ax_i = init_earth_fig(title_i, figsize, proj, batch_pred.lat_range, batch_pred.long_range, height_data)
            pcm = ax_i.pcolormesh(lon_mesh, lat_mesh, predicted_grid_points, cmap="coolwarm", shading="auto")
            cbar = fig_i.colorbar(pcm, ax=ax_i, orientation="vertical", pad=0.05)
            cbar.set_label("Temperature (°C)")
            ax_i.scatter(long_ctx, lat_ctx, c="k", s=10, label="Context")
            ax_i.legend()
            save_plot(fig_i, name, i, "I", logging, savefig)

# Converts number of hours since 1st Jan 1931 into a formatted string
def convert_time_to_str(unnorm_time: int):
    ZERO_TIME = datetime.datetime(1931, 1, 1)
    final_datetime = ZERO_TIME + datetime.timedelta(hours=unnorm_time)
    return final_datetime.strftime("%H:00 %d %B %Y")

# Creates earth map figure outline to be used for plotting
def init_earth_fig(title, figsize, proj, lat_range, long_range, height_data):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([*long_range, *lat_range], crs=proj)
    if height_data is not None:
        lon_mesh, lat_mesh, elev_np = height_data
        pcm_elev = ax.pcolormesh(lon_mesh, lat_mesh, elev_np, cmap="terrain", shading="auto")
    ax.set_title(title)
    return fig, ax


# Saves each plot
def save_plot(fig, name, i, panel, logging, savefig):
    tag = f"{name}/{i:03d}_{panel}"
    if wandb.run is not None and logging:
        wandb.log({tag: wandb.Image(fig)})
    elif savefig:
        base_folder = f"{name}"
        save_name = base_folder + f"/{i:03d}_{panel}.png"
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)
        fig.savefig(save_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
