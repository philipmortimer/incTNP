# Plotting script to plot various permutations of causal tnp - particularly highlight the extrema
import numpy as np
import torch
from scipy import stats
from check_shapes import check_shapes
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.lightning_utils import LitWrapper
import time
import warnings
from tnp.data.gp import RandomScaleGPGenerator
from tnp.networks.gp import RBFKernel
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import random
import os
import wandb
from tnp.data.base import Batch, GroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch
from tnp.utils.np_functions import np_pred_fn, np_loss_fn
from typing import Callable, List, Tuple, Union, Optional
from torch import nn
import copy
import hiyapyco
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tnp.utils.experiment_utils import deep_convert_dict, extract_config
import matplotlib.patheffects as pe
from tnp.data.base import Batch
from tnp.models.incTNPBatchedPrior import IncTNPBatchedPrior


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# Looks at results of random permutations of the context set
@check_shapes(
    "xc: [1, nc, dx]", "yc: [1, nc, dy]", "xt: [1, nt, dx]", "yt: [1, nt, dy]"
)
@torch.no_grad()
def gather_rand_perms(tnp_model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor,
    no_permutations: int, device: str='cuda', batch_size: int=16):
    tot_time = time.time()
    inf_time = 0
    data_time = 0
    xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
    _, nc, dx = xc.shape
    perms_left = no_permutations
    log_p_list = []
    perm_list = []
    while perms_left > 0:
        # Batches permutations together to speed up computation
        data_start_time = time.time()
        batch_size_perm = min(batch_size, perms_left)

        # Permutations generated. Can use torch.randperm but torch.randn + argsort has lower constant (much faster) despite worse big O
        #perms = torch.stack([torch.randperm(nc, device=device) for _ in range(batch_size_perm)])
        keys = torch.randn(batch_size_perm, nc, device=device)
        perms = keys.argsort(dim=-1) 

        xc_perm_batched = xc.expand(batch_size_perm, -1, -1).gather(1, perms.unsqueeze(-1).expand(-1, -1, xc.shape[-1]))
        yc_perm_batched = yc.expand(batch_size_perm, -1, -1).gather(1, perms.unsqueeze(-1).expand(-1, -1, yc.shape[-1]))

        xt_batched = xt.expand(batch_size_perm, -1, -1)
        yt_batched = yt.expand(batch_size_perm, -1, -1)

        data_time += time.time() - data_start_time
        inf_start_time = time.time()
        nt, dy = yt_batched.shape[-2:]
        log_p = tnp_model(xc_perm_batched, yc_perm_batched, xt_batched).log_prob(yt_batched).sum(dim=(-1, -2)) / (nt * dy)

        #batch = SyntheticBatch(xc=xc_perm_batched, yc=yc_perm_batched, xt=xt_batched, yt=yt_batched, 
        #    x=torch.cat([xc_perm_batched, xt_batched], dim=1), y=torch.cat([yc_perm_batched, yt_batched], dim=1))
        #yt_pred_dist = np_pred_fn(tnp_model, batch)
        #log_p = -np_loss_fn(tnp_model, batch)
        inf_time += time.time() - inf_start_time

        log_p_list.append(log_p)
        perm_list.append(perms)
        perms_left -= batch_size_perm
    log_p = torch.cat(log_p_list)[:no_permutations]
    perms = torch.cat(perm_list)[:no_permutations]
    end_time = time.time()
    return perms, log_p, (data_time, inf_time, end_time - tot_time)


# Based off of plot.py but adapted for this case
def plot_perm(
    *,
    model: Union[nn.Module,
                 Callable[..., torch.distributions.Distribution]],
    xc: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
    yt: torch.Tensor,
    perm: torch.Tensor,
    file_name: str,
    annotate: bool = True, # Number the points or not
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Tuple[float, float] = (-2.0, 2.0),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 64,
    savefig: bool = True,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
    gt_pred: Optional[GroundTruthPredictor] = None,
):
    # Permutes context and converts everything to the same device
    device = xc.device
    perm = perm.to(device)
    model = model.to(device)
    xc_perm = xc[:, perm, :]
    yc_perm = yc[:, perm, :]

    # Generates batch synthetically
    batch = SyntheticBatch(xc=xc_perm, yc=yc_perm, xt=xt, yt=yt, x=torch.cat([xc_perm, xt], dim=1),
        y=torch.cat([yc_perm, yt], dim=1), gt_pred=gt_pred)
    plot_batch = copy.deepcopy(batch)

    steps = int(points_per_dim * (x_range[1] - x_range[0]))
    x_plot = torch.linspace(*x_range, steps, device=device)[None, :, None]
    plot_batch.xt = x_plot

    with torch.no_grad():
        y_plot_pred_dist = pred_fn(model, plot_batch) # Gets model predictions over grid
        yt_pred_dist = pred_fn(model, batch) # Get model predictions for poiints
    model_nll = -yt_pred_dist.log_prob(yt).sum() / batch.yt[..., 0].numel()
    mean, std = y_plot_pred_dist.mean, y_plot_pred_dist.stddev

    # Make figure for plotting
    fig = plt.figure(figsize=figsize)
    # Plot context and target points
    x_ctx = xc_perm[0, :, 0].cpu()
    y_ctx = yc_perm[0, :, 0].cpu()
    plt.scatter(x_ctx, y_ctx, c="k", s=30, label="Context")
    # Labels context set ordering
    if annotate:
        for j, (xj, yj) in enumerate(zip(x_ctx, y_ctx), 1):
            plt.annotate(str(j), (xj, yj), textcoords="offset points", xytext=(5, 5), fontsize=25)

    plt.scatter(xt[0, :, 0].cpu(), yt[0, :, 0].cpu(), c="r", s=30, label="Target")

    # Plot model predictions
    plt.plot(
        x_plot[0, :, 0].cpu(),
        mean[0, :, 0].cpu(),
        c="tab:blue",
        lw=3,
    )

    plt.fill_between(
        x_plot[0, :, 0].cpu(),
        mean[0, :, 0].cpu() - 2.0 * std[0, :, 0].cpu(),
        mean[0, :, 0].cpu() + 2.0 * std[0, :, 0].cpu(),
        color="tab:blue",
        alpha=0.2,
        label="Model",
    )
    title_str = f"$N = {xc.shape[1]}$ NLL = {model_nll:.3f}"

    # Adds groundtruth
    if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
        with torch.no_grad():
            gt_mean, gt_std, _ = batch.gt_pred(
                xc=xc,
                yc=yc,
                xt=x_plot,
            )
            _, _, gt_loglik = batch.gt_pred(
                xc=xc,
                yc=yc,
                xt=xt,
                yt=yt,
            )
            gt_nll = -gt_loglik.sum() / batch.yt[..., 0].numel()

        # Plot ground truth
        plt.plot(
            x_plot[0, :, 0].cpu(),
            gt_mean[0, :].cpu(),
            "--",
            color="tab:purple",
            lw=3,
        )

        plt.plot(
            x_plot[0, :, 0].cpu(),
            gt_mean[0, :].cpu() + 2 * gt_std[0, :].cpu(),
            "--",
            color="tab:purple",
            lw=3,
        )

        plt.plot(
            x_plot[0, :, 0].cpu(),
            gt_mean[0, :].cpu() - 2 * gt_std[0, :].cpu(),
            "--",
            color="tab:purple",
            label="Ground truth",
            lw=3,
        )

        title_str += f" GT NLL = {gt_nll:.3f}"

    plt.title(title_str, fontsize=24)
    plt.grid()

    # Set axis limits
    plt.xlim(x_range)
    plt.ylim(y_lim)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(fontsize=20)
    plt.tight_layout()

    fname = f"{file_name}.png"
    if wandb.run is not None and logging:
        wandb.log({fname: wandb.Image(fig)})
    elif savefig:
        plt.savefig(fname, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

# Gets a subset of perms and log_p evenly spaced across all sampled perms
def get_spaced_examples( perms: torch.Tensor, log_p: torch.Tensor, max_perms_plot: int = 20) -> (torch.Tensor, torch.Tensor):
    K, nc = perms.shape
    # Selects subset of lines if required
    if K > max_perms_plot:
        indices_plot = torch.linspace(0, K - 1, steps=max_perms_plot).long() # Every nth line - so we get wide range
        #indices_plot = torch.randperm(K)[:max_perms_plot] # Random selection of lines
        # Select the evenly spaced permutations and their log probabilities
        perms = perms[indices_plot]
        log_p = log_p[indices_plot]
    return (perms, log_p) 

# Gets the best and worst perms only to plot
def get_best_and_worst( perms: torch.Tensor, log_p: torch.Tensor, top_and_bottom_n: int = 2) -> (torch.Tensor, torch.Tensor):
    K, nc = perms.shape
    if K < top_and_bottom_n * 2:# Too few perms
        return (perms, log_p)
    log_p_new = torch.cat((log_p[:top_and_bottom_n], log_p[-top_and_bottom_n:]))
    perms_new = torch.cat((perms[:top_and_bottom_n,:], perms[-top_and_bottom_n:,:]))
    return (perms_new, log_p_new) 


def plot_parallel_coordinates_bezier(
    perms: torch.Tensor,
    log_p: torch.Tensor,
    xc: torch.Tensor,
    xt: torch.Tensor,
    file_name: str,
    curvature_strength: float = 0.2,
    alpha_line: float = 0.4,
    plot_targets: bool = False,
):
    K, nc = perms.shape

    # Convert to cpu
    perms = perms.cpu()
    log_p = log_p.cpu()
    xc = xc.squeeze().cpu()
    xt = xt.squeeze().cpu()
    

    # Colourmap
    sm = ScalarMappable(cmap=plt.get_cmap('viridis'), norm=Normalize(vmin=log_p.min(), vmax=log_p.max()))
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(15, 10))
    positions = [i for i in range(nc)]

    # Plots each permutation to graph
    perm_xs = xc[perms] # [max_perms_plot, nc]
    for i in range(perm_xs.shape[0]):
        line_color = sm.to_rgba(log_p[i])
        points = np.column_stack((positions, perm_xs[i])) # Shape [nc, 2]

        # Bulds a path by introducing a random point to define a curve for the two lines
        path_cmds = [mpath.Path.MOVETO]
        path_pts = [points[0]]
        # Plots lines between points - using bezier curves to enable seeing different lines that go between same two points
        for j in range(len(points) - 1):
            p0 = points[j]
            p1 = points[j+1]
            
            midpoint = (p0 + p1) / 2.0
            perp_vec = np.array([-(p1[1] - p0[1]), p1[0] - p0[0]]) # Perpendicular vector to p1 - p0
            norm = np.linalg.norm(perp_vec)
            perp_vec = perp_vec / norm
            random_offset = (np.random.rand() - 0.5) * 2 * curvature_strength
            control_point = midpoint + perp_vec * random_offset # Defines offset point for curve
            path_cmds.extend([mpath.Path.CURVE3, mpath.Path.CURVE3])
            path_pts.extend([control_point, p1])

        path = mpath.Path(path_pts, path_cmds)
        patch = mpatches.PathPatch(
            path, 
            facecolor='none', 
            edgecolor=line_color, 
            linewidth=1.0,
            alpha=alpha_line
        )
        ax.add_patch(patch)

    # Grid of black dots to clearly show where context points are - overlayed at each point in sequence
    positions_grid, xc_values_grid = np.meshgrid(positions, xc.numpy())
    ax.scatter(
        positions_grid, 
        xc_values_grid, 
        c='black', 
        s=15,
        alpha=0.6, 
        zorder=3, # to ensure they are plotted over lines
        label='Context Point Locations'
    )


    # Adds target locations as red lines to give indicator of why sampling certain points may be good (i.e. close to target)
    if plot_targets:
        for target_x_val in xt:
            ax.axhline(y=target_x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
        ax.plot([], [], color='red', linestyle='--', label='Target X-Coordinates')

    # Aesthetics
    ax.set_xlabel("Context Point Order", fontsize=24)
    ax.set_ylabel("X-Coordinate of Context Point", fontsize=24)
    ax.set_title(f"Ordering Performance NC={xc.shape[0]} NT={xt.shape[0]}", fontsize=30)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([i+1 for i in positions])
    ax.tick_params(axis='x', rotation=45)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Log-Likelihood", fontsize=24, rotation=270, labelpad=24)
    ax.tick_params(axis='both', which='major', labelsize=19)
    x_min, x_max = xc.min() - 0.5, xc.max() + 0.5
    if plot_targets: x_min, x_max = min(x_min, xt.min() - 0.2), max(x_max, xt.max() + 0.2)
    ax.set_ylim(x_min, x_max)
    if plot_targets: ax.legend()

    fname = f"{file_name}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=300)

    plt.close()

# Simple line graph plotting log probs as context set incrementally built up
def plot_log_p_lines(log_p_s, fname, nt):
    mult = 1.0
    cmap_obj = plt.get_cmap('viridis', len(log_p_s))
    colours = cmap_obj(range(len(log_p_s)))
    markers = ("o", "s", "^", "D", "v", "X", "P", "*")
    fig, ax = plt.subplots(figsize=(8, 6))
    x_ticks = []
    for i, (log_p, name) in enumerate(log_p_s):
        xs = [i + 1 for i in range(len(log_p))]
        ax.plot(xs, log_p, label=name, marker=markers[i % len(markers)], color=colours[i],markersize=4,
            linewidth=2.5)
        x_ticks = xs
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.margins(x=0.01)
    ax.set_xlabel("Context Size", fontsize=16*mult)
    ax.set_ylabel("Log-Likelihood", fontsize=16*mult)
    ax.set_title(f'Incremental Log-Likelihood (NT={nt})', fontsize=24*mult)
    ax.set_xticks(x_ticks)
    ax.legend(fontsize=10*mult)
    fig.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=300)


# Plots range of likelihoods with different permutations
def plot_log_p_bins(log_p, file_name, nc, nt, plain_tnp_perf=None, lines=[]):
    fig, ax = plt.subplots(figsize=(15, 10))

    lp_mean = log_p.mean()
    ax.axvline(lp_mean, color="grey", linestyle=":", linewidth=2.0, label=fr"Mean ($\mu={lp_mean:.2f}$)")
    lp_median = log_p.median()
    ax.axvline(lp_median, color="black", linestyle=":", linewidth=2.0, label=fr"Median ($\mathrm{{median}} = {lp_median:.2f}$)")
    # Adds additional lines - e.g. from greedy search
    for (name, colour, location) in lines:
        ax.axvline(location, color=colour, linestyle="-.", linewidth=2.5, label=name)
    # Histogram
    ax.hist(log_p, bins='auto', density=True)
    ax.set_xlabel("Log-Likelihood", fontsize=24)
    #ax.set_xlim(0.44, 0.56)
    ax.set_ylabel("Density", fontsize=24)
    ax.set_title(rf"Fluctuation over Context Permutations (NC={nc} NT={nt} K={log_p.shape[0]:,})", fontsize=30)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.legend(frameon=False, fontsize=24, loc="best")
    plt.tight_layout()
    plt.savefig(file_name + "_withoutbasetnp.png", bbox_inches="tight", dpi=300)
    # Adds red line to show the performance of a plain tnp model if it is given
    if plain_tnp_perf is not None:
        ax.axvline(plain_tnp_perf, color="red", linestyle="--", linewidth=2.5, 
            label=fr"TNP-D ($\ell={{{plain_tnp_perf:.2f}}}$)")
        ax.legend(frameon=False, fontsize=24, loc="best")
        plt.tight_layout()
        plt.savefig(file_name + "_withbasetnp.png", bbox_inches="tight", dpi=300)
    plt.close()

# Gets permutations and log probabilities for greedy selection (and the opposite) plus also median selection
@check_shapes(
    "xc: [1, nc, dx]", "yc: [1, nc, dy]", "xt: [1, nt, dx]", "yt: [1, nt, dy]"
)
@torch.no_grad()
def greedy_selection_strats(masked_model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, policy: str = "best", device: str="cuda"):
    assert policy == "best" or policy == "worst" or policy == "median", "Invalid policy"
    xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
    # When deciding the context set ordering, start with an empty context set and build up greedily (or according to some shallow strategy)
    _, nc, dx = xc.shape
    _, nt, dy = yt.shape
    idx_not_picked = [i for i in range(nc)] # Stores all unpicked indexes in context set

    # Incrementally builds up representation
    perm, log_p = [], []
    xc_new, yc_new = torch.empty((1, 0, dx), device=xc.device), torch.empty((1, 0, dy), device=yc.device)
    for _ in range(nc):
        # Builds up greedy probs based on all remaining candidate context points. Can also batch these together to make more effecient !!
        log_p_this_iteration = []
        for i in idx_not_picked:
            xc_candidate = torch.cat((xc_new, xc[:, i:i+1, :]), dim=1) 
            yc_candidate = torch.cat((yc_new, yc[:, i:i+1, :]), dim=1)
            log_prob_candidate = (masked_model(xc_candidate, yc_candidate, xt).log_prob(yt).sum(dim=(-1, -2)) / (nt * dy)).item()
            log_p_this_iteration.append(log_prob_candidate)
        # Selects next point in order -> can have different strategies here
        best_idx = np.argmax(log_p_this_iteration)
        worst_idx = np.argmin(log_p_this_iteration)
        k = (len(log_p_this_iteration) - 1) // 2
        median_idx = int(np.argpartition(log_p_this_iteration, k)[k])
        idx_chosen = best_idx if policy == "best" else (worst_idx if policy == "worst" else median_idx)
        point_idx = idx_not_picked[idx_chosen]
        # Updates context set and log_prob and perm
        xc_new = torch.cat((xc_new, xc[:, point_idx:point_idx+1, :]), dim=1) 
        yc_new = torch.cat((yc_new, yc[:, point_idx:point_idx+1, :]), dim=1) 
        log_p.append(log_p_this_iteration[idx_chosen])
        perm.append(point_idx)
        idx_not_picked.remove(point_idx)
    return perm, log_p


# Uses a greedy variance selection strategy for a masked tnp model that supports conditioning on an empty context sete
@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]"
)
@torch.no_grad()
def greedy_variance_ctx_builder(masked_model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor,
    policy: str = "best", device: str="cuda") -> Tuple[torch.LongTensor, torch.Tensor]:
    assert policy in {"best", "worst", "median"}, "Invalid policy"
    #assert isinstance(masked_model, IncTNPBatchedPrior), "Only supports specific zero prioir conditioned model atm"

    xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
    # When deciding the context set ordering, start with an empty context set and build up greedily (or according to some shallow strategy)
    _, _, dx = xc.shape
    m, nc, dy = yc.shape
    _, nt, _ = xt.shape

    # Tracks which context points have been picked
    picked_mask = torch.zeros(m, nc, dtype=torch.bool, device=device) # Stores whether a context point has been picked
    perm = torch.full((m, nc), -1, dtype=torch.long, device=device) # Use to record perms - don't use this in a mission critical loop
    var_track = torch.zeros((m, nc), device=device) # Tracks variance of points over time - again for visualisation not performance
    ll_track = torch.zeros((m, nc), device=device)

    # Incrementally builds up representation
    batch_idx = torch.arange(m, device=device)
    xc_new, yc_new = torch.empty((m, 0, dx), device=xc.device), torch.empty((m, 0, dy), device=yc.device)
    for step in range(nc):
        # Gathers all unpicked points into a call (vectorised / batch together for speed)
        not_picked_idx = (~picked_mask).nonzero(as_tuple=False)
        n_remaining = nc - step
        idx_remaining = not_picked_idx[:, 1].reshape(m, n_remaining) # [m, n_remaining]
        xt_candidates = xc[batch_idx.unsqueeze(1), idx_remaining] # [m, n_remaining, dx]
        yt_candidates = yc[batch_idx.unsqueeze(1), idx_remaining]

        # Runs inference
        batch = Batch(xc=xc_new, yc=yc_new, xt=xt_candidates, yt=None, y=None, x=None) # This is dodgy because maybe some models rely on x, yt, y, x
        #pred_dist = masked_model(xc_new, yc_new, xt_candidates)
        pred_dist = np_pred_fn(masked_model, batch, predict_without_yt_tnpa=True) # [m, n_remaining, dy]
        #var = pred_dist.variance.mean(-1) # [m, n_remaining]
        var = (pred_dist.log_prob(yt_candidates).sum(dim=(-1)) / (n_remaining * dy))


        # Selection strategy
        if policy == "best": selected_points = var.argmax(dim=1) # [m]
        elif policy == "worst": selected_points = var.argmin(dim=1) # [m]
        else: selected_points = var.kthvalue(k=(n_remaining // 2) + 1, dim=1)[1] # [m] - median

        # Selects points per batch
        selected_points_global = idx_remaining[batch_idx, selected_points] # [m]
        selected_variance = var[batch_idx, selected_points] # [m]

        # Updates context representation
        added_xc = xc[batch_idx, selected_points_global].unsqueeze(1)
        added_yc = yc[batch_idx, selected_points_global].unsqueeze(1)
        xc_new = torch.cat([xc_new, added_xc], dim=1)
        yc_new = torch.cat([yc_new, added_yc], dim=1)
        picked_mask[batch_idx, selected_points_global] = True # Shows that point has been picked

        # Tracks performance just for plotting purposes - not to be used for performance cricital sections
        batch = Batch(xc=xc_new, yc=yc_new, xt=xt, yt=yt, y=None, x=None)
        pred_dist = np_pred_fn(masked_model, batch, predict_without_yt_tnpa=True)
        ll = (pred_dist.log_prob(yt).sum(dim=(-1, -2)) / (nt * dy)) # [m]

        # Updates tracked stats for visualisation
        perm[batch_idx, step] = selected_points_global
        var_track[batch_idx, step] = selected_variance
        ll_track[batch_idx, step] = ll

    return perm, ll_track # both have return shape of [m, nc] each


# Generates plots of permutations
@check_shapes(
    "perms: [K, nc]", "log_p: [K]", "xc: [1, nc, dx]", "yc: [1, nc, dy]", "xt: [1, nt, dx]", "yt: [1, nt, dy]"
)
def visualise_perms(tnp_model, perms: torch.tensor, log_p: torch.tensor, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, 
    perm_best, inc_logps_best, perm_worst, inc_logps_worst,
    perm_median, inc_logps_median,
    var_perm_median, inc_vars_median, var_perm_best, inc_vars_best,
    var_perm_worst, inc_vars_worst,
    folder_path: str="plot_results/adversarial", file_id: str=str(random.randint(0, 1000000)), gt_pred: Optional[GroundTruthPredictor] = None,
    plain_tnp_model = None):
    log_p, indices = torch.sort(log_p)
    perms = perms[indices]
    #print(perms)
    #print(log_p)
    file_name = f"{folder_path}/plain_tnp_id_{file_id}"
    plot_perm(model=plain_tnp_model, xc=xc, yc=yc, xt=xt, yt=yt, perm=perms[0], savefig=True, file_name=file_name, gt_pred=gt_pred, annotate=False)
    # Visualises permutations of various centiles (ie best, worst, median etc)
    perf_int = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    for perc in perf_int:
        perc_idx = round((perc / 100) * (len(perms) - 1))
        perm, log_prob = perms[perc_idx], log_p[perc_idx]
        file_name = f"{folder_path}/seq_perm_{perc:03d}_id_{file_id}"
        plot_perm(model=tnp_model, xc=xc, yc=yc, xt=xt, yt=yt, perm=perm, savefig=True, file_name=file_name, gt_pred=gt_pred, annotate=True)

    # Parralel coordinates plot to see permutations ordering
    file_name = f"{folder_path}/parr_cord_id_{file_id}"
    plot_targets = xt.shape[1] <= 5 # Plot targets if there are not too many of them
    perms_spaced, log_p_spaced = get_spaced_examples(perms=perms, log_p=log_p, max_perms_plot=20)
    perms_extreme, log_p_extreme = get_best_and_worst(perms=perms, log_p=log_p, top_and_bottom_n=2)
    plot_parallel_coordinates_bezier(perms=perms_spaced,log_p=log_p_spaced, xc=xc, xt=xt, 
        file_name=file_name+"_spaced", plot_targets=plot_targets, alpha_line=0.4)
    plot_parallel_coordinates_bezier(perms=perms_extreme,log_p=log_p_extreme, xc=xc, xt=xt, 
        file_name=file_name+"_extreme", plot_targets=plot_targets, alpha_line=1.0)

    # Bins log probabilities to show variation in log probability with differing permutations
    plain_tnp_mean = None
    if plain_tnp_model is not None: 
        batch = SyntheticBatch(xc=xc, yc=yc, xt=xt, yt=yt, x=torch.cat([xc, xt], dim=1), y=torch.cat([yc, yt], dim=1))
        nt, dy = yt.shape[-2:]
        plain_tnp_mean = (plain_tnp_model(xc, yc, xt).log_prob(yt).sum(dim=(-1, -2)) / (nt * dy)).item()
    # gets l -> r and r->l ordering
    left_right = logp_for_perm(tnp_model, xc, yc, xt, yt, torch.arange(xc.shape[1], device=xc.device))
    print(left_right)
    right_left = logp_for_perm(tnp_model, xc, yc, xt, yt, torch.arange(xc.shape[1] -1, -1, -1, device=xc.device))
    print(right_left)
    lines_ord=[("Left-to-Right","yellow", left_right), ("Right-to-Left", "orange", right_left)]

    plot_log_p_bins(log_p.cpu(), f"{folder_path}/bins_dist_id_{file_id}", xc.shape[1], xt.shape[1], plain_tnp_mean)
    plot_log_p_bins(log_p.cpu(), f"{folder_path}/bins_dist_id_lr_{file_id}", xc.shape[1], xt.shape[1], plain_tnp_mean, lines_ord)
    # Greedy line plots 
    #Plots bins with greedy lines also
    lines = [(fr"Best Greedy (${inc_logps_best[-1]:.2f}$)", "green", inc_logps_best[-1]), (fr"Median Greedy (${inc_logps_median[-1]:.2f}$)", "yellow", inc_logps_median[-1]),
        (fr"Worst Greedy (${inc_logps_worst[-1]:.2f}$)", "blue", inc_logps_worst[-1])]
    # Para coords greedy
    perms_greedy = torch.tensor([perm_worst, perm_median, perm_best])
    log_p_greedy = torch.tensor([inc_logps_worst[-1], inc_logps_median[-1], inc_logps_best[-1]])
    plot_log_p_bins(log_p.cpu(), f"{folder_path}/bins_dist_greedy_lines_id_{file_id}", xc.shape[1], xt.shape[1], plain_tnp_mean, lines)
    plot_parallel_coordinates_bezier(perms=perms_greedy,log_p=log_p_greedy,
         xc=xc, xt=xt, file_name=f"{folder_path}/greedy_parra_cords_{file_id}", plot_targets=plot_targets, alpha_line=1.0)
    plot_log_p_lines([(inc_logps_best, "Best Greedy"), (inc_logps_median, "Median Greedy"), (inc_logps_worst, "Greedy Worst")], f"{folder_path}/greedy_lines_{file_id}", yt.shape[1])

    # Variance greedy plots
    lines = [(fr"Best Greedy-V (${inc_vars_best[-1]:.2f}$)", "green", inc_vars_best[-1]), (fr"Median Greedy-V (${inc_vars_median[-1]:.2f}$)", "yellow", inc_vars_median[-1]),
        (fr"Worst Greedy-V (${inc_vars_worst[-1]:.2f}$)", "blue", inc_vars_worst[-1])]
    perms_greedy = torch.tensor([var_perm_worst, var_perm_median, var_perm_best])
    log_p_greedy = torch.tensor([inc_vars_worst[-1], inc_vars_median[-1], inc_vars_best[-1]])
    plot_log_p_bins(log_p.cpu(), f"{folder_path}/bins_dist_vargreedy_lines_id_{file_id}", xc.shape[1], xt.shape[1], plain_tnp_mean, lines)
    plot_parallel_coordinates_bezier(perms=perms_greedy,log_p=log_p_greedy,
         xc=xc, xt=xt, file_name=f"{folder_path}/vargreedy_parra_cords_{file_id}", plot_targets=plot_targets, alpha_line=1.0)
    plot_log_p_lines([(inc_vars_best, "Best Greedy-V"), (inc_vars_median, "Median Greedy-V"), (inc_vars_worst, "Worst Greedy-V")], f"{folder_path}/vargreedy_lines_{file_id}", yt.shape[1])

def logp_for_perm(model, xc, yc, xt, yt, perm):
    xc_p = xc[:, perm, :]
    yc_p = yc[:, perm, :]
    nt, dy = yt.shape[-2:]
    batch = SyntheticBatch(xc=xc_p, yc=yc_p, xt=xt, yt=yt, x=torch.cat([xc_p, xt], dim=1), y=torch.cat([yc_p, yt], dim=1))
    mean = (model(xc, yc, xt).log_prob(yt).sum(dim=(-1, -2)) / (nt * dy)).item()
    return mean


def get_model(config_path, weights_and_bias_ref, device='cuda', seed: bool = True, instantiate_only_model: bool = False, load_mod_weights: bool = True):
    raw_config = deep_convert_dict(
        hiyapyco.load(
            config_path,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    # Initialise experiment, make path.
    config, _ = extract_config(raw_config, None)
    config = deep_convert_dict(config)

    # Instantiate experiment and load checkpoint.
    if seed: pl.seed_everything(config.misc.seed)
    if instantiate_only_model:
        experiment = instantiate(config.model)
        model = experiment
    else:
        experiment = instantiate(config)
        model = experiment.model
    experiment.config = config
    if seed: pl.seed_everything(experiment.misc.seed)

    # Loads weights and bias model
    if load_mod_weights:
        artifact = wandb.Api().artifact(weights_and_bias_ref, type='model')
        artifact_dir = artifact.download()
        ckpt_file = os.path.join(artifact_dir, "model.ckpt")
        lit_model = (
            LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                ckpt_file, model=model,
            )
        )
        model = lit_model.model
    model.to(device)
    return model





if __name__ == "__main__":
    # E.g. run with: python experiments/plot_adversarial_perms.py
    # RBF kernel params
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    # Data generator params
    nc, nt = 12, 128
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 10
    batch_size = 1
    deterministic = True
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=nc, max_nc=nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=0.1,
        deterministic=True, kernel=kernels)
    data = next(iter(gen_val))
    #i=0
    #for data in gen_val:
    #    i += 1
    #    print(i)
    #    if i > 6: break
    # Gets plain model - ensure these strings are correct
    #plain_model = get_model('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml', 
     #   'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-7ib3k6ga:v200')
    plain_model = get_model('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-a3qwpptn:v200')
    # plain tnp model-a3qwpptn:v200
    plain_model.eval()

    #masked_model = get_model('experiments/configs/synthetic1dRBF/gp_causal_tnp.yml', 
    #    'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v200')
    masked_model = get_model('experiments/configs/synthetic1dRBF/gp_causal_tnp.yml', 
        'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-8mxfyfnw:v200')
    #masked model-8mxfyfnw:v200
    masked_model.eval()

    #masked_model = get_model('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml',
     #   'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v180')
    #masked_model.eval()

    # Sorts context in order
    xc = data.xc
    yc = data.yc
    xc, indices = torch.sort(xc, dim=1)
    yc = torch.gather(yc, dim=1, index=indices)

    print("Getting perms VARIANCE greedy search")
    start_t = time.time()
    var_perm_best, inc_vars_best = greedy_variance_ctx_builder(masked_model, xc, yc, data.xt, data.yt, policy="best")
    var_perm_worst, inc_vars_worst = greedy_variance_ctx_builder(masked_model, xc, yc,data.xt, data.yt, policy="worst")
    var_perm_median, inc_vars_median = greedy_variance_ctx_builder(masked_model, xc, yc,data.xt, data.yt, policy="median")
    var_perm_best, inc_vars_best = var_perm_best.squeeze(0).tolist(), inc_vars_best.squeeze(0).tolist()
    var_perm_worst, inc_vars_worst = var_perm_worst.squeeze(0).tolist(), inc_vars_worst.squeeze(0).tolist()
    var_perm_median, inc_vars_median = var_perm_median.squeeze(0).tolist(), inc_vars_median.squeeze(0).tolist()
    print(f'Time for perms VARIANCE greedy search: {time.time()-start_t:.2f}s')
    print("Getting perms greedy search")
    start_t = time.time()
    perm_best, inc_logps_best = greedy_selection_strats(masked_model, xc, yc, data.xt, data.yt, policy="best")
    perm_worst, inc_logps_worst = greedy_selection_strats(masked_model, xc, yc, data.xt, data.yt, policy="worst")
    perm_median, inc_logps_median = greedy_selection_strats(masked_model, xc, yc, data.xt, data.yt, policy="median")
    print(f'Time for perms greedy search: {time.time()-start_t:.2f}s')

    print("Starting search")
    perms, log_p, (data_time, inference_time, total_time) = gather_rand_perms(masked_model, xc, yc, data.xt, data.yt, 
        no_permutations=10_000_000, device='cuda', batch_size=2048)
    print(f"Data time: {data_time:.2f}s, Inference time: {inference_time:.2f}s, Total time: {total_time:.2f}s")
    visualise_perms(masked_model, perms, log_p, xc, yc, data.xt, data.yt,
        folder_path="experiments/plot_results/adversarial", file_id="1", gt_pred=data.gt_pred, 
        plain_tnp_model=plain_model,
        perm_best=perm_best, inc_logps_best=inc_logps_best, perm_worst=perm_worst, inc_logps_worst=inc_logps_worst,
        perm_median=perm_median, inc_logps_median=inc_logps_median,
        var_perm_median=var_perm_median, inc_vars_median=inc_vars_median, var_perm_best=var_perm_best, inc_vars_best=inc_vars_best,
        var_perm_worst=var_perm_worst, inc_vars_worst=inc_vars_worst
        )