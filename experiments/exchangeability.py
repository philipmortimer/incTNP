# Measures sensitivity of TNP model to exchanging order of data based on eq 5 from https://proceedings.mlr.press/v253/mlodozeniec24a.html
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
import wandb
import os
from typing import Optional
from plot_adversarial_perms import get_model
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
import random
from tnp.utils.np_functions import np_pred_fn
from tnp.models.gp_online import GPStreamRBF
from tnp.data.base import Batch
from matplotlib.ticker import LogFormatterMathtext
from tnp.models.tnpa import TNPA
import gpytorch
import torch.distributions as td
from pathlib import Path
import re


# Good generic plt figures that take the font used in plot.py already
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.1,
    "grid.alpha": 0.25,
    "grid.linestyle": "",
})

# Computes log joint variance of model - use Eq 5 but only for a fixed target and context set
@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]" , "perms_ctx: [K, nc]"
)
def m_var_fixed(tnp_model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, perms_ctx: torch.Tensor, 
    gt_pred, return_sample_index: Optional[int] = None,
    sub_batch_size=1024,
    use_torch_grad: bool = False):
    with torch.set_grad_enabled(use_torch_grad):
        is_gp_model = isinstance(tnp_model, GPStreamRBF)
        # Computes ground truth nll measure (as mentioned in caption of figure 2)
        _, _, gt_loglik = gt_pred(
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
        ) # [m, nt]
        gt_nll = -gt_loglik.sum(dim=1).to(xc.device) # sums over nt to get shape of [m]
        gt_nll = gt_nll.unsqueeze(0) # [1, m] - allows for broacasting when we subtract later on

        # perms_ctx = [K, nc] - K permutations for nc context points with their indices in the tensor
        K, nc = perms_ctx.shape
        _, _, dy = yc.shape
        m, _, dx = xc.shape
        nt = yt.shape[1]
        k_m = K * m
        assert (is_gp_model and dy == 1) or not is_gp_model, "GP must have dy = 1"

        assert return_sample_index == None or (return_sample_index >= 0 and return_sample_index < m), "Invalid return index"

        # Broadcasts context
        xc_broad = xc.unsqueeze(0).expand(K, -1, -1, -1) # [K, m, nc, dx]
        yc_broad = yc.unsqueeze(0).expand(K, -1, -1, -1) # [K, m, nc, dy]

        # Generates perm gather indices
        gather_x_idx = perms_ctx.view(K, 1, nc, 1).expand(-1, m, -1, dx)
        gather_y_idx = perms_ctx.view(K, 1, nc, 1).expand(-1, m, -1, dy)

        # Permutations
        xc_perm = torch.gather(xc_broad, 2, gather_x_idx)
        yc_perm = torch.gather(yc_broad, 2, gather_y_idx)

        # Broadcast target points to match shape
        xt_rep = xt.unsqueeze(0).expand(K, -1, -1, -1) # [K, m, nc, dx]
        yt_rep = yt.unsqueeze(0).expand(K, -1, -1, -1) # [K, m, nc, dy]

        # Flattens data into a single batch
        xc_perm = xc_perm.reshape(k_m, nc, dx)
        yc_perm = yc_perm.reshape(k_m, nc, dy)
        xt_rep = xt_rep.reshape(k_m, nt, dx)
        yt_rep = yt_rep.reshape(k_m, nt, dy)

        # Creates a batch
        x = torch.cat((xc_perm, xt_rep), dim=1)
        y = torch.cat((yc_perm, yt_rep), dim=1)

        # Chunked forward pass to prevent out of memory errors
        all_log_probs = []
        for i in range(0, k_m, sub_batch_size):
            chunk_end = min(i + sub_batch_size, k_m)
            chunk_xc = xc_perm[i:chunk_end]
            chunk_yc = yc_perm[i:chunk_end]
            chunk_xt = xt_rep[i:chunk_end]
            chunk_yt = yt_rep[i:chunk_end]
            chunk_x = torch.cat((chunk_xc, chunk_xt), dim=1)
            chunk_y = torch.cat((chunk_yc, chunk_yt), dim=1)
            chunk_batch = Batch(xc=chunk_xc, yc=chunk_yc, xt=chunk_xt, yt=chunk_yt, y=chunk_y, x=chunk_x)
            
            # Model inference.
            pred_dist = np_pred_fn(tnp_model, chunk_batch, predict_without_yt_tnpa=True)

            # Computes log likelihood
            if is_gp_model:
                # Computes marg (ie takes diagonal)
                pred_dist = pred_dist.to_data_independent_dist()
                pred_dist = td.Normal(loc=pred_dist.mean.unsqueeze(-1), scale=pred_dist.stddev.unsqueeze(-1)) # Expands for dy=1 term
            # Handles MVN
            if isinstance(pred_dist, torch.distributions.MultivariateNormal):
                log_p_chunk = pred_dist.log_prob(chunk_yt.squeeze(-1))
            else:
                log_p_chunk = pred_dist.log_prob(chunk_yt).sum(dim=(-1, -2))
            all_log_probs.append(log_p_chunk)

            # Delete unused datastraight away - probably excessive but just to be sure
            #del chunk_batch, chunk_xc, chunk_yc, chunk_xt, chunk_yt, log_p_chunk

        # More deletions
        #del xc_perm, yc_perm, xt_rep
        #torch.cuda.empty_cache()

        # Converts list to tensor required
        log_probs = torch.cat(all_log_probs, dim=0) # [K * m, nt, dy] 
        # Handles GP vs TNP cases seperately
        #if is_gp_model:
        #    log_probs = log_probs.view(K, m) # [K, m]
        
        #log_probs = log_probs.sum(dim=(-1, -2)).view(K, m)  # sums over nt and dy [K, m]
        log_probs = log_probs.view(K, m)

        variance = log_probs.var(dim=0, unbiased=True) # Variance over K - this is Var_PI from eq 5 in paper (this gives [m])
        m_var_val = variance.mean().item() # Mean over m batches sampled from D^(n) - ie the monte-carlo approximation of the expectation
        # Also computes simplified version of eq 6 - the average NLL (a measure of model performance)
        neg_lp = - log_probs # [K, m]
        excess_nll = neg_lp - gt_nll # This done to be in line with caption from figure 4 where mean is subtracted by gt_mean
        mean = excess_nll.mean(dim=0) # expectation over K - i.e. E_PI (shape is [m])
        m_mean_val = mean.mean().item() # Avergaes over the m batches sampled from D^(n)

        # Can also randomly return a sample
        rand_m_var = variance[return_sample_index] if return_sample_index is not None else None
        rand_m_mean = mean[return_sample_index] if return_sample_index is not None else None
        # Note - in this current implementation we only permutes the final full context. We don't account for incremental context points.
        # This may need reconsidering when we consider the masked variant with incremental learning. 
        # This func computes E_{(C, T) ~ D^(n)}[Var_{PI}[sigma_{j=1}^{nt} log p_{\theta}(y_{t, j} | C_{PI}, x_{t, j})]]
        # when using a non-diagonal TNP. The inner expression changes when using different NPs (e.g. autoreg) and this needs to be considered also.
        return m_var_val, m_mean_val, rand_m_var, rand_m_mean

# Computes log joint variance of model. This is the full equation 5 from the paper - but more expensive
@check_shapes(
    "x: [m, n, dx]", "y: [m, n, dy]", "perms: [K, n]"
)
@torch.no_grad()
def m_var_autoreg(tnp_model, x: torch.Tensor, y: torch.Tensor, perms: torch.Tensor, return_sample_index: Optional[int] = None):
    # Computes the full equation 5 from the paper. Does this by incrementally expanding the context set and predicting only the next point
    K, n = perms.shape
    m = x.shape[0]

    assert return_sample_index == None or (return_sample_index >= 0 and return_sample_index < m), "Invalid return index"

    log_probs_list = []
    for k in range(K):
        x_perm = x[:, perms[k], :]
        y_perm = y[:, perms[k], :]
        log_prob_list = torch.zeros(m, device=x.device)
        # Note Appendix F.3.1 from paper about conditioning on no data (i.e. dont condition on empty context set)
        for i in range(1, n):
            xc = x_perm[:, :i, :]
            yc = y_perm[:, :i, :]
            xt = x_perm[:, i:i+1, :]
            yt = y_perm[:, i:i+1, :]
            log_p = tnp_model(xc, yc, xt).log_prob(yt) # [m, 1, dy]
            log_p = log_p.sum(dim=(-1, -2)) # sums out nt(=1) and dy giving shape [m] this gives us joint
            log_prob_list += log_p
        log_probs_list.append(log_prob_list)
    
    log_probs = torch.stack(log_probs_list, dim=0) # [K, m]
    variance = log_probs.var(dim=0, unbiased=True) # Variance over K - this is Var_PI from eq 5 in paper (this gives [m])
    m_var_val = variance.mean().item() # Mean over m batches sampled from D^(n) - ie the monte-carlo approximation of the expectation

    # Also computes eq 6 - the average NLL (a measure of model performance)
    mean = log_probs.mean(dim=0) # expectation over K - i.e. E_PI (shape is [m])
    m_mean_val = mean.mean().item() # Avergaes over the m batches sampled from D^(n)
    # Can also randomly return a sample
    rand_m_var = variance[return_sample_index] if return_sample_index is not None else None
    rand_m_mean = mean[return_sample_index] if return_sample_index is not None else None
    # Note - this function computes full equation 5 - incrementally building up the context set and predicting the next point.
    # This is slow and may look to optimise with KV.
    # Additionally, may wish to provide more control e.g. over batch size (do we actually want a batch size of 1 here?).
    # This func computes E_{(X, Y) ~ D^(n)}[Var_{PI}[sigma_{j=1}^{n} log p_{\theta}(y_{j, PI} | x_{1:j, PI}, y_{1:j-1, PI})]]
    # Also want to consider implication of only predicting the next point. Does this make sense for non-diagonal TNPs where there may be an
    # advantage to modelling more targets together.
    return m_var_val, m_mean_val, rand_m_var, rand_m_mean


# Samples variance over trained models with different seeds
def exchange(models_with_different_seeds, data_loader, no_permutations, device, use_autoreg_eq, max_samples, seq_len, use_torch_grad, return_samples=None):
    assert return_samples == None or return_samples <= max_samples, "Cant return more samples than are computed"
    no_models = len(models_with_different_seeds)
    # Note - may want to consider diving by nt in future? (or even nt * dy)

    m_vars = []
    m_nlls = []
    m_var_nll_samples = []
    i = 0
    nc_prev, nt_prev = None, None
    for data in data_loader:
        # Ensures sequence length is correct and that nc and nt remain constant over samples.
        # This prevents greater variance for longer sequences (ie lacking comparison) but comes at cost of expressivity.
        # Could normalise variances or look at multiple sequence lengths.
        seq_len_data = data.x.shape[1]
        batch_size = data.xc.shape[0]
        assert seq_len_data == seq_len, f"Data sequence length {seq_len_data} does not match required sequence length {seq_len}."
        if nc_prev is not None and nt_prev is not None:
            assert data.xc.shape[1] == nc_prev, f"Context set size {data.xc.shape[1]} does not match previous size {nc_prev}."
            assert data.xt.shape[1] == nt_prev, f"Target set size {data.xt.shape[1]} does not match previous size {nt_prev}."
        nc_prev, nt_prev = data.xc.shape[1], data.xt.shape[1]

        if use_autoreg_eq:
            x, y = data.x, data.y
            x, y = x.to(device), y.to(device)
            n = x.shape[1]
            perms = torch.stack([torch.randperm(n, device=device) for _ in range(no_permutations)])
            #keys = torch.randn(no_permutations, n, device=device)
            #perms = keys.argsort(dim=-1) 
        else:
            xc, yc, xt, yt = data.xc, data.yc, data.xt, data.yt
            xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
            nc, nt = xc.shape[1], xt.shape[1]
            n = nc + nt
            perms = torch.stack([torch.randperm(nc, device=device) for _ in range(no_permutations)])
            #keys = torch.randn(no_permutations, nc, device=device)
            #perms = keys.argsort(dim=-1) 
        mods_out_mvar = []
        mods_out_mnll = []
        # Calculates if an individual sample is needed for plotting
        more_samples = return_samples is not None and len(m_var_nll_samples) < return_samples
        if more_samples: 
            #random.seed(13)
            return_sample_index = random.randint(0, batch_size - 1)
            
        # Computes m_var for each model
        for model in models_with_different_seeds:
            val = m_var_autoreg(model, x, y, perms, return_sample_index=return_sample_index) if use_autoreg_eq else m_var_fixed(model, xc, yc, xt, yt, perms, gt_pred=data.gt_pred, return_sample_index=return_sample_index, use_torch_grad=use_torch_grad)

            mods_out_mvar.append(val[0])
            mods_out_mnll.append(val[1])
            if more_samples: m_var_nll_samples.append((val[2], val[3]))
        m_vars.append(mods_out_mvar)
        m_nlls.append(mods_out_mnll)
        i += 1
        print(f'i={i} max_samples={max_samples} processed_so_far={i * batch_size}')
        if max_samples is not None and i >= max_samples: break

    assert i>=1, "No data batches were processed."
    #assert return_samples is None or len(m_var_nll_samples) == return_samples, "Not enough return samples due to too small data loader"

    m_vars = np.array(m_vars)
    m_nlls = np.array(m_nlls)

    model_vars = m_vars.mean(axis=0) # Average for each model over the data batches
    model_nlls = m_nlls.mean(axis=0) # Average NLL over the data batches

    mean_m_vars = model_vars.mean() # Mean over the models
    mean_m_nlls = model_nlls.mean() # Mean NLL over the models

    # Can't do t test with single model
    if no_models == 1:
        return (mean_m_vars, None), (mean_m_nlls, None), m_var_nll_samples
    student_t_crit = stats.t.ppf(0.975, df=no_models - 1)
    sem_m_var = stats.sem(model_vars)
    sem_m_nll = stats.sem(model_nlls)
    half_w_m_var = student_t_crit * sem_m_var
    half_w_m_nll = student_t_crit * sem_m_nll
    return (mean_m_vars, half_w_m_var), (mean_m_nlls, half_w_m_nll), m_var_nll_samples

# Computes the exchangeability - this is the function to be called when computing exchangeability
def exchangeability_test(models, data, no_permutations=20, device='cuda', use_autoreg_eq=False, max_samples=200, seq_len=100, batch_size=16):
    assert no_permutations >= 2, "Must have at least 2 permutations to compute variance"
    data.batch_size=batch_size
    val_loader = torch.utils.data.DataLoader(
        data,
        batch_size=None,
        num_workers=experiment.misc.num_val_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_val_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_val_workers > 0 else False,
        pin_memory=True,
    )
    start_time = time.time()
    # Logs exchangeability
    (mean_m_var, half_w_m_var), (mean_m_nlls, half_w_m_nll), _ = exchange(models, val_loader, no_permutations, device, use_autoreg_eq, max_samples, seq_len)
    end_time = time.time()
    if half_w_m_var is None: half_w_m_var = 'N/A'
    if half_w_m_nll is None: half_w_m_nll = 'N/A'
    print("-----------")
    print(f"Exchangeability test time: {end_time - start_time:.4f} seconds")
    print(f"Exchangeability (eq 5): {mean_m_var} +/- {half_w_m_var}")
    print(f"NLL (eq 6) - mean: {mean_m_nlls} +/- {half_w_m_nll}")
    print("-----------")

# Converts string to file that can be written safely with regex
def _slug(s: str) -> str:
    return re.sub(r"[^\w\-\.]", "_", s)

def get_plot_rbf(nc, nt, samples_per_epoch, batch_size):
    # Data loader - RBF kernel in this case
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    noise_std=0.1
    deterministic = True

    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=nc, max_nc=nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    return gen_val

# Attempts to recreate something like figure 2
def plot_models_setup_rbf_same():
    # Defines each model
    tnp_plain = ['experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml', 'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-a3qwpptn:v200', "TNP-D"]
    inc_tnp = ['experiments/configs/synthetic1dRBF/gp_causal_tnp_rangesame.yml', 'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v200', "incTNP"]
    inc_tnp_batched=['experiments/configs/synthetic1dRBF/gp_batched_causal_tnp_rbf_rangesame.yml', 'pm846-university-of-cambridge/mask-batched-tnp-rbf-rangesame/model-xtnh0z37:v200', "incTNP-Batched"]
    models_tnp = [tnp_plain, inc_tnp, inc_tnp_batched]

    tnp_ar_cptk, tnp_ar_yml, tnp_name = 'experiments/configs/synthetic1dRBF/gp_tnpa_rangesame.yml', 'pm846-university-of-cambridge/tnpa-rbf-rangesame/model-wbgdzuz5:v200', "TNP-A"
    tnp_ar_100 = [tnp_ar_cptk, tnp_ar_yml, tnp_name, 100]
    tnp_ar_50 = [tnp_ar_cptk, tnp_ar_yml, tnp_name, 50]
    tnp_ar_10 = [tnp_ar_cptk, tnp_ar_yml, tnp_name, 50]
    tnp_ar_20 = [tnp_ar_cptk, tnp_ar_yml, tnp_name, 20]
    models_ar = [tnp_ar_10, tnp_ar_20]

    gp_name = "GP-Expanding"
    gp_streamed_expanding_1 = ["", "", gp_name, 1, "Expanding"]
    gp_streamed_expanding_2 = ["", "", gp_name, 2, "Expanding"]
    gp_streamed_expanding_4 = ["", "", gp_name, 4, "Expanding"]
    gp_streamed_expanding_8 = ["", "", gp_name, 8, "Expanding"]
    gp_streamed_expanding_16 = ["", "", gp_name, 16, "Expanding"]
    gp_streamed_expanding_32 = ["", "", gp_name, 32, "Expanding"]
    gp_streamed_sliding_1 = ["", "", gp_name, 1, "Sliding"] # Sliding window now
    gp_streamed_sliding_2 = ["", "", gp_name, 2, "Sliding"]
    gp_streamed_sliding_4 = ["", "", gp_name, 4, "Sliding"]
    gp_streamed_sliding_8 = ["", "", gp_name, 8, "Sliding"]
    gp_streamed_sliding_16 = ["", "", gp_name, 16, "Sliding"]
    gp_streamed_sliding_32 = ["", "", gp_name, 32, "Sliding"]
    models_gp = [gp_streamed_expanding_1, gp_streamed_expanding_2, gp_streamed_expanding_4, gp_streamed_expanding_8, gp_streamed_expanding_16, gp_streamed_expanding_32,
        gp_streamed_sliding_1, gp_streamed_sliding_2, gp_streamed_sliding_4, gp_streamed_sliding_8, gp_streamed_sliding_16, gp_streamed_sliding_32]
    models_gp_expanding = [gp_streamed_expanding_4, gp_streamed_expanding_8, gp_streamed_expanding_16]
    models_gp_sliding = [gp_streamed_sliding_4, gp_streamed_sliding_8, gp_streamed_sliding_16]

    gp_sparse_name = "Streamed Sparse GP"
    gp_sparse_1 = ["", "", gp_sparse_name, 1, ""]
    gp_sparse_2 = ["", "", gp_sparse_name, 2, ""]
    gp_sparse_4 = ["", "", gp_sparse_name, 4, ""]
    gp_sparse_8 = ["", "", gp_sparse_name, 8, ""]
    gp_sparse_16 = ["", "", gp_sparse_name, 16, ""]
    gp_sparse_32 = ["", "", gp_sparse_name, 32, ""]

    models_all_no_ar = models_tnp + models_gp
    models_all = models_tnp + models_gp + models_ar
    #return [tnp_plain, inc_tnp, inc_tnp_batched]
    return [tnp_ar_10]

def extract_vars_from_folder_name(folder_name):
    patterns = {
        'nc': r'nc_(\d+)',
        'nt': r'nt_(\d+)',
        'no_permutations': r'np_(\d+)',
        'samples_per_epoch': r'spe_(\d+)',
        'batch_size': r'bs_(\d+)',
        'max_samples': r'ms_(\d+)',
        'return_samples': r'rs_(\d+)',
        'use_autoreg_eq': r'uae_(True|False)'
    }
    variables_found = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, folder_name)
        assert match, "Invalid folder name no match found"
        value_str_found = match.group(1)
        # Sorts types
        if key == 'use_autoreg_eq': variables_found[key] = (value_str_found == 'True')
        else: variables_found[key] = int(value_str_found)
    return variables_found

def generate_folder_name(nc, nt, samples_per_epoch, no_permutations, batch_size, use_autoreg_eq, max_samples, return_samples):
    file_str = f'nc_{nc}_nt_{nt}_np_{no_permutations}_spe_{samples_per_epoch}_bs_{batch_size}_ms_{max_samples}_rs_{return_samples}_uae_{use_autoreg_eq}'
    return file_str

# Takes a folder with data written and plots the fig
def plot_from_folder(folder):
    # Plot hypers
    max_samples_plot = 10 # Max number of samples
    filter_ugly_thres = 0.005 # Values not to include on plot
    max_dist_from_centroid = 100
    # End of plot hypers

    pars = extract_vars_from_folder_name(folder)
    nc, nt = pars["nc"], pars["nt"]

    data_directory = Path(folder)
    model_folders_unstr = [p for p in data_directory.iterdir() if p.is_dir()]
    # Imposes plot ordering
    remaining_folders = set(model_folders_unstr)
    order = ["TNP-D", "incTNP", "incTNP-Batched", "TNP-A", "GP-Expanding"]
    model_folders = []
    for prefix in order:
        matches = [p for p in remaining_folders if p.name.startswith(prefix)]
        if matches:
            matches.sort(key=lambda p: p.name)
            model_folders.extend(matches)
            remaining_folders.difference_update(matches)
    if remaining_folders:
        model_folders.extend(sorted(list(remaining_folders), key=lambda p: p.name))

    # Colour pallete to use - sensible but pretty
    tableau_colorblind_10 = ['#006BA4','#FF800E','#ABABAB','#595959','#5F9ED1','#C85200','#898989','#A2C8EC','#FFBC79','#CFCFCF']
    colours = cycle(tableau_colorblind_10)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    # Stores all plotted points to calculate graph limits
    all_xs = []
    all_ys = []
    # Loops through models and plots them
    for (model_folder, colour) in zip(model_folders, colours):
        with open(model_folder / 'summary.txt', 'r', encoding='utf-8') as f:
            model_summary_txt = f.read()
        # Extracts from the summary fixed format
        lines = model_summary_txt.split("\n")
        model_name = lines[1].split(": ")[1]
        if model_name.startswith("Streamed GP-S"): continue
        mean_m_var = float(lines[2].split(" ")[1])
        mean_m_nlls = float(lines[3].split(" ")[1])
        npz_file = lines[4].split(": ")[1]
        data = np.load(model_folder / npz_file)
        samples_m_var = data["samples_m_var"]
        samples_m_nll = data["samples_m_nll"]

        # Hacky code - filters out ugly examples
        print(len(samples_m_nll))
        idx_to_rem = [i for i in range(len(samples_m_nll)) if samples_m_nll[i] <= filter_ugly_thres]
        samples_m_nll = [samples_m_nll[i] for i in range(len(samples_m_nll)) if i not in idx_to_rem]
        samples_m_var = [samples_m_var[i] for i in range(len(samples_m_var)) if i not in idx_to_rem]
        samples_m_nll = samples_m_nll[:max_samples_plot] if len(samples_m_nll) >= max_samples_plot else samples_m_nll
        samples_m_var = samples_m_var[:max_samples_plot] if len(samples_m_var) >= max_samples_plot else samples_m_var


        all_xs.extend(samples_m_var)
        all_xs.append(mean_m_var)
        all_ys.extend(samples_m_nll)
        all_ys.append(mean_m_nlls)

        # Plots small dots
        ax.scatter(
            samples_m_var,
            samples_m_nll,
            s=50,
            c=[colour],
            alpha=1.0,
            marker='o',
            edgecolors='none',
            zorder=2,
        )

        # Line to centroid dot
        for (sx, sy) in zip(samples_m_var, samples_m_nll):
            ax.plot([mean_m_var, sx], [mean_m_nlls, sy], lw=1.6, c=colour, alpha=1.0,zorder=1)

        # Plots large dot
        ax.scatter(
            mean_m_var,
            mean_m_nlls,
            s=200,
            c=[colour],
            alpha=1.0,
            marker='o',
            edgecolors='none',
            label=model_name,
            zorder=3,
        )

    # Calculates log limits for graphs - will probably break in case of 0 / neg values (which can occurr but is prolly unlikely)
    min_x_log = np.floor(np.log10(min(all_xs)))
    max_x_log = np.ceil(np.log10(max(all_xs)))
    min_y_log = np.floor(np.log10(min(all_ys)))
    max_y_log = np.ceil(np.log10(max(all_ys)))
    ax.set_xlim(10**min_x_log, 10**max_x_log)
    ax.set_ylim(10**min_y_log, 10**max_y_log)

    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 8))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel("Joint Log-Likelihood Variance")
    ax.set_ylabel("Neg. Joint Log-Likelihood Mean (- Optimal)")

    # Tick params
    ax.tick_params(axis='both', which='major', length=4, width=0.8)

    ax.legend()

    ax.set_title(f"Exchangeability vs Performance (NC={nc}, NT={nt})")

    plt.savefig(folder + "/brunofig.png", bbox_inches="tight")

# Attempts to recreate something like figure 2. Use plot_models_setup as helper for this func.
def gather_stats_models(helper_tuple, base_folder_name):
    # Exchange hyperparams
    nc, nt = 10, 10 
    samples_per_epoch = 4096 # How many datapoints in datasets
    no_permutations=64
    batch_size = 128
    use_autoreg_eq=False
    max_samples=samples_per_epoch
    return_samples=max_samples # essentially return as many as possible (but one per batch)
    skip_existing_folders = True # Skips existing file writes - no need to do work again
    # End of hypers

    (models) = helper_tuple
    seq_len = nc + nt

    # Ensures base and data folder already exists
    base_path = Path(base_folder_name)
    base_path.mkdir(exist_ok=True)
    data_folder = base_folder_name + f"/{generate_folder_name(nc=nc, nt=nt, samples_per_epoch=samples_per_epoch, no_permutations=no_permutations, use_autoreg_eq=use_autoreg_eq, max_samples=max_samples, return_samples=return_samples,batch_size=batch_size)}"
    data_path = Path(data_folder)
    data_path.mkdir(exist_ok=True)

    for mod_data in models:
        mod_cptk, mod_yml, model_name = mod_data[0], mod_data[1], mod_data[2]
        # Formats model names
        if model_name == "GP-Expanding":
            _, _, name_base, chunk_size, strat = mod_data
            gp_ext = "" if strat == "Expanding" else ""
            model_name_fmt = model_name + gp_ext+ f' (ch={chunk_size})'
        elif model_name == "TNP-A":
            model_name_fmt = model_name + f' ({mod_data[3]} samples)'
        elif model_name == "Streamed Sparse GP":
            _, _, name_base, chunk_size, strat = mod_data
            model_name_fmt = model_name + f' (ch={chunk_size})'
        else: model_name_fmt = model_name
        print(model_name_fmt)

        # Checks to see if exact model run has already been written (no need to do again if it has)
        model_folder = data_folder + "/" + _slug(model_name_fmt)
        if skip_existing_folders and os.path.exists(model_folder): continue

        use_torch_grad = False
        # Handles GP case seperately
        if model_name == "GP-Expanding":
            _, _, name_base, chunk_size, strat = mod_data
            model = GPStreamRBF(chunk_size=chunk_size, train_strat=strat)
            use_torch_grad = True # Need gradient to train the GP model
        elif model_name == "TNP-A":
            model = get_model(mod_cptk, mod_yml)
            model.num_samples = mod_data[3]
            model.eval()
        elif model_name == "Streamed Sparse GP":
            assert False, "Unspported model"
            #_, _, _, chunk_size, _ = mod_data
            #model = GPStreamSparseWrapperRBF(num_inducing=nc, chunk_size=chunk_size)
            #use_torch_grad = True
        else:
            model = get_model(mod_cptk, mod_yml)
            model.eval()

        (mean_m_var, _,), (mean_m_nlls, _), m_var_nll_samples = exchange([model], get_plot_rbf(nc, nt, samples_per_epoch, batch_size=batch_size), no_permutations=no_permutations, device='cuda', use_autoreg_eq=use_autoreg_eq, max_samples=max_samples, seq_len=seq_len, return_samples=return_samples, use_torch_grad=use_torch_grad)
        samples_m_var = [x[0].item() for x in m_var_nll_samples]
        samples_m_nll = [x[1].item() for x in m_var_nll_samples]

        # Writes recorded results to folder
        model_folder_path = Path(model_folder)
        model_folder_path.mkdir(exist_ok=True)
        samples_m_var_np = np.array(samples_m_var)
        samples_m_nll_np = np.array(samples_m_nll)
        rel_sample_name = "samples.npz"
        save_samples_path = model_folder + "/" + rel_sample_name
        summary_block = ("-" * 20) + "\n" + f"Model_Name: {model_name_fmt}\nMean_M_Var: {mean_m_var}\nMean_M_NLL: {mean_m_nlls}\nSamples_File (samples_m_var and samples_m_nll): {rel_sample_name}"
        print(summary_block)
        np.savez_compressed(save_samples_path, samples_m_var=samples_m_var_np, samples_m_nll=samples_m_nll_np)
        with open(model_folder + '/summary.txt', 'w') as file_object:
            file_object.write(summary_block)

    plot_from_folder(data_folder) # Plots generated data



if __name__ == "__main__":
    gather_stats_models(plot_models_setup_rbf_same(), base_folder_name="experiments/plot_results/exchange/resexc")
    exit(0)
    # E.g. run with: python experiments/exchangeability.py --config experiments/configs/synthetic1d/gp_plain_tnp.yml
    experiment = initialize_experiment() # Gets config file
    model_arch = experiment.model # Gets type of model
    # RBF kernel params
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    # Data generator params
    nc, nt = 32, 64 
    batch_size = 16
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 16_000
    batch_size = 1024
    deterministic = True
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=nc, max_nc=nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=0.1,
        deterministic=True, kernel=kernels)
    models = []
    useWandb = True # Defines if weights and biases model is to be used
    #wanddName = 'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-7ib3k6ga:v200'
    wanddName = 'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v200'
    if useWandb:
        artifact = wandb.Api().artifact(wanddName, type='model')
        artifact_dir = artifact.download()
        ckpt_file = os.path.join(artifact_dir, "model.ckpt")
        print(ckpt_file)
        lit_model = (
            LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                ckpt_file, model=model_arch,
            )
        )
        tnp_model = lit_model.model
        tnp_model.eval()
        models.append(tnp_model)
    else:
        model_arch.to('cuda')
        model_arch.eval()
        tnp_model=model_arch
        models.append(tnp_model)

    exchangeability_test(models, gen_val, no_permutations=20, device='cuda', use_autoreg_eq=False, max_samples = 100, seq_len = nc+nt, batch_size=16)
