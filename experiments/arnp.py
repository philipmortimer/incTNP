# Autoregressive neural process - test time only.
# Based on https://arxiv.org/pdf/2303.14468 - takes a normal NP model and treats predicted target points as context points
# Inspired by https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses
import torch
from check_shapes import check_shapes
from torch import nn
from typing import Optional, Union, Literal, Callable, Tuple
from tnp.utils.np_functions import np_pred_fn
from tnp.data.base import Batch
from tnp.models.incUpdateBase import IncUpdateEff, IncUpdateEffFixed
from plot_adversarial_perms import get_model
from tnp.data.gp import RandomScaleGPGenerator
from tnp.networks.gp import RBFKernel
from functools import partial
from tqdm import tqdm
import numpy as np
import torch.distributions as td
from plot import plot
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.titlesize"]= 14


@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]",
)
@torch.no_grad
def _shuffle_targets(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: Optional[torch.Tensor],
    order: Literal["random", "given", "left-to-right", "variance"]):
    m, nt, dx = xt.shape
    _, _, dy = yc.shape
    device = xt.device
    if order == "given":
        perm = torch.arange(nt, device=device).repeat(m, 1)
        return xt, yt, perm
    elif order == "random":
        perm = torch.rand(m, nt, device=device).argsort(dim=1)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_shuffled = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_shuffled = torch.gather(yt, 1, perm_y)
        else: yt_shuffled = None
        return xt_shuffled, yt_shuffled, perm
    elif order == "left-to-right":
        assert dx == 1, "left-to-right ordering only supported for one dimensional dx"
        perm = torch.argsort(xt.squeeze(-1), dim=1)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_sorted = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_sorted = torch.gather(yt, 1, perm_y)
        else: yt_sorted = None
        return xt_sorted, yt_sorted, perm
    elif order == "variance":
        # Predicts all target points conditioned on context points and orders (highest variance first) - this is obviously much more expensive
        batch = Batch(xc=xc, yc=yc, xt=xt, yt=None, x=None, y=None)
        pred_dist = np_pred_fn(np_model, batch)
        var = pred_dist.variance.mean(-1) # Gets variance (averaged over dy) [m, nt]
        perm = torch.argsort(var, dim=1, descending=True)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_sorted = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_sorted = torch.gather(yt, 1, perm_y)
        else: yt_sorted = None
        return xt_sorted, yt_sorted, perm



@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]", "return: [m]"
)
@torch.no_grad
def ar_loglik(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor,
    normalise: bool = True, order: Literal["random", "given", "left-to-right", "variance"] = "random") -> torch.Tensor:
    xt, yt, _ = _shuffle_targets(np_model, xc, yc, xt, yt, order)
    np_model.eval()
    m, nt, dx = xt.shape
    _, nc, dy = yc.shape
    log_probs = torch.zeros((m), device=xt.device)
    for i in range(nt):
        # Sets context and target
        xt_sel = xt[:,i:i+1,:]
        yt_sel = yt[:,i:i+1,:]
        xc_it = torch.cat((xc, xt[:, :i, :]), dim=1)
        yc_it = torch.cat((yc, yt[:, :i, :]), dim=1)
        batch = Batch(xc=xc_it, yc=yc_it, xt=xt_sel, yt=yt_sel, x=torch.cat((xc_it, xt_sel), dim=1), y=torch.cat((yc_it, yt_sel), dim=1))

        # Prediction + log prob
        pred_dist = np_pred_fn(np_model, batch)
        log_probs += pred_dist.log_prob(yt_sel).sum(dim=(-1, -2))
    if normalise:
        log_probs /= (nt * dy)
    return log_probs


@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]"
)
@torch.no_grad
def ar_predict(model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor,
    order: Literal["random", "given", "left-to-right", "variance"] = "random",
    num_samples: int = 10,
    prioritise_fixed: bool = False, # If incremental updates are available prioritise fixed or true dynamic algorithm
    device: str = "cuda", # Device for computing
    device_ret: str = "cpu", # Return device
    use_flash: bool = False, # Use flash attention if possible? - experimental
    ):
    m, nt, dx = xt.shape
    _, nc, dy = yc.shape
    xc, yc, xt = xc.to(device), yc.to(device), xt.to(device)

    xc_stacked = xc.repeat_interleave(num_samples, dim=0)
    yc_stacked = yc.repeat_interleave(num_samples, dim=0)
    xt_stacked = xt.repeat_interleave(num_samples, dim=0)

    xt_stacked, _, perm = _shuffle_targets(model, xc_stacked, yc_stacked, xt_stacked, None, order) # Should I shuffle before or after stacking?

    yt_preds_mean, yt_preds_std = torch.empty((m * num_samples, nt, dy), device=device), torch.empty((m * num_samples, nt, dy), device=device)

    is_fixed_inc_update = isinstance(model, IncUpdateEffFixed)
    is_inc_gen_update = isinstance(model, IncUpdateEff)
    is_fixed_inc_update = (is_fixed_inc_update and prioritise_fixed) or (is_fixed_inc_update and not is_inc_gen_update)
    is_inc_gen_update = (is_inc_gen_update and not prioritise_fixed) or (is_inc_gen_update and not is_fixed_inc_update)
    assert is_fixed_inc_update != is_inc_gen_update or (not is_fixed_inc_update and not is_inc_gen_update), "Xor onf fixed vs inc update"
    if is_inc_gen_update:
        model.init_inc_structs(m=xc_stacked.shape[0], max_nc=nc+nt, device=device, use_flash=use_flash)
        model.update_ctx(xc=xc_stacked, yc=yc_stacked,use_flash=use_flash)
    elif is_fixed_inc_update:
        model.init_inc_structs_fixed(m=xc_stacked.shape[0], max_nc=nc+nt, xt=xt_stacked, dy=dy, device=device,use_flash=use_flash)

    for i in range(nt):
        xt_tmp = xt_stacked[:, i:i+1,:]
        if is_inc_gen_update:
            pred_dist = model.query(xt=xt_tmp, dy=dy,use_flash=use_flash)
        elif is_fixed_inc_update:
            pred_dist = model.query_fixed(tgt_start_ind=i, tgt_end_ind=i+1, use_flash=use_flash)
        else:
            batch = Batch(xc=xc_stacked, yc=yc_stacked, xt=xt_tmp, yt=None, x=None, y=None)
            pred_dist = np_pred_fn(model, batch)
        assert isinstance(pred_dist, td.Normal), "Must predict a gaussian"
        pred_mean, pred_std = pred_dist.mean, pred_dist.stddev
        yt_preds_mean[:,i:i+1,:] = pred_mean
        yt_preds_std[:,i:i+1,:] = pred_std
        # Samples from the predictive distribution and updates the context
        if i < nt - 1:
            yt_sampled = pred_dist.sample() # [m * num_samples, 1, dy]
            if is_inc_gen_update:
                model.update_ctx(xc=xt_tmp, yc=yt_sampled, use_flash=use_flash)
            elif is_fixed_inc_update:
                model.update_ctx_fixed(xc=xt_tmp, yc=yt_sampled, use_flash=use_flash)
            else:
                xc_stacked = torch.cat((xc_stacked, xt_tmp), dim=1)
                yc_stacked = torch.cat((yc_stacked, yt_sampled), dim=1)
                
    # Unshuffles the target ordering to be in line with what was passed in
    inv_perm = perm.argsort(dim=1)
    idx = inv_perm.unsqueeze(-1).expand(-1, -1, dy)
    yt_preds_mean = yt_preds_mean.gather(dim=1, index=idx)
    yt_preds_std = yt_preds_std .gather(dim=1, index=idx)

    yt_preds_mean = yt_preds_mean.view(num_samples, m, nt, dy)
    yt_preds_std = yt_preds_std.view(num_samples, m, nt, dy)
    # Permutes to [m, nt, dy, num_samples]
    yt_preds_mean = yt_preds_mean.permute(1,2,3,0)
    yt_preds_std = yt_preds_std.permute(1,2,3,0)
    mix = td.Categorical(torch.full((m, nt, dy, num_samples), 1.0 / num_samples, device=device_ret))
    comp = td.Normal(yt_preds_mean.to(device_ret), yt_preds_std.to(device_ret))
    approx_dist = td.MixtureSameFamily(mix, comp)

    # For sample draws return raw samples and run through model again for smooth samples (see paper / code)
    return approx_dist



# -------------------------------------------------------------------------------------------------------

# Plots performance of select models with varying nc, nt, s
def plot_rmse_predict_vs_time():
    # Would be cool to plot like figure 7 of LBANP paper but with runtime vs rmse
    # only worry is that to see big O runtime changes. Also RBF kernel could be bad example
    # since quite simple. But would be really great to see like whole spectrum
    # of TNP-D, incTNP-Batched, convCNP and CNP as a spectrum of performance vs time.
    device="cuda"
    out_folder = "experiments/plot_results/ar/rmse/"
    burn_in = 1
    order="random"
    aggregate_over = 1
    max_batch = 20
    prioritise_fixed = False
    tnp_plain = ('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml',
        'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-a3qwpptn:v200', 'TNP-D')
    incTNP = ('experiments/configs/synthetic1dRBF/gp_causal_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-8mxfyfnw:v200', 'incTNP')
    batchedTNP = ('experiments/configs/synthetic1dRBF/gp_batched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-rbf-rangesame/model-xtnh0z37:v200', 'incTNP-Batched')
    priorBatched = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'incTNP-Batched (Prior)')
    cnp = ('experiments/configs/synthetic1dRBF/gp_cnp_rangesame.yml',
        'pm846-university-of-cambridge/cnp-rbf-rangesame/model-uywfyrx7:v200', 'CNP')
    conv_cnp = ('experiments/configs/synthetic1dRBF/gp_convcnp_rangesame.yml',
        'pm846-university-of-cambridge/convcnp-rbf-rangesame/model-uj54q1ya:v200', 'ConvCNP')
    models =[tnp_plain, incTNP, batchedTNP, priorBatched, cnp, conv_cnp]
    # Number of samples
    samples = [1, 5, 10, 20, 30, 40, 50, 100, 200]
    runtime = np.zeros((len(models), len(samples)))
    memory = np.zeros((len(models), len(samples)))
    rmse = np.zeros((len(models), len(samples)))
    ll = np.zeros((len(models), len(samples)))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    data = get_rbf_rangesame_testset()
    summary_txt = ""
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        for sample_idx, num_samples in enumerate(samples):
            batch_runtimes = []
            batch_memories = []
            batch_rmses = []
            batch_lls = []
            for batch_idx, batch in tqdm(enumerate(data), desc=f's={num_samples} mod={model_name}'):
                if max_batch is not None and batch_idx >= max_batch: break
                run_runtimes = []
                run_memories = []
                run_rmses = []
                run_lls = []
                for j in range(burn_in+aggregate_over):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    starter.record()
                    with torch.no_grad():
                        pred_dist = ar_predict(model=model, xc=batch.xc, yc=batch.yc, xt=batch.xt, order=order, num_samples=num_samples,
                            device=device, device_ret=device, prioritise_fixed=prioritise_fixed)
                    # Measures runtime
                    ender.record()
                    torch.cuda.synchronize()
                    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    runtime_ms = starter.elapsed_time(ender)
                    loglik = (pred_dist.log_prob(batch.yt.to(device)).sum() / batch.yt[..., 0].numel()).item()
                    rmse_point = nn.functional.mse_loss(pred_dist.mean, batch.yt.to(device)).sqrt().cpu().mean()
                    if j >= burn_in:
                        run_runtimes.append(runtime_ms)
                        run_memories.append(peak_memory_mb)
                        run_rmses.append(rmse_point)
                        run_lls.append(loglik)
                # Aggregates metrics
                batch_runtimes.append(np.mean(run_runtimes))
                batch_memories.append(np.mean(run_memories))
                batch_rmses.append(np.mean(run_rmses))
                batch_lls.append(np.mean(run_lls))
            runtime[model_idx, sample_idx] = np.mean(batch_runtimes)
            rmse[model_idx, sample_idx] = np.mean(batch_rmses)
            memory[model_idx, sample_idx] = np.mean(batch_memories)
            ll[model_idx, sample_idx] = np.mean(batch_lls)
            model_samp_summary = ("*" * 20) + f'\nModel: {model_name}\nRuntime(ms): {runtime[model_idx, sample_idx]}\nSamples: {num_samples}\nAverageMemoryUse(MB): {memory[model_idx, sample_idx]}\nRMSEMean: {rmse[model_idx, sample_idx]}\nLL: {ll[model_idx, sample_idx]}\n'
            summary_txt += model_samp_summary
            print(model_samp_summary)
    # Writes model data to output files
    with open(out_folder + 'summary_rmse.txt', 'w') as file_obj:
        file_obj.write(summary_txt)


# Measures timings of different models
def measure_perf_timings():
    # Measure hypers
    burn_in = 1 # Number of burn in runs to ignore
    aggregate_over = 1 # Number of runs to aggregate data over
    token_step = 200 # How many increments of tokens to go up in
    min_nt, max_nt = 1, 2003
    dx, dy, m = 1, 1, 1
    nc_start = 1
    num_samples=50 # Samples to unroll in ar_predict
    device = "cuda"
    order="random"
    prioritise_fixed = False
    plot_name_folder = "experiments/plot_results/ar/perf/"
    use_flash = True # whether to use flash for ar inctnp
    # End of measure hypers
    models = get_model_list()
    max_high = 2
    xc = (torch.rand((m, nc_start, dx), device=device) * max_high * 2) - max_high
    yc = (torch.rand((m, nc_start, dy), device=device) * max_high * 2) - max_high
    target_sizes = np.arange(start=min_nt, stop=max_nt, step=token_step, dtype=int)
    runtime = np.zeros((len(models), aggregate_over, len(target_sizes)))
    memory = np.zeros((len(models), aggregate_over, len(target_sizes)))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval() 
        for t_index, nt in tqdm(enumerate(target_sizes), desc=f'Targ {model_name}'):
            xt = (torch.rand((m, nt, dx), device=device) * max_high * 2) - max_high
            yt = (torch.rand((m, nt, dy), device=device) * max_high * 2) - max_high

            for j in range(burn_in + aggregate_over):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                starter.record()
                if use_flash and model_name == "incTNP":
                    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                        pred_dist = ar_predict(model=model, xc=xc, yc=yc, xt=xt, order=order, num_samples=num_samples,
                        device=device, device_ret=device, prioritise_fixed=prioritise_fixed, use_flash=True)
                else:
                    with torch.no_grad():
                        pred_dist = ar_predict(model=model, xc=xc, yc=yc, xt=xt, order=order, num_samples=num_samples,
                            device=device, device_ret=device, prioritise_fixed=prioritise_fixed, use_flash=False)
                # Measures time and memory
                ender.record()
                torch.cuda.synchronize()
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                runtime_ms = starter.elapsed_time(ender)
                # Stores results
                write_idx = j - burn_in
                if write_idx >= 0:
                    runtime[model_idx, write_idx, t_index] = runtime_ms
                    memory[model_idx, write_idx, t_index] = peak_memory_mb
    # Aggregates results
    runtime = np.mean(runtime, axis=1) # [no_models, len(target_sizes)]
    memory = np.mean(memory, axis=1)
    # Plots runtime
    runtime_file_name = plot_name_folder + f'runtime_od_{order}_samples_{num_samples}_nc{nc_start}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        ax.plot(target_sizes, runtime[model_idx] / 1000.0, label=model_name)
    ax.set_xlabel('Target Size')
    ax.set_ylabel('Runtime (s)')
    ax.legend()
    ax.set_title(f'Runtime of AR NPs (S={num_samples} NC={nc_start})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(runtime_file_name, dpi=300)
    # Plots memory
    memory_file_name = plot_name_folder + f'memory_od_{order}_samples_{num_samples}_nc{nc_start}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        ax.plot(target_sizes, memory[model_idx], label=model_name)
    ax.set_xlabel('Target Size')
    ax.set_ylabel('Memory Usage (MB)')
    ax.legend()
    ax.set_title(f'Memory Usage of AR NPs (S={num_samples} NC={nc_start})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(memory_file_name, dpi=300)



# Plots a handful of kernels
def plot_ar_unrolls():
    # Hypers
    order="random"
    #no_samples = [1, 2, 5, 10, 50, 100, 500, 1000]
    no_samples = [10, 50]
    folder_name = "experiments/plot_results/ar/plots/"
    no_kernels = 5#20
    device="cuda"
    # End of hypers
    models = get_model_list()
    data = get_rbf_rangesame_testset()
    for (model_yml, model_wab, model_name) in models:
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        model_folder = f"{folder_name}/{model_name}"
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        for sample in no_samples:
            def pred_fn_pred(model, batch, predict_without_yt_tnpa=True):
                return ar_predict(model, batch.xc, batch.yc, batch.xt, order, sample, device=device)

            plot(model=model, batches=data, num_fig=min(no_kernels, len(data)), name=model_folder+f"/ns_{sample}_od_{order}",
                savefig=True, logging=False, pred_fn=pred_fn_pred, x_range = (-2.0, 2.0),
                model_lbl=f"AR {model_name} (S={sample}) ")
                


def get_rbf_rangesame_testset():
    # RBF Dataset
    min_nc = 1
    max_nc = 64
    nt= 128
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 4_096
    batch_size = 16
    noise_std = 0.1
    deterministic = True
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    gen_test = RandomScaleGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    data = list(gen_test)
    return data

def get_model_list():
    # List of models to compare
    tnp_plain = ('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml',
        'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-a3qwpptn:v200', 'TNP-D')
    incTNP = ('experiments/configs/synthetic1dRBF/gp_causal_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-8mxfyfnw:v200', 'incTNP')
    batchedTNP = ('experiments/configs/synthetic1dRBF/gp_batched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-rbf-rangesame/model-xtnh0z37:v200', 'incTNP-Batched')
    priorBatched = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'incTNP-Batched (Prior)')
    cnp = ('experiments/configs/synthetic1dRBF/gp_cnp_rangesame.yml',
        'pm846-university-of-cambridge/cnp-rbf-rangesame/model-uywfyrx7:v200', 'CNP')
    conv_cnp = ('experiments/configs/synthetic1dRBF/gp_convcnp_rangesame.yml',
        'pm846-university-of-cambridge/convcnp-rbf-rangesame/model-uj54q1ya:v200', 'ConvCNP')
    models = [tnp_plain, incTNP, batchedTNP, priorBatched, conv_cnp, cnp]
    models = [tnp_plain, incTNP, conv_cnp, cnp]
    models = [incTNP]
    return models

# Compares NP models in AR mode on RBF set
def compare_rbf_models(base_out_txt_file: str, device: str = "cuda"):
    # Hypers to select - also look at dataset hypers
    ordering = "random"
    # End of hypers
    # Main loop - loads each model than compares writes performances to a text file
    models = get_model_list()
    data = get_rbf_rangesame_testset()
    out_txt = ""
    for (model_yml, model_wab, model_name) in models:
        ll_list = []
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        for batch in tqdm(data, desc=f'{model_name} eval'):
            ll = ar_loglik(np_model=model, xc=batch.xc.to(device), yc=batch.yc.to(device),
                xt=batch.xt.to(device), yt=batch.yt.to(device), normalise=True, order=ordering)
            mean_ll = torch.mean(ll).item() # Goes from [m] to a float
            ll_list.append(mean_ll)
        ll_average = np.mean(ll_list)
        mod_sum = ("-" * 20) + f"\nModel: {model_name}\nMean LL: {ll_average}\n"
        print(mod_sum)
        out_txt += mod_sum
    with open(base_out_txt_file + f'_{ordering}.txt', 'w') as file:
        file.write(out_txt)


if __name__ == "__main__":
    #plot_rmse_predict_vs_time()
    measure_perf_timings()
    #plot_ar_unrolls()
    #compare_rbf_models(base_out_txt_file="experiments/plot_results/ar/ar_rbf_comp")