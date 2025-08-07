# Plots performance of models as they get increasingly more context - including beyond the original size seen in training
from tnp.data.gp import RandomScaleGPGenerator
from tnp.networks.gp import RBFKernel
from tnp.networks.gp import MaternKernel
from tnp.networks.gp import PeriodicKernel
from functools import partial
from plot_adversarial_perms import get_model
import numpy as np
from tnp.models.incUpdateBase import IncUpdateEff
from tnp.data.base import Batch
from tnp.utils.np_functions import np_pred_fn
import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from arnp import ar_loglik
import json
from tnp.utils.data_loading import adjust_num_batches


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.titlesize"]= 14

def get_rbf_rangesame_testset(nc: int, nt: int, batch_size: int):
    # RBF Dataset
    min_nc = nc
    max_nc = nc
    nt = nt
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 4_096
    batch_size = batch_size
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
    #data = list(gen_test)

    val_workers = 3
    test_loader = torch.utils.data.DataLoader(
        gen_test,
        batch_size=None,
        num_workers=val_workers,
        worker_init_fn=(
            (
                adjust_num_batches
            )
            if val_workers > 0
            else None
        ),
        persistent_workers=True if val_workers > 0 else False,
        pin_memory=True,
    )
    return test_loader, "RBF Kernel", samples_per_epoch

# Gets combined dataset with specific number of context points
def get_combined_rangesame_testset(nc: int, nt: int, batch_size: int):
    # RBF Dataset
    min_nc = nc
    max_nc = nc
    nt= 128
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 4_096
    noise_std = 0.1
    deterministic = True
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    min_log10_period = 0.301
    max_log10_period = 0.301
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    matern12_kernel_factory = partial(MaternKernel, nu=0.5, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    matern32_kernel_factory = partial(MaternKernel, nu=1.5, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    matern52_kernel_factory = partial(MaternKernel, nu=2.5, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    periodic_kernel_factory = partial(PeriodicKernel, min_log10_period=min_log10_period, max_log10_period=max_log10_period, 
            ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale, max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory, matern12_kernel_factory, matern32_kernel_factory, matern52_kernel_factory, periodic_kernel_factory]
    gen_test = RandomScaleGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)

    val_workers = 3
    test_loader = torch.utils.data.DataLoader(
        gen_test,
        batch_size=None,
        num_workers=val_workers,
        worker_init_fn=(
            (
                adjust_num_batches
            )
            if val_workers > 0
            else None
        ),
        persistent_workers=True if val_workers > 0 else False,
        pin_memory=True,
    )
    #data = list(gen_test)
    return test_loader, "Combined Kernel", samples_per_epoch

def get_model_list_combined():
    # List of models to compare trained on combined kernel
    tnp_plain = ('experiments/configs/synthetic1d/gp_plain_tnp_rangesame.yml',
        'pm846-university-of-cambridge/plain-tnp-rangesame/model-fyr9u053:v200', 'TNP-D', False)
    incTNP = ('experiments/configs/synthetic1d/gp_causal_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/mask-tnp-rangesame/model-l69k9pix:v200', 'incTNP', False)
    batchedTNP = ('experiments/configs/synthetic1d/gp_batched_causal_tnp_rangesame.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-rangesame/model-lmywe04f:v200', 'incTNP-Batched', False)
    priorBatched = ('experiments/configs/synthetic1d/gp_priorbatched_causal_tnp_combined_rangesame.yml',
        'pm846-university-of-cambridge/mask-priorbatched-tnp-combined-rangesame/model-xdgnof8x:v200', 'incTNP-Batched (Prior)', False)
    cnp = ('experiments/configs/synthetic1d/gp_cnp_rangesame.yml',
        'pm846-university-of-cambridge/cnp-combined-rangesame/model-1pzsub0x:v200', 'CNP', False)
    conv_cnp = ('experiments/configs/synthetic1d/gp_convcnp_rangesame.yml',
        'pm846-university-of-cambridge/convcnp-combined-rangesame/model-awxl9sr4:v200', 'ConvCNP', False)
    models_plain = [tnp_plain, incTNP, batchedTNP, priorBatched, cnp, conv_cnp]
    # TNP-A Class models
    inctnpa = ('experiments/configs/synthetic1d/gp_inctnpa_rangesame.yml',
        'pm846-university-of-cambridge/inctnpa/model-dnbp0124:v199', "incTNP-A", False)
    tnpa = ('experiments/configs/synthetic1d/gp_tnpa_rangesame.yml',
        'pm846-university-of-cambridge/tnpa/model-56ktaaqp:v199', "TNP-A", False)
    models_a = [inctnpa, tnpa]
    # AR NPS
    ar_tnp = ('experiments/configs/synthetic1d/gp_plain_tnp_rangesame.yml',
        'pm846-university-of-cambridge/plain-tnp-rangesame/model-fyr9u053:v200', ' AR TNP-D', True)
    ar_inctnp = ('experiments/configs/synthetic1d/gp_causal_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/mask-tnp-rangesame/model-l69k9pix:v200', 'AR incTNP', True)
    ar_batchedtnp = ('experiments/configs/synthetic1d/gp_batched_causal_tnp_rangesame.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-rangesame/model-lmywe04f:v200', 'AR incTNP-Batched', True)
    ar_priorbatched =('experiments/configs/synthetic1d/gp_priorbatched_causal_tnp_combined_rangesame.yml',
        'pm846-university-of-cambridge/mask-priorbatched-tnp-combined-rangesame/model-xdgnof8x:v200', 'AR incTNP-Batched (Prior)', True)
    ar_cnp = ('experiments/configs/synthetic1d/gp_cnp_rangesame.yml',
        'pm846-university-of-cambridge/cnp-combined-rangesame/model-1pzsub0x:v200', 'AR CNP', True)
    ar_conv_cnp = ('experiments/configs/synthetic1d/gp_convcnp_rangesame.yml',
        'pm846-university-of-cambridge/convcnp-combined-rangesame/model-awxl9sr4:v200', 'AR ConvCNP', True)
    models_ar = [ar_tnp, ar_inctnp, ar_batchedtnp, ar_priorbatched, ar_cnp, ar_conv_cnp]
    # Return
    models_combined = models_plain + models_ar + models_a
    model_cust = [tnp_plain, incTNP, batchedTNP, priorBatched, inctnpa, tnpa, ar_tnp, ar_inctnp, ar_batchedtnp, ar_priorbatched, ar_cnp, cnp]
    models_within = [batchedTNP, conv_cnp, cnp, ar_batchedtnp, ar_cnp, ar_conv_cnp, tnpa]
    models_5k = [tnp_plain, incTNP, batchedTNP, priorBatched, cnp]
    models_1k = [ar_batchedtnp, inctnpa, tnpa, ar_cnp]
    return models_1k

def get_model_list_rbf():
    # List of models to compare trained on rbf kernel
    tnp_plain = ('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml',
        'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-a3qwpptn:v200', 'TNP-D', False)
    incTNP = ('experiments/configs/synthetic1dRBF/gp_causal_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-8mxfyfnw:v200', 'incTNP', False)
    batchedTNP = ('experiments/configs/synthetic1dRBF/gp_batched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-rbf-rangesame/model-xtnh0z37:v200', 'incTNP-Batched', False)
    priorBatched = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'incTNP-Batched (Prior)', False)
    cnp = ('experiments/configs/synthetic1dRBF/gp_cnp_rangesame.yml',
        'pm846-university-of-cambridge/cnp-rbf-rangesame/model-uywfyrx7:v200', 'CNP', False)
    conv_cnp = ('experiments/configs/synthetic1dRBF/gp_convcnp_rangesame.yml',
        'pm846-university-of-cambridge/convcnp-rbf-rangesame/model-uj54q1ya:v200', 'ConvCNP', False)
    models_plain = [tnp_plain, incTNP, batchedTNP, priorBatched, cnp, conv_cnp]
    # TNP-A Class models
    inctnpa = ('experiments/configs/synthetic1dRBF/gp_inctnpa_rangesame.yml',
        'pm846-university-of-cambridge/inctnpa-rbf-rangesame/model-5ob47t8l:v199', "incTNP-A", False)
    tnpa = ('experiments/configs/synthetic1dRBF/gp_tnpa_rangesame.yml',
        'pm846-university-of-cambridge/tnpa-rbf-rangesame/model-e6yry1ri:v199', "TNP-A", False)
    models_a = [inctnpa, tnpa]
    # AR NPS
    ar_tnp = ('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml',
        'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-a3qwpptn:v200', 'TNP-D', True)
    ar_inctnp = ('experiments/configs/synthetic1dRBF/gp_causal_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-8mxfyfnw:v200', 'incTNP', True)
    ar_batchedtnp = ('experiments/configs/synthetic1dRBF/gp_batched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-rbf-rangesame/model-xtnh0z37:v200', 'incTNP-Batched', True)
    ar_priorbatched = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'incTNP-Batched (Prior)', True)
    ar_cnp = ('experiments/configs/synthetic1dRBF/gp_cnp_rangesame.yml',
        'pm846-university-of-cambridge/cnp-rbf-rangesame/model-uywfyrx7:v200', 'CNP', True)
    ar_conv_cnp = ('experiments/configs/synthetic1dRBF/gp_convcnp_rangesame.yml',
        'pm846-university-of-cambridge/convcnp-rbf-rangesame/model-uj54q1ya:v200', 'ConvCNP', True)
    models_ar = [ar_tnp, ar_inctnp, ar_batchedtnp, ar_priorbatched, ar_cnp, ar_conv_cnp]
    # Return
    models_combined = models_plain + models_ar + models_a
    models_within = [batchedTNP, conv_cnp, cnp, ar_batchedtnp, ar_cnp, ar_conv_cnp, tnpa]
    models_5k = [tnp_plain, incTNP, batchedTNP, priorBatched, cnp]
    models_1k = [ar_batchedtnp, inctnpa, tnpa, ar_cnp]
    model_cust = [tnp_plain, incTNP, batchedTNP, priorBatched, inctnpa, tnpa, ar_tnp, ar_inctnp, ar_batchedtnp, ar_priorbatched, ar_cnp, cnp]
    return [ar_batchedtnp, ar_cnp, ar_conv_cnp, tnpa, ar_tnp]


def stream_data_test_rbf():
    # Hypers
    burn_in = 0
    aggregate_over = 1
    batch_size = 16
    max_batches = None # Set to None for no limit
    max_nc = 64
    nt = 128
    start_ctx = 1
    end_ctx = max_nc
    ctx_step = 1
    trained_ctx_end = 64
    device="cuda"
    folder = "experiments/plot_results/llstream/summfreshar/"
    # End of hypers
    stream_data_test(get_rbf_rangesame_testset(max_nc, nt, batch_size), get_model_list_rbf(), max_nc, nt, start_ctx, end_ctx, ctx_step, device, folder, trained_ctx_end, max_batches, burn_in, aggregate_over)


def stream_data_test_combined():
    # Hypers
    burn_in = 0
    aggregate_over = 1
    batch_size = 16
    max_batches = None # Set to None for no limit
    max_nc = 1_000
    nt = 128
    start_ctx = 1
    end_ctx = max_nc
    ctx_step = 50
    trained_ctx_end = 64
    device="cuda"
    folder = "experiments/plot_results/llstream/long1k/"
    # End of hypers
    stream_data_test(get_combined_rangesame_testset(max_nc, nt, batch_size), get_model_list_combined(), max_nc, nt, start_ctx, end_ctx, ctx_step, device, folder, trained_ctx_end, max_batches, burn_in, aggregate_over)

@torch.no_grad
def stream_data_test(dataset, models, max_nc, nt, start_ctx, end_ctx, ctx_step, device, folder, trained_ctx_end, max_batches, burn_in, aggregate_over):
    data, kernel_name, len_dat = dataset
    ctx = list(range(start_ctx, end_ctx, ctx_step))
    ctx.append(end_ctx)
    ctx = np.array(ctx)
    ll_list = np.zeros((len(models), len(ctx), len_dat, aggregate_over))
    condition_time_list = np.zeros((len(models), len(ctx), len_dat, aggregate_over))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gt_lls = np.zeros(len_dat)
    gt_ll_caculated = False
    for model_idx, (model_yml, model_wab, model_name, use_ar) in enumerate(models):
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        is_model_inc = isinstance(model, IncUpdateEff) and model_name != "CNP" and model_name != "ConvCNP"
        for batch_idx, batch in tqdm(enumerate(data), desc=f'{model_name}'):
            if max_batches is not None and batch_idx >= max_batches: break
            # Moves batch to gpu
            batch.xc, batch.yc, batch.xt, batch.yt = batch.xc.to(device), batch.yc.to(device), batch.xt.to(device), batch.yt.to(device)
            xc, yc, xt, yt = batch.xc, batch.yc, batch.xt, batch.yt
            m, nt, dy = yt.shape
            # Compute gt ll
            if not gt_ll_caculated:
                _, _, gt_loglik = batch.gt_pred(
                            xc=xc, yc=yc, xt=xt, yt=yt
                        )
                gt_loglik = (gt_loglik.sum() / (m * nt)).item()
                gt_lls[batch_idx] = gt_loglik
            if is_model_inc: model.init_inc_structs(m=m, max_nc=max_nc, device=device)
            for ctx_idx, ctx_upper in enumerate(ctx):
                # Does model burn in and aggregation
                for j in range(burn_in + aggregate_over):
                    ctx_lower = 0 if ctx_idx == 0 else ctx[ctx_idx - 1]
                    if use_ar:
                        xc_new, yc_new = xc[:, :ctx_upper, :], yc[:, :ctx_upper, :]
                        # Times LL but this doesnt really make sense for a timing as is teacher forcing
                        torch.cuda.synchronize()
                        starter.record()
                        with torch.no_grad():
                            loglik = ar_loglik(np_model=model, xc=xc_new, yc=yc_new, xt=xt, yt=yt, 
                                           normalise=True, order="random").mean().item()
                        ender.record()
                        torch.cuda.synchronize()
                        runtime_ms = starter.elapsed_time(ender)
                    elif is_model_inc:
                        xc_new, yc_new = xc[:, ctx_lower:ctx_upper, :], yc[:, ctx_lower:ctx_upper, :]
                        # Time the conditioning phase
                        torch.cuda.synchronize()
                        starter.record()
                        with torch.no_grad():
                            model.update_ctx(xc=xc_new, yc=yc_new)
                        ender.record()
                        torch.cuda.synchronize()
                        runtime_ms = starter.elapsed_time(ender)
                        # Gets predictive distribution
                        pred_dist = model.query(xt=xt, dy=dy)
                        loglik = (pred_dist.log_prob(yt).sum() / yt[..., 0].numel()).item()
                    else:
                        xc_new, yc_new = xc[:, :ctx_upper, :], yc[:, :ctx_upper, :]
                        batch = Batch(xc=xc_new, yc=yc_new, xt=xt, yt=yt, x=None, y=None)
                        # Times whole prediction and treats it as conditioning cost
                        torch.cuda.synchronize()
                        starter.record()
                        with torch.no_grad():
                            pred_dist = np_pred_fn(model, batch, predict_without_yt_tnpa=False) # uses teacher forcing if possible
                        ender.record()
                        torch.cuda.synchronize()
                        runtime_ms = starter.elapsed_time(ender)
                        loglik = (pred_dist.log_prob(yt).sum() / yt[..., 0].numel()).item()
                    # Records likelihood and runtime
                    write_idx = j - burn_in
                    if write_idx >= 0:
                        ll_list[model_idx, ctx_idx, batch_idx, write_idx] = loglik
                        condition_time_list[model_idx, ctx_idx, batch_idx, write_idx] = runtime_ms
        gt_ll_caculated = True
    # Averages over batches
    ll_mean = np.mean(ll_list, axis=(2,3))
    gt_average = np.mean(gt_lls, axis=0)
    no_runs = ll_list.shape[2] * ll_list.shape[3]
    ll_sem = np.std(ll_list, axis=(2,3), ddof=1) / np.sqrt(no_runs)
    condition_time_list = np.mean(condition_time_list, axis=(2, 3))

    # Saves data to output file to be used when plotting
    file_name_npz = f'npz_kernel_{kernel_name}.npz'
    npz_arr_path = folder + file_name_npz
    np.savez(npz_arr_path, ll=ll_mean, ctx=ctx, time=condition_time_list, ll_sem=ll_sem)
    json_file_path = folder + f'json_{kernel_name}.json'
    summary_meta = {
        "model_names": [m[2] for m in models],
        "trained_ctx_end": trained_ctx_end,
        "gt_average_ll": gt_average,
        "kernel_name": kernel_name,
        "npz_path": npz_arr_path,
        "folder": folder,
    }
    with open(json_file_path, 'w') as fileobj:
        json.dump(summary_meta, fileobj, indent=4)
    print(f"Summary at {json_file_path}")
    
    plot_saved_info(json_file_path)


def plot_saved_info(json_path):
    with open(json_path, 'r') as fileobj:
        metadata = json.load(fileobj)
    model_names = metadata['model_names']
    trained_ctx_end = metadata['trained_ctx_end']
    gt_average = metadata['gt_average_ll']
    kernel_name = metadata['kernel_name']
    folder = metadata['folder']
    npz_path = metadata['npz_path']
    # Loads np arrays
    data = np.load(npz_path)
    ll_list = data['ll']
    ll_sem = data['ll_sem']
    condition_time_list = data['time']
    ctx = data['ctx']

    # Plots LL as context size increases - red dotted line to show when going beyond trained context size
    ll_file_name = folder + f'll_kernel_{kernel_name}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, model_name in enumerate(model_names):
        mean, sem = ll_list[model_idx], ll_sem[model_idx]
        ax.plot(ctx, mean, label=model_name)
        ax.fill_between(ctx, mean - sem, mean + sem, alpha=0.25)
    ax.axvline(x=trained_ctx_end, color='red', linestyle=':')
    ax.axhline(y=gt_average, color='grey', linestyle='--', label='Mean GT LL')
    ax.text(x=trained_ctx_end + 5, y=ax.get_ylim()[1] * 0.40, s='Max Trained NC', color='red', rotation=90, verticalalignment='top')
    ax.set_xlabel('Number of Context Points')
    ax.set_ylabel('Mean Log-Likelihood')
    ax.legend()
    ax.set_title(f'Streamed Performance of NP Models on {kernel_name}')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(ll_file_name, dpi=300)
    # Plots conditioning time vs number of context points
    runtime_file_name = folder + f'runtime_kernel_{kernel_name}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, model_name in enumerate(model_names):
        ax.plot(ctx, condition_time_list[model_idx], label=model_name)
    ax.set_xlabel('Number of Context Points')
    ax.set_ylabel('Mean Conditioning Time (ms)')
    ax.legend()
    ax.set_title(f'Conditioning Time of NP Models')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(runtime_file_name, dpi=300)



if __name__ == "__main__":
    stream_data_test_rbf()
    exit(0)
    stream_data_test_combined()
    # TODO look at distribution drift maybe

