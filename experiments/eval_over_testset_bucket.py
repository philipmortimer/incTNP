# Evaluates over many instance of test set similar to eval_over_testset.py
# Difference is it buckets items by the context size. This allows for a more vectorised approach.
# A much better approach would be to implement block mask support for all TNP models to run multiple instances together
# but this is a good medium without altering any TNP API calls atm.
import numpy as np
import torch
from scipy import stats
from check_shapes import check_shapes
from tnp.utils.experiment_utils import initialize_experiment, deep_convert_dict, extract_config
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.lightning_utils import LitWrapper
import time
import warnings
from tnp.data.gp import RandomScaleGPGenerator
from tnp.networks.gp import RBFKernel
from functools import partial
import random
from typing import Callable, List, Tuple, Optional, Union
import os
import wandb
from tnp.data.base import Batch, GroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch
from tnp.utils.np_functions import np_pred_fn, np_loss_fn
from torch import nn
import copy
import hiyapyco
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tnp.data.base import Batch
from tnp.models.incTNPBatchedPrior import IncTNPBatchedPrior
from plot_adversarial_perms import get_model
from tqdm import tqdm
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple
from copy import deepcopy

# Compiled pred fn - investigate this but no python 3.12 support so need to make new conda env atm
compiled_pred_fn = torch.compile(
    partial(np_pred_fn, predict_without_yt_tnpa=True),
    disable=True,
    #mode="reduce-overhead",
    #fullgraph=False,
)

# Loads data effeciently for fast computation
def load_data(data_gen, device="cuda"):
    start_t = time.time()
    for b in data_gen:
        b.xc = b.xc.to(device, non_blocking=True)
        b.yc = b.yc.to(device, non_blocking=True)
        b.xt = b.xt.to(device, non_blocking=True)
        b.yt = b.yt.to(device, non_blocking=True)
        b.x = b.x.to(device, non_blocking=True)
        b.y = b.y.to(device, non_blocking=True)
        # Pre computes GTLL
        m, nt, _ = b.xt.shape
        _, _, gt_loglik = b.gt_pred(
                    xc=b.xc, yc=b.yc, xt=b.xt, yt=b.yt
                )
        gt_loglik = gt_loglik.sum() / (m * nt)
        b.gtll = gt_loglik.to(device, non_blocking=True)
        b.gt_pred = None
    data = list(data_gen)
    print(f'Data Time {time.time() - start_t:.2f}')
    return data

# Gets rbf kernel with rangesame default test params used
def get_rbf_rangesame_test_set():
    # RBF kernel params
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    # Data generator params for test set
    min_nc = 1
    max_nc = 64
    nt= 128
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 4096
    #min_nc, max_nc = 256, 256
    batch_size = 16
    noise_std = 0.1
    deterministic = True
    gen_test = RandomScaleGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    data = load_data(gen_test)
    return data, "RBF"


# List of models to be tested, adjust this as required
def get_model_list(N_PERMUTATIONS, ar_runs):
    # Models available
    tnp_plain = ('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-7ib3k6ga:v200', 'random', "TNP-D", "",
        1)
    tnp_causal = ('experiments/configs/synthetic1dRBF/gp_causal_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v200', 'random', "IncTNP", "",
        N_PERMUTATIONS)
    tnp_causal_batched = ('experiments/configs/synthetic1dRBF/gp_batched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-batched-tnp-rbf-rangesame/model-xtnh0z37:v200', 'random', "IncTNP (Batched)", "",
        N_PERMUTATIONS)
    tnp_causal_batched_prior = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'random', "IncTNP-Prior (Batched)", "",
        N_PERMUTATIONS)
    # TNP Causal Batched Prior Greedy Strategies
    greedy_best_tnp_causal_batched_prior_logp = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'GreedyBestPriorLogP', 
        "IncTNP-Prior (Batched) - Best Greedy LL", "",
        1)
    greedy_worst_tnp_causal_batched_prior_logp = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'GreedyWorstPriorLogP', 
        "IncTNP-Prior (Batched) - Worst Greedy LL","",
        1) 
    greedy_median_tnp_causal_batched_prior_logp = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'GreedyMedianPriorLogP', 
        "IncTNP-Prior (Batched) - Median Greedy LL", "",
        1)
    greedy_best_tnp_causal_batched_prior_var = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'GreedyBestPriorVar', 
        "IncTNP-Prior (Batched) - Best Greedy Var", "",
        1)
    greedy_worst_tnp_causal_batched_prior_var = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'GreedyWorstPriorVar', 
        "IncTNP-Prior (Batched) - Worst Greedy Var","",
        1) 
    greedy_median_tnp_causal_batched_prior_var = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'GreedyMedianPriorVar', 
        "IncTNP-Prior (Batched) - Median Greedy Var", "",
        1)
    # TNP AR models
    ar_yml, ar_mod, name = 'experiments/configs/synthetic1dRBF/gp_tnpa_rangesame.yml', 'pm846-university-of-cambridge/tnpa-rbf-rangesame/model-wbgdzuz5:v200', "TNP-A"
    tnp_ar_5 = (ar_yml, ar_mod, 'random', name + " (5)", "TNPAR_5", ar_runs)
    tnp_ar_50 = (ar_yml, ar_mod, 'random', name + " (50)", "TNPAR_50", ar_runs)
    tnp_ar_100 = (ar_yml, ar_mod, 'random', name + " (100)", "TNPAR_100", ar_runs)
    
    # Defines models to be used
    models_all = [tnp_plain, tnp_causal, tnp_causal_batched, tnp_causal_batched_prior, 
        greedy_best_tnp_causal_batched_prior_logp, greedy_worst_tnp_causal_batched_prior_logp, greedy_median_tnp_causal_batched_prior_logp,
        greedy_best_tnp_causal_batched_prior_var, greedy_worst_tnp_causal_batched_prior_var, greedy_median_tnp_causal_batched_prior_var,
        tnp_ar_5, tnp_ar_50, tnp_ar_100]
    models_no_ar = [tnp_plain, tnp_causal, tnp_causal_batched, tnp_causal_batched_prior, 
        greedy_best_tnp_causal_batched_prior_logp, greedy_worst_tnp_causal_batched_prior_logp, greedy_median_tnp_causal_batched_prior_logp,
        greedy_best_tnp_causal_batched_prior_var, greedy_worst_tnp_causal_batched_prior_var, greedy_median_tnp_causal_batched_prior_var]
    models_ar = [tnp_ar_5, tnp_ar_50, tnp_ar_100]
    models_me = [tnp_causal, tnp_causal_batched, tnp_causal_batched_prior]
    models_greedy = [greedy_best_tnp_causal_batched_prior_logp, greedy_worst_tnp_causal_batched_prior_logp, greedy_median_tnp_causal_batched_prior_logp,
        greedy_best_tnp_causal_batched_prior_var, greedy_worst_tnp_causal_batched_prior_var, greedy_median_tnp_causal_batched_prior_var]
    return models_me


def shuffle_batch(model, batch, shuffle_strategy: str, device: str="cuda"):
    assert shuffle_strategy in {"random", "GreedyBestPriorLogP", "GreedyWorstPriorLogP", "GreedyMedianPriorLogP", "GreedyBestPriorVar", "GreedyWorstPriorVar", "GreedyMedianPriorVar"}, "Invalid context shuffle strategy"
    m, nc, dx = batch.xc.shape
    _, nt, dy = batch.yt.shape
    # Converts batch to cuda
    #batch.xc, batch.yc, batch.xt, batch.yt, batch.x, batch.y = batch.xc.to(device), batch.yc.to(device), batch.xt.to(device), batch.yt.to(device), batch.x.to(device), batch.y.to(device)
    xc_new, yc_new = None, None
    if shuffle_strategy == "random":
        perms = torch.rand(m, nc, device=batch.xc.device).argsort(dim=1)
        perm_x = perms.unsqueeze(-1).expand(-1, -1, dx)
        perm_y = perms.unsqueeze(-1).expand(-1, -1, dy)
        xc_new = torch.gather(batch.xc, 1, perm_x) 
        yc_new = torch.gather(batch.yc, 1, perm_y)
    elif shuffle_strategy == "GreedyBestPriorLogP":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(batch.xc, batch.yc, policy="best", select="logp")
    elif shuffle_strategy == "GreedyWorstPriorLogP":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(batch.xc, batch.yc, policy="worst", select="logp")
    elif shuffle_strategy == "GreedyMedianPriorLogP":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(batch.xc, batch.yc, policy="median", select="logp")
    elif shuffle_strategy == "GreedyBestPriorVar":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(batch.xc, batch.yc, policy="best", select="var")
    elif shuffle_strategy == "GreedyWorstPriorVar":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(batch.xc, batch.yc, policy="worst", select="var")
    elif shuffle_strategy == "GreedyMedianPriorVar":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(batch.xc, batch.yc, policy="median", select="var")
    
    x = torch.cat((xc_new, batch.xt), dim=1)
    y = torch.cat((yc_new, batch.yt), dim=1)
    batch_new = Batch(xc=xc_new, yc=yc_new, xt=batch.xt, yt=batch.yt, y=y, x=x)
    return batch_new

# Repeats batch with shuffle each time and then combines them - shuffling each time
def _replicate_batch(
    model, batch: Batch, n_rep: int, shuffle_strategy: str, perms: Optional[torch.Tensor] = None,
) -> Batch:
    # Adds precomputed support for random shuffling
    if shuffle_strategy == "random" and perms is not None:
        m, nc, dx = batch.xc.shape
        _, _, dy = batch.yc.shape
        # Repeats tensors
        xc_rep = batch.xc.repeat(n_rep, 1, 1)
        yc_rep = batch.yc.repeat(n_rep, 1, 1)
        xt_rep = batch.xt.repeat(n_rep, 1, 1)
        yt_rep = batch.yt.repeat(n_rep, 1, 1)

        # Broadcast permutation indices
        perm_rows = perms.repeat_interleave(m, 0)
        perm_x = perm_rows.unsqueeze(-1).expand(-1, -1, dx)
        perm_y = perm_rows.unsqueeze(-1).expand(-1, -1, dy)

        # Gathers shuffled data
        xc_new = torch.gather(xc_rep, 1, perm_x)
        yc_new = torch.gather(yc_rep, 1, perm_y)

        big_batch = Batch(xc=xc_new, yc=yc_new, xt=xt_rep, yt=yt_rep, x=None, y=None)
    else: # Default no vectorised approach
        reps = [shuffle_batch(model, batch, shuffle_strategy) for _ in range(n_rep)]
        xc = torch.cat([b.xc for b in reps], dim=0)
        yc = torch.cat([b.yc for b in reps], dim=0)
        xt = torch.cat([b.xt for b in reps], dim=0)
        yt = torch.cat([b.yt for b in reps], dim=0)

        big_batch = Batch(xc=xc, yc=yc, xt=xt, yt=yt, x=None, y=None)

    return big_batch


@torch.no_grad
def fast_eval_model(
    model,
    test_set: List[Batch],
    n_permutations: int = 100,
    shuffle_strategy: str = "random",
    max_size_gpu: int = 2048,
    USE_HALF_PREC: bool = False,
    device: str = "cuda",
):
    model.eval()
    device = torch.device(device)
    model.to(device) 

    # Buckets by context length (our API requires same size nc atm so need to do this which is big cause of ineffeciency)
    buckets: DefaultDict[int, List[Batch]] = defaultdict(list)
    for batch in test_set:
        nc = batch.xc.shape[1]
        buckets[nc].append(batch)

    sum_lls_perm = [0.0] * n_permutations
    count_perm = [0] * n_permutations
    ll_sum = rmse_sum = gtll_sum = 0.0
    ll_sq_sum = rmse_sq_sum = gtll_sq_sum = 0.0
    total_samples = 0

    shuffle_time = inf_time = stat_time = 0.0
    i = 0
    with torch.no_grad():
        for nc, batch_group in buckets.items():
            desc = f"nc={nc:02d}"
            for base_batch in tqdm(batch_group, desc=desc, leave=False):
                gt_ll= base_batch.gtll.item() # GT LL
                gtll_sum += gt_ll
                gtll_sq_sum += (gt_ll ** 2)
                i+=1
                m = base_batch.xc.shape[0] # Original batch size (e.g. m = 16)
                no_reps_allowed = max(max_size_gpu // m, 1) # How many full shuffles can be put into gpu
                # Loops through all permutations
                processed = 0
                while processed < n_permutations:
                    n_rep = min(no_reps_allowed, n_permutations - processed)

                    # Shuffles dataset and batches it
                    t0 = time.time()
                    if shuffle_strategy == "random": perms_slice = torch.rand(n_rep, nc, device=device).argsort(1)
                    else: perms_slice = None
                    if USE_HALF_PREC and processed==0:
                        base_batch.xc = base_batch.xc.half()
                        base_batch.yc = base_batch.yc.half()
                        base_batch.xt = base_batch.xt.half()
                        base_batch.yt = base_batch.yt.half()
                    big_batch = _replicate_batch(model, base_batch, n_rep, shuffle_strategy, perms=perms_slice)
                    shuffle_time += time.time() - t0

                    # Prediction
                    t1 = time.time()
                    if USE_HALF_PREC:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            pred_dist = compiled_pred_fn(model, big_batch)#pred_dist = np_pred_fn(model, big_batch, predict_without_yt_tnpa=True)
                    else: pred_dist = compiled_pred_fn(model, big_batch)#pred_dist = np_pred_fn(model, big_batch, predict_without_yt_tnpa=True)
                    inf_time += time.time() - t1

                    # Tracked stats
                    t2 = time.time()
                    mean_f32 = pred_dist.mean.float()
                    yt_f32 = big_batch.yt.float()
                    n_tot, nt, _ = big_batch.yt.shape
                    ll = pred_dist.log_prob(big_batch.yt).float().sum(dim=(1, 2)) / nt
                    rmse = nn.functional.mse_loss(mean_f32, yt_f32, reduction="none")
                    rmse = rmse.mean(dim=(1, 2)).sqrt()
    
                    ll_rep = ll.view(n_rep, m).mean(1).cpu()
                    for off, idx in enumerate(range(processed, processed + n_rep)):
                        sum_lls_perm[idx] += ll_rep[off].item()
                        count_perm[idx] += 1

                    # Welford online update
                    ll_sum += ll.sum().item()
                    ll_sq_sum += (ll ** 2).sum().item()
                    rmse_sum += rmse.sum().item()
                    rmse_sq_sum += (rmse ** 2).sum().item()

                    total_samples += n_tot
                    stat_time += time.time() - t2
                    processed += n_rep

    def _mean_std_calc(sum_, sq_sum, n_calc):
        mean = sum_ / n_calc
        var = max(sq_sum / (n_calc) - mean ** 2, 0.0)
        return mean, var ** 0.5

    mean_ll, std_ll = _mean_std_calc(ll_sum, ll_sq_sum, n_calc=total_samples)
    mean_rmse, std_rmse = _mean_std_calc(rmse_sum, rmse_sq_sum, n_calc=total_samples)
    mean_gtll, std_gtll = _mean_std_calc(gtll_sum, gtll_sq_sum, n_calc=i)
    mean_lls = [s / c if c else float("nan") for s, c in zip(sum_lls_perm, count_perm)]

    if False: print(f"\n Timing shuffle: {shuffle_time:.1f}s - inference: {inf_time:.1f}s - stats: {stat_time:.1f}s")

    return {
        "mean_ll": mean_ll,
        "std_ll": std_ll,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "mean_gt_ll": mean_gtll,
        "std_gt_ll": std_gtll,
        "total_samples": total_samples,
        "shuffle_time": shuffle_time,
        "inference_time": inf_time,
        "stat_time": stat_time,
        "mean_lls": mean_lls
    }

def run_eval():
    MAX_SIZE_GPU = 4096 # Max size - tune with GPU used to maximisie throughput
    N_PERMUTATIONS = 1_000 # How many permutations of dataset to test
    ar_runs = 1 # How many TNP A runs per model to try - if TNPA is being included
    USE_HALF_PREC = True # Use float16?
    pl.seed_everything(1)

    # 4096 good for CBL - 32768 good for csd3

    folder_name = "experiments/plot_results/eval_set/"
    test_data = get_rbf_rangesame_test_set()
    test_set, set_name = test_data
    file_txt = f'Summary over eval data set {set_name}'
    print(file_txt)
    model_list = get_model_list(N_PERMUTATIONS=N_PERMUTATIONS, ar_runs=ar_runs)
    for yml_path, wandb_id, shuffle_strategy, model_name, special_args, perms_over in model_list:
        model = get_model(yml_path, wandb_id, seed=False)
        if USE_HALF_PREC: model = model.half()
        if special_args.startswith("TNPAR_"):
            model.num_samples = int(special_args.split("_")[1])

        res = fast_eval_model(
            model,
            test_set,
            n_permutations=perms_over,
            shuffle_strategy=shuffle_strategy,
            max_size_gpu=MAX_SIZE_GPU,
            USE_HALF_PREC=USE_HALF_PREC,
            device="cuda",
        )

        summary_block = "\n" + ("-" * 20) + "\n" + f"Model: {model_name}"
        for k, v in res.items():
            if k == "mean_lls":
                file_out = f"{folder_name}datallsgpu5/{model_name.replace(' ', '_')}_mean_lls_{set_name}.txt"
                np.savetxt(file_out, np.array(v, dtype=np.float32), fmt="%.12f")
                summary_block += f"\n{ k:>15}: wrote {len(v)} values to {file_out}"
            else:
                summary_block += f"\n{ k:>15}: {v}"
        file_txt += summary_block
        print(summary_block)
    with open(folder_name + 'eval_summary_bucketed.txt', 'w') as file_object:
        file_object.write(file_txt)


if __name__ == "__main__":
    run_eval()