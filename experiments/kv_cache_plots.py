# File that tests kv caching and plots it to show speedup
import torch
from plot_adversarial_perms import get_model
import torch.distributions as td
from tnp.networks.kv_cache import init_kv_cache, update_kv_cache
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

np.set_printoptions(threshold=np.inf) # Prints the whole numpy array for the file

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.titlesize"]= 14

# Tests that KV caching works exactly the same as without
@torch.no_grad
def test_kv_cache():
    atol=1e-4 # Tolerance for close checks
    rtol = 1e-4
    device='cuda'
    # Fetches model
    model = get_model('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', device=device)
    model.eval()
    # Generates random dataset
    N, m, nc, nt, dx, dy = 100, 16, 32, 128, 1, 1
    max_high = 2
    xcs = (torch.rand((N, m, nc, dx), device=device) * (2 * max_high)) - max_high
    ycs = (torch.rand((N, m, nc, dy), device=device) * (2 * max_high)) - max_high
    xts = (torch.rand((N, m, nt, dx), device=device) * (2 * max_high)) - max_high
    yts = (torch.rand((N, m, nt, dy), device=device) * (2 * max_high)) - max_high
    # Loops through data and asserts that with / without KV caching produce the same model prediction
    for i in range(N):
        # Initialises KV cache with start token
        start_token = model.encoder.empty_token.expand(m, -1, -1) # Starts with empty token (prior condition)
        kv_cache = init_kv_cache()
        model.encoder.update_context(start_token, kv_cache)
        xc, yc, xt, yt = xcs[i,:,:,:], ycs[i,:,:,:], xts[i,:,:,:], yts[i,:,:,:]
        # Number of context tokens conditioned on
        for ctx_toks in range(nc):
            xc_red, yc_red = xc[:, :ctx_toks, :], yc[:, :ctx_toks, :]
            xc_new, yc_new = xc[:, ctx_toks:ctx_toks+1, :], yc[:, ctx_toks:ctx_toks+1, :]
            # Non KV-cached
            pred_dist_non_cached = model.likelihood(model.decoder(model.encoder(xc=xc_red, yc=yc_red, xt=xt), xt))
            # KV cached
            zt = model.encoder._preprocess_targets(xt, dy)
            pred_dist_kv_cached = model.likelihood(model.decoder(model.encoder.query(zt, kv_cache), xt))
            new_zc = model.encoder._preprocess_context(xc_new, yc_new)
            model.encoder.update_context(new_zc, kv_cache)
            # Checks that distributions are same
            assert isinstance(pred_dist_non_cached, td.Normal) and isinstance(pred_dist_kv_cached, td.Normal), "Both should be normal predictions"
            assert torch.allclose(pred_dist_non_cached.mean, pred_dist_kv_cached.mean, atol=atol, rtol=rtol), "Dist means must be same"
            assert torch.allclose(pred_dist_non_cached.stddev, pred_dist_kv_cached.stddev, atol=atol, rtol=rtol), "Dist std must be same"#
    print("KV Cache tests all passed")

    
# Measures the conditioning time for the model
@torch.no_grad
def measure_condition_time_memory_kv():
    # Gets model
    device='cuda'
    model = get_model('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', device=device)
    model.eval()
    # Dataset
    burn_in = 1 # Number of burn in runs to ignore
    aggregate_over = 5 # Number of runs to aggregate data over
    token_step = 1 # How many increments of tokens to go up in
    max_nc, dx, dy, m = 50_000, 1, 1, 1
    max_high = 2
    xcs = (torch.rand((1, max_nc, dx), device=device) * max_high * 2) - max_high
    ycs = (torch.rand((1, max_nc, dy), device=device) * max_high * 2) - max_high
    # Results structures
    context_sizes = np.arange(start=0, stop=max_nc, step=token_step, dtype=int)
    upper_ctxs = np.array([min(i + token_step, max_nc) for i in context_sizes])
    runtime = np.zeros((aggregate_over, len(context_sizes)))
    memory = np.zeros((aggregate_over, len(context_sizes)))
    ctx_inc = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for j in range(burn_in + aggregate_over):
        # Initailises KV cache
        start_token = model.encoder.empty_token.expand(m, -1, -1) # Starts with empty token (prior condition)
        kv_cache = init_kv_cache()
        model.encoder.update_context(start_token, kv_cache)
        torch.cuda.reset_peak_memory_stats() #  Resets memory stats - we want cumulative memory
        # Adds context tokens n at a time
        ctx_inc = 0
        for lower_ctx, upper_ctx in zip(context_sizes, upper_ctxs):
            xc_new = xcs[:, lower_ctx:upper_ctx, :]
            yc_new = ycs[:, lower_ctx:upper_ctx, :]

            # Sets up measures
            torch.cuda.synchronize()
            starter.record()
            # Core update step
            with torch.no_grad():
                new_zc = model.encoder._preprocess_context(xc_new, yc_new)
                model.encoder.update_context(new_zc, kv_cache)
            # Measures time and memory
            ender.record()
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            update_time_ms = starter.elapsed_time(ender)

            # Writes measured results
            if j >= burn_in:
                write_j = j - burn_in
                runtime[write_j, ctx_inc] = update_time_ms
                memory[write_j, ctx_inc] = peak_memory_mb
            ctx_inc += 1
    # Averages runtime and memory
    runtime_std = runtime.std(axis=0, ddof=1)
    memory_std = memory.std(axis=0, ddof=1)
    runtime = np.mean(runtime, axis=0)
    memory = np.mean(memory, axis=0)

    # Writes results to file
    summary_block = f"""
    ----------------------------
    Cumulative Context Size: {upper_ctxs}
    Runtime Incremental (ms): {runtime}
    Runtime std: {runtime_std}
    memory std: {memory_std}
    Memory Cumulative (Mb): {memory}
    """
    print(summary_block)
    with open('experiments/plot_results/kv/' + 'mem_run.txt', 'w') as file_object:
        file_object.write(summary_block)
    confidence_bars = True
    # Plots runtime
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(upper_ctxs, runtime)
    if confidence_bars:
        ci95 = 1.96 * runtime_std / np.sqrt(aggregate_over)
        ax.fill_between(upper_ctxs,
                        runtime - ci95,
                        runtime + ci95,
                        alpha=0.25)
    ax.set_xlabel('Context Size')
    ax.set_ylabel('Runtime for Conditioning (ms)')
    ax.set_title(f'Runtime as Context Size Increases with KV-Caching (M={token_step})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig('experiments/plot_results/kv/time_vs_ctx.png', dpi=300) 

    # Plots cumulative memory 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(upper_ctxs, memory)
    if confidence_bars:
        ci95 = 1.96 * memory_std / np.sqrt(aggregate_over)
        ax.fill_between(upper_ctxs,
                        memory - ci95,
                        memory + ci95,
                        alpha=0.25)
    ax.set_xlabel('Context Size')
    ax.set_ylabel('Cumulative Memory Usage (MB)')
    ax.set_title('Memory Use with KV-Caching')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig('experiments/plot_results/kv/memory_vs_ctx.png', dpi=300) 

    # Plots both on same grid
    COL_RT, COL_MEM = 'C0', 'C3' # Colour palleter
    fig, ax_rt = plt.subplots(figsize=(7, 5))
    ax_mem = ax_rt.twinx()
    # Runtime
    ax_rt.plot(upper_ctxs, runtime, color=COL_RT, label="Runtime")
    if confidence_bars:
        ci95 = 1.96 * runtime_std / np.sqrt(aggregate_over)
        ax_rt.fill_between(upper_ctxs,
                        runtime - ci95,
                        runtime + ci95,
                        alpha=0.25,color=COL_RT)
    ax_rt.set_xlabel('Context Size')
    ax_rt.set_ylabel('Runtime for Conditioning (ms)', color=COL_RT)
    ax_rt.tick_params(axis='y', colors=COL_RT)
    ax_rt.grid(True, linestyle='--', alpha=0.4)
    # Memory
    ax_mem.plot(upper_ctxs, memory, color=COL_MEM, label="Memory")
    if confidence_bars:
        ci95 = 1.96 * memory_std / np.sqrt(aggregate_over)
        ax_mem.fill_between(upper_ctxs,
                        memory - ci95,
                        memory + ci95,
                        alpha=0.25,color=COL_MEM)
    ax_mem.set_ylabel('Cumulative Memory Usage (MB)', color=COL_MEM)
    ax_mem.tick_params(axis='y', colors=COL_MEM)
    #Other details
    lines = ax_rt.get_lines() + ax_mem.get_lines()
    labels = [l.get_label() for l in lines]
    ax_rt.legend(lines, labels)
    fig.suptitle(f'KV-Cache Scaling (M={token_step})', fontsize=14)
    fig.tight_layout()
    fig.savefig('experiments/plot_results/kv/runtime_memory_combined.png',dpi=300)
  

# Measures the conditioning time for the model
@torch.no_grad
def compare_kv_against_none(strategy="fixed", targets=128):
    assert strategy in {"fixed", "scale"}, "Invalid strategy"
    # Gets model
    device='cuda'
    model = get_model('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', device=device)
    model.eval()
    # Dataset
    burn_in = 1 # Number of burn in runs to ignore
    aggregate_over = 20 # Number of runs to aggregate data over
    token_step = 50 # How many increments of tokens to go up in
    max_nc, dx, dy, m = 20_000, 1, 1, 1
    nt = targets if strategy == "fixed" else max_nc
    max_high = 2
    xcs = (torch.rand((1, max_nc, dx), device=device) * max_high * 2) - max_high
    ycs = (torch.rand((1, max_nc, dy), device=device) * max_high * 2) - max_high
    xts = (torch.rand((1, nt, dx), device=device) * max_high * 2) - max_high
    #yts = (torch.rand((1, nt, dy), device=device) * max_high * 2) - max_high
    # Results structures
    context_sizes = np.arange(start=0, stop=max_nc, step=token_step, dtype=int)
    upper_ctxs = np.array([min(i + token_step, max_nc) for i in context_sizes])
    runtime_no_kv = np.zeros((aggregate_over, len(context_sizes)))
    condition_kv = np.zeros((aggregate_over, len(context_sizes)))
    query_kv = np.zeros((aggregate_over, len(context_sizes)))
    memory_kv = np.zeros((aggregate_over, len(context_sizes)))
    memory_no_kv = np.zeros((aggregate_over, len(context_sizes)))
    ctx_inc = 0
    # Measures memory and runtime for model with no KV cache
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for j in range(burn_in + aggregate_over):
        # Initailises KV cache
        start_token = model.encoder.empty_token.expand(m, -1, -1) # Starts with empty token (prior condition)
        kv_cache = init_kv_cache()
        model.encoder.update_context(start_token, kv_cache)
        torch.cuda.reset_peak_memory_stats() # Resets memory stats - we want cumulative memory
        # Adds context tokens n at a time
        ctx_inc = 0
        for lower_ctx, upper_ctx in zip(context_sizes, upper_ctxs):
            xc = xcs[:, :upper_ctx, :]
            yc = ycs[:, :upper_ctx, :]
            if strategy == "fixed": xt = xts
            else: xt = xts[:,:upper_ctx,:]

            # Sets up measures
            torch.cuda.synchronize()
            starter.record()
            # Core update step
            with torch.no_grad():
                pred_dist = model.likelihood(model.decoder(model.encoder(xc=xc, yc=yc, xt=xt), xt))
            # Measures time and memory
            ender.record()
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            update_time_ms = starter.elapsed_time(ender)

            # Writes measured results
            if j >= burn_in:
                write_j = j - burn_in
                runtime_no_kv[write_j, ctx_inc] = update_time_ms
                memory_no_kv[write_j, ctx_inc] = peak_memory_mb
            ctx_inc += 1

    # Measures KV caching performance
    for j in range(burn_in + aggregate_over):
        # Initailises KV cache
        start_token = model.encoder.empty_token.expand(m, -1, -1) # Starts with empty token (prior condition)
        kv_cache = init_kv_cache()
        model.encoder.update_context(start_token, kv_cache)
        torch.cuda.reset_peak_memory_stats() # Resets memory stats - we want cumulative memory
        # Adds context tokens n at a time
        ctx_inc = 0
        for lower_ctx, upper_ctx in zip(context_sizes, upper_ctxs):
            xc_new = xcs[:, lower_ctx:upper_ctx, :]
            yc_new = ycs[:, lower_ctx:upper_ctx, :]
            if strategy == "fixed": xt = xts
            else: xt = xts[:,:upper_ctx,:]

            # Measures condition step
            torch.cuda.synchronize()
            starter.record()
            # Core update step
            with torch.no_grad():
                new_zc = model.encoder._preprocess_context(xc_new, yc_new)
                model.encoder.update_context(new_zc, kv_cache)
            # Measures time and memory
            ender.record()
            torch.cuda.synchronize()
            condition_time_ms = starter.elapsed_time(ender)

            # Measures query step
            torch.cuda.synchronize()
            starter.record()
            # Core update step
            with torch.no_grad():
                zt = model.encoder._preprocess_targets(xt, dy)
                pred_dist = model.likelihood(model.decoder(model.encoder.query(zt, kv_cache), xt))
            # Measures time and memory
            ender.record()
            torch.cuda.synchronize()
            query_time_ms = starter.elapsed_time(ender)

            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            # Writes measured results
            if j >= burn_in:
                write_j = j - burn_in
                condition_kv[write_j, ctx_inc] = condition_time_ms
                query_kv[write_j, ctx_inc] = query_time_ms
                memory_kv[write_j, ctx_inc] = peak_memory_mb
            ctx_inc += 1

    # Aggregates results
    memory_kv_std = memory_kv.std(axis=0, ddof=1)
    memory_no_kv_std = memory_no_kv.std(axis=0, ddof=1)
    runtime_no_kv = np.mean(runtime_no_kv, axis=0)
    condition_kv = np.mean(condition_kv, axis=0)
    query_kv = np.mean(query_kv, axis=0)
    memory_kv = np.mean(memory_kv, axis=0)
    memory_no_kv = np.mean(memory_no_kv, axis=0)
    runtime_kv = condition_kv + query_kv

    # Writes results to file
    summary_block = f"""
    ----------------------------
    Cumulative Context Size: {upper_ctxs}
    Runtime Incremental no KV (ms): {runtime_no_kv}
    Query Runtime Incremental KV (ms): {query_kv}
    Condition Runtime Incremental KV (ms): {condition_kv}
    Runtime Incremental KV (ms): {runtime_kv}
    Memory Cumulative KV (Mb): {memory_kv}
    Memory Cumulative No KV (Mb): {memory_no_kv}
    """
    print(summary_block)
    with open('experiments/plot_results/kv/' + f'comparison_kv_vs_none_{strategy}-{targets}.txt', 'w') as file_object:
        file_object.write(summary_block)

    # Plots runtime results
    runtime_file_name = f'experiments/plot_results/kv/kv_without_runtime_{strategy}_{targets}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(upper_ctxs, runtime_no_kv, label='No Caching')
    ax.plot(upper_ctxs, runtime_kv, label='KV Caching (Condition + Query)')
    ax.plot(upper_ctxs, condition_kv, label='Condition (KV Caching)', linestyle="dashed")
    ax.plot(upper_ctxs, query_kv, label='Query (KV Caching)', linestyle="dashed")
    ax.set_xlabel('Context Size')
    ax.set_ylabel('Runtime (ms)')
    ax.legend()
    tit_text = f'NT={nt}' if strategy == "fixed" else 'NT=NC'
    ax.set_title(f'Runtime as Context Size Increases ({tit_text})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(runtime_file_name, dpi=300)

    # Plots memory
    memory_file_name = f'experiments/plot_results/kv/kv_without_memory_{strategy}_{targets}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(upper_ctxs, memory_no_kv, label='No Caching')
    confidence_bars = False
    if confidence_bars:
        ci95 = 1.96 * memory_kv_std / np.sqrt(aggregate_over)
        ax.fill_between(upper_ctxs,
                        memory_no_kv - ci95,
                        memory_no_kv + ci95,
                        alpha=0.25)
    ax.plot(upper_ctxs, memory_kv, label='KV Caching (Condition + Query)')
    if confidence_bars:
        ci95 = 1.96 * memory_kv / np.sqrt(aggregate_over)
        ax.fill_between(upper_ctxs,
                        memory_kv - ci95,
                        memory_kv + ci95,
                        alpha=0.25)
    ax.set_xlabel('Context Size')
    ax.set_ylabel('Cumulative Memory Usage (MB)')
    ax.set_title(f'Memory Usage as Context Size Increases ({tit_text})')
    ax.legend()
    tit_text = f'NT={nt}' if strategy == "fixed" else 'NT=NC'
    ax.set_title(f'Memory Usage as Context Size Increases ({tit_text})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(memory_file_name, dpi=300)


# Compares decoding time w / w out KV caching
@torch.no_grad
def compare_decoding_times():
    # Compare params
    burn_in = 1 # Number of burn in runs to ignore
    aggregate_over = 1 # Number of runs to aggregate data over
    token_step = 5 # How many increments of tokens to go up in
    min_nc = 1
    max_nc = 100
    dx, dy, m = 1, 1, 1
    # End of params
    device='cuda'
    model = get_model('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', device=device)
    model.eval()
    context_sizes = np.arange(start=min_nc, stop=max_nc, step=token_step, dtype=int)
    #print(context_sizes)
    runtime = np.zeros((aggregate_over, len(context_sizes)))
    memory = np.zeros((aggregate_over, len(context_sizes)))
    max_high = 2
    xcs = (torch.rand((m, max_nc, dx), device=device) * max_high * 2) - max_high
    ycs = (torch.rand((m, max_nc, dy), device=device) * max_high * 2) - max_high

    runtime_no_kv = np.zeros((aggregate_over, len(context_sizes)))
    runtime_kv = np.zeros((aggregate_over, len(context_sizes)))
    memory_kv = np.zeros((aggregate_over, len(context_sizes)))
    memory_no_kv = np.zeros((aggregate_over, len(context_sizes)))
    # Measures memory and runtime for model with no KV cache
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    i = 0
    for ctx_size in tqdm(context_sizes, desc="Comparing context sizes"):
        xc = xcs[:, :ctx_size, :]
        yc = ycs[:, :ctx_size, :]
        for j in range(burn_in + aggregate_over):
            # Measures kv runtime and memory
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            starter.record()
            with torch.no_grad():
                xc_ordered, yc_ordered = model.kv_cached_greedy_variance_ctx_builder(xc, yc)
            ender.record()
            torch.cuda.synchronize()
            peak_memory_mb_kv = torch.cuda.max_memory_allocated() / (1024 * 1024)
            update_time_ms_kv = starter.elapsed_time(ender)

            # Measures no kv runtime and memory
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            starter.record()
            with torch.no_grad():
                xc_ordered, yc_ordered = model.greedy_variance_ctx_builder(xc, yc)
            ender.record()
            torch.cuda.synchronize()
            peak_memory_mb_no_kv = torch.cuda.max_memory_allocated() / (1024 * 1024)
            update_time_ms_no_kv = starter.elapsed_time(ender)

            # Records results
            write_idx = j - burn_in
            if write_idx >= 0:
                runtime_no_kv[write_idx, i] = update_time_ms_no_kv
                memory_no_kv[write_idx, i] = peak_memory_mb_no_kv
                runtime_kv[write_idx, i] = update_time_ms_kv
                memory_kv[write_idx, i] = peak_memory_mb_kv
        i += 1

    # Aggregates results
    memory_no_kv_std = memory_no_kv.std(axis=0, ddof=1)
    memory_kv_std = memory_kv.std(axis=0, ddof=1)
    runtime_kv_std = runtime_kv.std(axis=0, ddof=1)
    runtime_no_kv_std = runtime_no_kv.std(axis=0, ddof=1)

    runtime_no_kv = np.mean(runtime_no_kv, axis=0)
    memory_no_kv = np.mean(memory_no_kv, axis=0)
    runtime_kv = np.mean(runtime_kv, axis=0)
    memory_kv = np.mean(memory_kv, axis=0)


    # Plots memory
    memory_file_name = f'experiments/plot_results/kv/decoding_kv_vs_none.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(context_sizes, memory_no_kv, label='No Caching')
    confidence_bars = False
    if confidence_bars:
        ci95 = 1.96 * memory_no_kv_std / np.sqrt(aggregate_over)
        ax.fill_between(context_sizes,
                        memory_no_kv - ci95,
                        memory_no_kv + ci95,
                        alpha=0.25)
    ax.plot(context_sizes, memory_kv, label='KV Caching')
    if confidence_bars:
        ci95 = 1.96 * memory_kv_std / np.sqrt(aggregate_over)
        ax.fill_between(context_sizes,
                        memory_kv - ci95,
                        memory_kv + ci95,
                        alpha=0.25)
    ax.set_xlabel('Context Size')
    ax.set_ylabel('Peak Memory Usage (MB)')
    ax.legend()
    ax.set_title(f'Greedy Ordering Memory Usage as Context Size Increases')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(memory_file_name, dpi=300)

    # Plots runtime
    memory_file_name = f'experiments/plot_results/kv/runtime_decoding_kv_vs_none.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(context_sizes, runtime_no_kv, label='No Caching')
    confidence_bars = False
    if confidence_bars:
        ci95 = 1.96 * runtime_no_kv_std / np.sqrt(aggregate_over)
        ax.fill_between(context_sizes,
                        runtime_no_kv - ci95,
                        runtime_no_kv + ci95,
                        alpha=0.25)
    ax.plot(context_sizes, runtime_kv, label='KV Caching')
    if confidence_bars:
        ci95 = 1.96 * runtime_kv_std / np.sqrt(aggregate_over)
        ax.fill_between(context_sizes,
                        runtime_kv - ci95,
                        runtime_kv + ci95,
                        alpha=0.25)
    ax.set_xlabel('Context Size')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title(f'Greedy Ordering Runtime as Context Size Increases')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(memory_file_name, dpi=300)


if __name__ == "__main__":
    compare_decoding_times()
    #test_kv_cache()
    #measure_condition_time_memory_kv()
    #compare_kv_against_none(strategy="fixed", targets=128)
    #compare_kv_against_none(strategy="scale")
    #compare_kv_against_none(strategy="fixed", targets=512)
    #compare_kv_against_none(strategy="fixed", targets=2048)
    #compare_kv_against_none(strategy="fixed", targets=10_000)
    #compare_kv_against_none(strategy="fixed", targets=100_000)
