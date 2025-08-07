# Little file to measure tnpa speedups obtained
from plot_adversarial_perms import get_model
from tnp.utils.np_functions import np_pred_fn
import torch
import numpy as np
from tnp.data.base import Batch
from tnp.models.tnpa_opt import make_tnpa_opt_from_tnpa


# Measures TNP-A vs the kv cached version for unrolling
def measure_speeds():
    # Hypers
    burn_in = 1
    aggregate_over = 1
    s_nc_nt_list = [(10_000, 3, 2)]
    device="cuda"
    m, dy, dx = 1, 1, 1
    # End of hypers
    tnpa_model = get_model('experiments/configs/synthetic1dRBF/gp_tnpa_rangesame.yml', 'pm846-university-of-cambridge/tnpa-rbf-rangesame/model-e6yry1ri:v199', seed=False, device=device)
    tnpa_model.permute = False
    tnpa_model.eval()
    opt_tnpa_model = make_tnpa_opt_from_tnpa(tnpa_model)
    opt_tnpa_model.permute = False
    opt_tnpa_model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for (samples, nc, nt) in s_nc_nt_list:
        tnpa_model.num_samples = samples
        opt_tnpa_model.num_samples = samples
        max_high = 2
        xc = (torch.rand((m, nc, dx), device=device) * max_high * 2) - max_high
        yc = (torch.rand((m, nc, dy), device=device) * max_high * 2) - max_high
        xt = (torch.rand((m, nt, dx), device=device) * max_high * 2) - max_high
        yt = (torch.rand((m, nt, dy), device=device) * max_high * 2) - max_high
        batch = Batch(xc=xc, yc=yc, xt=xt, yt=yt, x=None, y=None)
        runtimes_normal = []
        runtimes_opt = []
        for j in range(burn_in + aggregate_over):
            # KV TNP-A
            torch.cuda.synchronize()
            starter.record()
            with torch.no_grad():
                pred_dist_opt = np_pred_fn(opt_tnpa_model, batch, predict_without_yt_tnpa=True)
            ender.record()
            torch.cuda.synchronize()
            runtime_ms_opt = starter.elapsed_time(ender)
            # TNP-a
            torch.cuda.synchronize()
            starter.record()
            with torch.no_grad():
                pred_dist_normal = np_pred_fn(tnpa_model, batch, predict_without_yt_tnpa=True)
            ender.record()
            torch.cuda.synchronize()
            runtime_ms_normal = starter.elapsed_time(ender)

            if j - burn_in >= 0: 
                runtimes_normal.append(runtime_ms_normal)
                runtimes_opt.append(runtime_ms_opt)
        # Prints results
        runtime_norm = np.mean(runtimes_normal)
        runtime_opt = np.mean(runtimes_opt)
        print('---------------------------------------------------')
        print(f'TNP-A s={samples} nc={nc} nt={nt} m={m} runtime={runtime_norm}')
        print(f'KV TNP-A s={samples} nc={nc} nt={nt} m={m} runtime={runtime_opt}')
        print(pred_dist_normal.mean[:, :5, :])
        print(pred_dist_opt.mean[:, :5, :])
        print("****")
        print(pred_dist_normal.stddev[:, :5, :])
        print(pred_dist_opt.stddev[:, :5, :])



if __name__ == "__main__":
    measure_speeds()