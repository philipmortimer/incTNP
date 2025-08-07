from data_temp.data_processing.elevations import get_cached_elevation_grid
from tnp.utils.np_functions import np_pred_fn
from plot_hadISD import plot_hadISD
from tnp.data.hadISD import HadISDDataGenerator, scale_pred_temp_dist, get_true_temp, HadISDBatch
import torch
from tnp.utils.data_loading import adjust_num_batches
import numpy as np
from plot_adversarial_perms import get_model
import os


def get_had_testset_and_plot_stuff():
    # Change these for correct machine / directory
    data_directory = "/scratch/pm846/TNP/data/data_processed/test"
    dem_path = "/scratch/pm846/TNP/data/elev_data/ETOPO_2022_v1_60s_N90W180_surface.nc"
    cache_dem_dir = "/scratch/pm846/TNP/data/elev_data/"

    #data_directory = "/home/pm846/rds/hpc-work/Thesis/TNP-Inc/experiments/data_temp/csd3_all_data/downloaded/other/data_processed/test"
    #dem_path = "/home/pm846/rds/hpc-work/Thesis/TNP-Inc/experiments/data_temp/csd3_all_data/downloaded/other/elev_data/ETOPO_2022_v1_60s_N90W180_surface.nc"
    #cache_dem_dir = "/home/pm846/rds/hpc-work/Thesis/TNP-Inc/experiments/data_temp/csd3_all_data/downloaded/other/elev_data/"
    num_grid_points_plot = 200
    # Normal hypers
    min_nc = 1
    max_nc = 2033
    nt = 250
    samples_per_epoch= 4_000 # 80_000
    batch_size = 1
    deterministic = True
    ordering_strategy = "random"
    num_val_workers = 2

    # Loads had dataset
    gen_test = HadISDDataGenerator(min_nc=min_nc, max_nc=max_nc, nt=nt, ordering_strategy=ordering_strategy,
        samples_per_epoch=samples_per_epoch, batch_size=batch_size, data_directory=data_directory,deterministic=deterministic)
    
    # Wraps data set in a proper torch set loader for less IO bottlenecking
    test_loader = torch.utils.data.DataLoader(
       gen_test,
        batch_size=None,
        num_workers=num_val_workers,
        worker_init_fn=(
            (
                adjust_num_batches
            )
            if num_val_workers > 0
            else None
        ),
        persistent_workers=True if num_val_workers > 0 else False,
        pin_memory=True,
    )

    # Loads elevation data from DEM file
    lat_mesh, lon_mesh, elev_np = get_cached_elevation_grid(gen_test.lat_range, gen_test.long_range,
        num_grid_points_plot, cache_dem_dir,
        dem_path)

    return test_loader, lat_mesh, lon_mesh, elev_np

def get_model_list():
    # List of models to compare
    tnp_plain = ('experiments/configs/hadISD/had_tnp_plain.yml',
        'pm846-university-of-cambridge/plain-tnp-had/model-o20d6s1q:v99', 'TNP-D')
    incTNP = ('experiments/configs/hadISD/had_incTNP.yml', 
        'pm846-university-of-cambridge/mask-tnp-had/model-9w1vbqjh:v99', 'incTNP')
    batchedTNP = ('experiments/configs/hadISD/had_incTNP_batched.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-had/model-z5nlguxq:v99', 'incTNP-Batched')
    priorBatched = ('experiments/configs/hadISD/had_incTNP_priorbatched.yml',
        'pm846-university-of-cambridge/mask-priorbatched-tnp-had/model-83h4gpp2:v99', 'incTNP-Batched (Prior)')
    lbanp =('experiments/configs/hadISD/had_lbanp.yml',
        'pm846-university-of-cambridge/lbanp-had/model-zyzq4mno:v99', 'LBANP',)
    cnp = ('experiments/configs/hadISD/had_cnp.yml',
        'pm846-university-of-cambridge/cnp-had/model-suqmhf9v:v99', 'CNP')
    conv_cnp = ('experiments/configs/hadISD/had_convcnp.yml',
        'pm846-university-of-cambridge/convcnp-had/model-p4f775ey:v98', 'ConvCNP (50 x 50)')
    conv_cnp_100 = ('experiments/configs/hadISD/alt_variants/had_big_convcnp.yml',
        'pm846-university-of-cambridge/convcnp-had/model-ecytkrfq:v99', 'ConvCNP (100 x 100)')
    conv_cnp_125 = ('experiments/configs/hadISD/alt_variants/had_125_convcnp.yml',
        'pm846-university-of-cambridge/convcnp-had/sa1sz4c9:v99', 'ConvCNP (125 x 125)')
    conv_cnp_150 = ('experiments/configs/hadISD/alt_variants/had_between_convcnp.yml',
        'pm846-university-of-cambridge/convcnp-had/s8gzetnn:v99', 'ConvCNP (150 x 150)')
    all_models = [tnp_plain, incTNP, batchedTNP, priorBatched, lbanp, cnp, conv_cnp, conv_cnp_100]
    return all_models

def plot_models():
    folder_name = "experiments/plot_results/had/plots/"
    huge_grid_plots = True
    order="random"
    no_kernels = 200
    device="cuda"

    data, lat_mesh, lon_mesh, elev_np = get_had_testset_and_plot_stuff()
    models = get_model_list()
    batches_plot = []
    for i, batch in enumerate(data):
        batches_plot.append(batch)
        if i >= no_kernels: break

    for (model_yml, model_wab, model_name) in models:
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        model_folder = f"{folder_name}/{model_name}"
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        
        plot_hadISD(
            model=model,
            batches=batches_plot,
            num_fig=len(batches_plot),
            name=model_folder+f"/{model_name}",
            pred_fn=np_pred_fn,
            lat_mesh=lat_mesh,
            lon_mesh=lon_mesh,
            elev_np=elev_np,
            savefig=True, 
            logging=False,
            model_lbl=f"{model_name}",
            huge_grid_plots=huge_grid_plots,
            device=device,
        )

if __name__ == "__main__":
    plot_models()