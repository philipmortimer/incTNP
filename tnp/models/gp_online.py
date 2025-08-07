# A streamed Gaussian Process to compare to TNPs and provide a baseline
# Hyperparameters are updated in chunks and then exact GP conditioned on whole data is updated
import torch
from torch import nn
import gpytorch
from check_shapes import check_shapes
from typing import Callable, Optional, Literal
import numpy as np


class ExactGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
        batch_shape: torch.Size,
    ):
        super().__init__(
            train_inputs=torch.zeros(1,1), # Dummy inputs
            train_targets=torch.zeros(1), # Dummy targets
            likelihood=likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = kernel

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# GP Stream class that handles logic of streaming context points to be used with np_pred_fun easily. Uses gaussian likelihood
class GPStream(nn.Module):
    def __init__(
        self,
        kernel_name: Literal["rbf"],
        lr: float, # LR for grad updates
        n_steps: int, # Number of grad steps per update
        chunk_size: int, # Size of chunks to be streamed in to model
        train_strat: Literal["Expanding", "Sliding"], # Whether to use an ever expanding window or a sliding one
        convergence_tolerance: float, # Change in loss between update steps before classed as converged
        use_adam: bool,
        device: str,
    ):
        super().__init__()
        self.lr = lr
        self.n_steps = n_steps
        self.chunk_size = chunk_size
        self.device = device
        self.kernel_name = kernel_name
        self.strategy = train_strat
        self.tol = convergence_tolerance
        self.use_adam = use_adam

    # Streams data in chunks and updates the hypers of the gp for the given example. May want to alter this
    # eg do we want to see data update then continue or do we want to keep expanding the window (alleviates catastrophic forgetting but may fit too well)
    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, 1]"
    )
    def _stream_through_gp(self, gp, likelihood, mll, xc, yc):
        m, nc, dx = xc.shape
        likelihood.train()
        gp.train()
        for i in range(0, nc, self.chunk_size):
            end = min(i + self.chunk_size, nc)
            cur_chunk_size = end - i

            if self.strategy == "Expanding":
                xc_sub = xc[:,:end,:].reshape(m, -1, dx) # [m, end, dx]
                yc_sub = yc[:,:end,:].reshape(m, -1) # [m, end]
            elif self.strategy == "Sliding":
                xc_sub = xc[:,i:end,:].reshape(m, -1, dx) # [m, cur_chunk_size, dx]
                yc_sub = yc[:,i:end,:].reshape(m, -1) # [m, cur_chunk_size]
            else: raise ValueError(f"Incorrect strategy: {self.strategy}")


            gp.set_train_data(inputs=xc_sub, targets=yc_sub, strict=False)
            if self.use_adam:
                # Performs update steps for hyper
                optimiser = torch.optim.Adam(gp.parameters(), lr=self.lr) # Creates new optimiser (to be properly retraining)
                prev_loss = None
                converged=False
                for train_step in range(self.n_steps):
                    optimiser.zero_grad()
                    loss = -mll(gp(xc_sub), yc_sub)
                    loss_sum = loss.sum()
                    loss_sum.backward()
                    optimiser.step()
                    if prev_loss is not None:
                        mean_diff = torch.abs(prev_loss - loss).mean().item()
                        max_diff = torch.abs(prev_loss - loss).max().item()
                        #print(f"Step {mean_diff:.10f}")
                    # Convergence check
                    if prev_loss is not None and torch.all(torch.abs(prev_loss - loss) < self.tol): 
                        converged = True
                        break
                    prev_loss = loss.detach()
                if not converged:
                    if True: print(f"GP Training not converged divergence max {max_diff:.10f} mean diff {mean_diff:.10f} tol {self.tol:.10f}")
                    #print(likelihood.noise[:5])
            else:
                optimiser = torch.optim.LBFGS(gp.parameters(), lr=self.lr, max_iter=self.n_steps)
                def closure():
                    optimiser.zero_grad()
                    loss = -mll(gp(xc_sub), yc_sub)
                    loss_sum = loss.sum()
                    loss_sum.backward()
                    return loss_sum
                
                optimiser.step(closure)
            
        likelihood.eval()
        gp.eval()

    
    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, 1]", "xt: [m, nt, dx]"
    )
    def forward(self, xc, yc, xt) -> gpytorch.distributions.MultitaskMultivariateNormal:
        xc, yc, xt = xc.to(self.device), yc.to(self.device), xt.to(self.device)
        m, nc, dx = xc.shape
        _, nt, _ = xt.shape

        # Constructs batched GP and optimiser
        if self.kernel_name == "rbf":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([m]), lengthscale_constraint=gpytorch.constraints.Interval(0.25, 1.0)),batch_shape=torch.Size([m]))
        else: raise ValueError(f"Invalid kernel: {self.kernel_name}")
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([m])).to(self.device)
        # Noise prior - TNP likely learns true noise over training so probably fair to give GP this info
        noise_prior = True
        if noise_prior:
            likelihood.noise = torch.full((m,), 0.01, device=self.device)
            likelihood.register_prior(
                "noise_prior",
                gpytorch.priors.LogNormalPrior(
                    loc=np.log(0.01),
                    scale=0.1,
                ),
                "noise"
            )
        gp = ExactGP(likelihood=likelihood, kernel=kernel, batch_shape=torch.Size([m])).to(self.device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        self._stream_through_gp(gp, likelihood, mll, xc, yc)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            function_dist = gp(xt)
            pred_dist = likelihood(function_dist)
        return pred_dist
        #return gpytorch.distributions.MultitaskMultivariateNormal.from_batch(pred_dist)


# RBF GP Stream - lr =0.05 and n_steps=50 work well for adam
class GPStreamRBF(GPStream):
    def __init__(
        self,
        chunk_size: int,
        lr: float = 0.05, # LR for grad updates
        n_steps: int = 50, # Number of grad steps per update
        use_adam: bool = True, # Whether to use adam or LBFGS
        train_strat: Literal["Expanding", "Sliding"] = "Expanding",
        convergence_tolerance: float=1e-5, 
        device: str = "cuda",
    ):
        super().__init__(kernel_name="rbf",
            chunk_size=chunk_size, lr=lr, n_steps=n_steps, device=device, train_strat=train_strat, 
            convergence_tolerance=convergence_tolerance, use_adam=use_adam)