import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.deepset import DeepSet
from .base import ConditionalNeuralProcess
from .tnp import TNPDecoder
from .incUpdateBase import IncUpdateEff
import torch.distributions as td


class CNPEncoder(nn.Module):
    def __init__(
        self,
        deepset: DeepSet,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.deepset = deepset
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, .]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        yc_encoded = self.y_encoder(yc)

        zc = self.deepset(xc_encoded, yc_encoded)

        # Use same context representation for every target point.
        zc = einops.repeat(zc, "m d -> m n d", n=xt.shape[-2])

        # Concatenate xt to zc.
        zc = torch.cat((zc, xt_encoded), dim=-1)

        return zc


class CNP(ConditionalNeuralProcess, IncUpdateEff):
    def __init__(
        self,
        encoder: CNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


    # Effecient incremental updates should only be used for hadIsd where this results in measurable speedup
    def init_inc_structs(self, m: int, max_nc: int, device: str, use_flash: bool=False):
        if self.encoder.deepset.agg_strat_str != "mean" and self.encoder.deepset.agg_strat_str!= "sum":
            raise ValueError("Only mean and sum CNP inc supported atm")

        self.inc_cache = {}
        self.inc_cache["n_points"] = None
        self.inc_cache["running_sum"] = None

    # Adds new context points
    def update_ctx(self, xc: torch.Tensor, yc: torch.Tensor, use_flash: bool=False):
        xc_encoded = self.encoder.x_encoder(xc)
        yc_encoded = self.encoder.y_encoder(yc)
        z = torch.cat((xc_encoded, yc_encoded), dim=-1)
        z = self.encoder.deepset.z_encoder(z)

        _, n_new, _ = xc.shape
        sum_new = torch.nansum(z, dim=-2) # [m, dz]
        m, dz = sum_new.shape
        # Inits tensors for first time lazily
        if self.inc_cache["running_sum"] is None: self.inc_cache["running_sum"] = torch.zeros((m, dz), device=xc.device)
        if self.inc_cache["n_points"] is None: self.inc_cache["n_points"] = torch.zeros((m, 1), device=xc.device)

        self.inc_cache["n_points"] += n_new
        self.inc_cache["running_sum"] += sum_new

    def query(self, xt: torch.Tensor, dy: int, use_flash: bool=False) -> td.Normal:
        xt_encoded = self.encoder.x_encoder(xt)

        zc = self.inc_cache["running_sum"]
        if self.encoder.deepset.agg_strat_str == "mean": zc /= self.inc_cache["n_points"]
        zc = einops.repeat(zc, "m d -> m n d", n=xt.shape[-2])
        zc = torch.cat((zc, xt_encoded), dim=-1)
        return self.likelihood(self.decoder(zc, xt))

